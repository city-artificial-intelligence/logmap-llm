import argparse
import os
import sys
import tomllib
import pandas as pd
import json
from datetime import datetime, timezone

import oracle_prompt_building as opb
from log_utils import TeeWriter, error, warning, warn, info, success

# column names duplicated from bridging.py since bridging imports JPype java types at module level (requires a running JVM)  
# this subprocess is isolated from the JVM specifically to avoid conflicts with owlready2

column_source_entity_uri = 'source_entity_uri'
column_target_entity_uri = 'target_entity_uri'
column_relation = 'relation'
column_confidence = 'confidence'
column_entityType = 'entityType'

def get_m_ask_column_names():
  column_names = [
      column_source_entity_uri,
      column_target_entity_uri,
      column_relation,
      column_confidence,
      column_entityType
  ]
  return column_names


parser = argparse.ArgumentParser(
    description="LogMap-LLM: Stage 2 prompt building (subprocess)"
)

parser.add_argument(
    "--config", "-c",
    type=str,
    required=True,
    help="path to the TOML configuration file",
)
parser.add_argument(
    "--reuse-align",
    action="store_true",
    default=False,
    help="override config to reuse existing LogMap alignment",
)
parser.add_argument(
    "--reuse-prompts",
    action="store_true",
    default=False,
    help="override config to reuse existing prompts; implies --reuse-align",
)
parser.add_argument(
    "--track",
    type=str,
    default=None,
    choices=["conference", "bioml", "anatomy", "knowledgegraph"],
    help="track identifier (forwarded from parent process)",
)
parser.add_argument(
    "--no-cache",
    action="store_true",
    default=False,
    help="disable owlready2 quadstore caching (parse ontologies from scratch)",
)

args = parser.parse_args()

config_path = args.config

if not os.path.isfile(config_path):
    error(f"Configuration file not found: {config_path}")
    sys.exit(1)

with open(config_path, mode="rb") as fp:
    config = tomllib.load(fp)

# applies CLI overrides
if args.reuse_prompts:
    config['pipeline']['build_oracle_prompts'] = 'reuse'
    config['pipeline']['align_ontologies'] = 'reuse'
elif args.reuse_align:
    config['pipeline']['align_ontologies'] = 'reuse'

info(f"Configuration loaded from: {config_path}")

# resolve track
track = args.track or config.get('alignmentTask', {}).get('track', None)
if track:
    info(f"  Track: {track}")

task_name = config['alignmentTask']['task_name']
onto_src_filepath = config['alignmentTask']['onto_source_filepath']
onto_tgt_filepath = config['alignmentTask']['onto_target_filepath']

logmap_outputs_dir_path = config['outputs']['logmap_initial_alignment_output_dirpath']

# initialise timing and logging & determine output dir
run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

results_dir = config['outputs']['logmapllm_output_dirpath']
os.makedirs(results_dir, exist_ok=True)

# start the tee-logger --all subsequent stdout goes to both terminal and log file
log_filename = f"pipeline_log_{run_timestamp}.txt"
log_filepath = os.path.join(results_dir, log_filename)
original_stdout = sys.stdout
tee = TeeWriter(log_filepath, original_stdout)
sys.stdout = tee

print()
info(f"Pipeline log file: {log_filepath}")

info('Reusing existing initial LogMap alignment ...')
print()
filename = task_name + '-logmap_mappings.txt'
filepath = os.path.join(logmap_outputs_dir_path, filename)
mappings = pd.read_csv(filepath, sep='|', header=None)
info(f'Number of mappings in initial alignment: {len(mappings)}')
print()
filename = task_name + '-logmap_mappings_to_ask_oracle_user_llm.txt'
print('Loading mappings to ask an Oracle from file:')
print(filename)
filepath = os.path.join(logmap_outputs_dir_path, filename)

# handle empty M_ask: LogMap may produce no uncertain mappings for small ontology pairs (this is common in Conference track)
if os.path.getsize(filepath) == 0:
    warning("M_ask file is empty — no uncertain mappings to build prompts for.")
    warning("Exiting Stage 2 subprocess (nothing to do).")
    sys.stdout = original_stdout
    tee.close()
    sys.exit(0)

m_ask_df = pd.read_csv(filepath, sep='|', header=None)
m_ask_df.columns = get_m_ask_column_names()

oupt_name = config['oracle']['oracle_user_prompt_template_name']
bidirectional_mode = opb.is_bidirectional_template(oupt_name)

# apply answer format configuration to prompt templates... this must happen before any prompt functions are called
answer_format = config['oracle'].get('answer_format', 'true_false')
if answer_format != 'true_false':
    from oracle_user_prompt_templates import set_answer_format
    set_answer_format(answer_format)
    info(f'Answer format set to: {answer_format}')

# apply ontology domain qualifier to prompt templates... 
# this must happen before any prompt functions are called

# maps track -> domain string (
# e.g., "conference" -> "conference", "bioml" -> "biomedical", "knowledgegraph" -> "knowledge graph")

from oracle_user_prompt_templates import set_ontology_domain
set_ontology_domain(track)
if track:
    from oracle_user_prompt_templates import get_ontology_domain
    info(f'Ontology domain set to: {get_ontology_domain()}')

# initialise SiblingSelector if the template requires sibling context

sibling_selector = None
if opb.is_sibling_template(oupt_name):
    try:
        from sibling_retrieval import SiblingSelector, DEFAULT_GENERAL_MODEL
        if track == 'conference':
            info(f'Initialising SiblingSelector (general model) for conference track ...')
            sibling_selector = SiblingSelector(model_name_or_path=DEFAULT_GENERAL_MODEL, pooling="mean")
        elif track == 'knowledgegraph':
            info(f'Initialising SiblingSelector (general model) for KG track ...')
            sibling_selector = SiblingSelector(model_name_or_path=DEFAULT_GENERAL_MODEL, pooling="mean")
        else:
            info('Initialising SiblingSelector (SapBERT) for sibling-aware template ...')
            sibling_selector = SiblingSelector()
        success(f'SiblingSelector ready (device: {sibling_selector.device})')
    except Exception as e:
        warning(f'SiblingSelector initialisation failed: {e}')
        warn('Falling back to alphabetical sibling selection.')
    print()

# resolve property prompt function (Conference track)

property_prompt_function = None
if track == 'conference':
    prop_template_name = config['oracle'].get('oracle_property_prompt_template_name', 'prop_labels_only')
    property_prompt_function = opb.get_property_prompt_template_function(prop_template_name)
    if property_prompt_function:
        info(f'Property prompt template: {prop_template_name}')
    else:
        warn(f'Property prompt template "{prop_template_name}" not found — property mappings will be skipped')
    print()

# resolve instance prompt function (KG track)

instance_prompt_function = None
if track == 'knowledgegraph':
    inst_template_name = config['oracle'].get('oracle_instance_prompt_template_name', 'inst_labels_only')
    instance_prompt_function = opb.get_instance_prompt_template_function(inst_template_name)
    if instance_prompt_function:
        info(f'Instance prompt template: {inst_template_name}')
    else:
        warn(f'Instance prompt template "{inst_template_name}" not found — instance mappings will be skipped')
    print()

info('Building fresh Oracle user prompts ...')
if bidirectional_mode:
    print(f' (bidirectional mode: two prompts per candidate)')
print()

# load ontologies once — shared by prompt building and few-shot building

info('Loading ontologies ...')
cache_dir = None if args.no_cache else os.path.expanduser('~/.cache/logmap-llm/owlready2')
OA_source, OA_target = opb.load_ontologies(onto_src_filepath, onto_tgt_filepath, cache_dir=cache_dir)
success('Ontologies loaded.')
print()

# ontology loading + prompt construction:

if bidirectional_mode:
    m_ask_oracle_user_prompts, n_equiv_cands, n_non_equiv_skipped = (
        opb.build_oracle_user_prompts_bidirectional(
            oupt_name, onto_src_filepath, onto_tgt_filepath, m_ask_df,
            OA_source=OA_source, OA_target=OA_target,
            sibling_selector=sibling_selector
        )
    )
else:
    m_ask_oracle_user_prompts = opb.build_oracle_user_prompts(
        oupt_name, onto_src_filepath, onto_tgt_filepath, m_ask_df,
        OA_source=OA_source, OA_target=OA_target,
        sibling_selector=sibling_selector,
        property_prompt_function=property_prompt_function,
        instance_prompt_function=instance_prompt_function
    )

if m_ask_oracle_user_prompts is not None:
    print()
    info(f"Number of LLM Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}")
    print()

if config['pipeline']['build_oracle_prompts'] == 'build':
    # save the newly built oracle user prompts to a .json file so they can be reused
    dirpath = config['outputs']['logmapllm_output_dirpath']
    filename = task_name + '-' + oupt_name + '-mappings_to_ask_oracle_user_prompts.json'
    print('LLM Oracle user prompts saved to file:')
    print(filename)
    filepath = os.path.join(dirpath, filename)
    with open(filepath, 'w') as fp:
        json.dump(m_ask_oracle_user_prompts, fp)

# --------------------------
# Few-shot example building
# --------------------------

few_shot_k = config['oracle'].get('few_shot_k', 0)
_anchor_tmp_path = None  # track temp file for cleanup

if few_shot_k > 0:
    train_path = config.get('evaluation', {}).get('train_alignment_path', None)

    if train_path is None or not os.path.isfile(str(train_path)):
        info("No train.tsv found — deriving few-shot positives from LogMap high-confidence anchors")

        # build M_ask pair set for subtraction
        m_ask_pairs = set()
        for _, row in m_ask_df.iterrows():
            m_ask_pairs.add((row[column_source_entity_uri], row[column_target_entity_uri]))

        # IMPORTANT:
        # filter initial alignment to anchors not in M_ask to avoid data leakage
        # for KG track: filter to INST entities (instance matching); for all other tracks, filter to CLS entities (i.e., class matching)
        # ...mixing entity types with the wrong template would produce incoherent prompts
        # initial alignment columns: 0=src, 1=tgt, 2=relation, 3=conf, 4=entityType
        
        if track == 'knowledgegraph':
            anchor_entity_type = "INST"
        else:
            anchor_entity_type = "CLS"
        
        info(f"  Filtering anchors to {anchor_entity_type} entities")

        anchors = []
        
        for _, row in mappings.iterrows():
            src, tgt = row.iloc[0], row.iloc[1]
            entity_type = row.iloc[4] if len(row) > 4 else "CLS"
            if entity_type != anchor_entity_type:
                continue
            if (src, tgt) not in m_ask_pairs:
                anchors.append((src, tgt))

        if len(anchors) == 0:
            warn(f"No {anchor_entity_type} anchors available for few-shot examples; "
                 "falling back to zero-shot")
            few_shot_k = 0  # prevent downstream build attempt
        else:
            info(f"  Anchor pool: {len(anchors)} {anchor_entity_type} pairs "
                 f"(from {len(mappings)} initial - {len(m_ask_pairs)} M_ask)")

            # write to a temporary TSV that FewShotExampleBuilder can read
            import tempfile
            anchor_fd, anchor_path = tempfile.mkstemp(suffix='.tsv', prefix='anchors_')
            os.close(anchor_fd)
            with open(anchor_path, 'w') as f:
                for src, tgt in anchors:
                    f.write(f"{src}\t{tgt}\n")
            _anchor_tmp_path = anchor_path
            train_path = anchor_path

    # build few-shot examples (runs for both train.tsv and anchor-derived paths)
    if few_shot_k > 0 and train_path is not None:
        info(f'Building {few_shot_k} few-shot examples (strategy: {config["oracle"].get("few_shot_negative_strategy", "hard")}) ...')

        from functools import partial
        from few_shot_examples import build_few_shot_examples

        # get the bound prompt function matching the anchor entity typ
        # for KG track, use the instance template; for all others, use the class template (same as used for main prompt building)
        
        if track == 'knowledgegraph' and instance_prompt_function is not None:
            prompt_function = instance_prompt_function
        else:
            prompt_function = opb.get_oracle_user_prompt_template_function(oupt_name)
            if sibling_selector is not None and opb.is_sibling_template(oupt_name):
                prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

        few_shot_seed = config['oracle'].get('few_shot_seed', 42) # default seed: 42 (override in toml config)
        negative_strategy = config['oracle'].get('few_shot_negative_strategy', 'hard')

        # for hard/hard-similar negatives, initialise SiblingSelector if not already done
        # note: hard-similar not yet fully tested
        
        fs_sibling_selector = sibling_selector
        if negative_strategy in ('hard', 'hard-similar') and fs_sibling_selector is None:
            try:
                from sibling_retrieval import SiblingSelector, DEFAULT_GENERAL_MODEL
                if track == 'conference':
                    fs_sibling_selector = SiblingSelector(model_name_or_path=DEFAULT_GENERAL_MODEL, pooling="mean")
                elif track == 'knowledgegraph':
                    fs_sibling_selector = SiblingSelector(model_name_or_path=DEFAULT_GENERAL_MODEL, pooling="mean")
                else:
                    fs_sibling_selector = SiblingSelector()
                info(f' SiblingSelector initialised for hard negatives (device: {fs_sibling_selector.device})')
            except Exception as e:
                warn(f' SiblingSelector init failed for hard negatives: {e}')
                warn(f' Hard negatives will fall back to random')

        try:
            dirpath = config['outputs']['logmapllm_output_dirpath']

            if negative_strategy == 'hard-similar':
                # --------------------------------------------
                # Pool-based building for per-query selection
                # --------------------------------------------
                import numpy as np_io
                from few_shot_examples import FewShotPoolBuilder

                if fs_sibling_selector is None:
                    warn('hard-similar requires SiblingSelector for embeddings; falling back to zero-shot')
                else:
                    # build M_ask exclusion set
                    m_ask_exclusion = set()
                    for _, row in m_ask_df.iterrows():
                        m_ask_exclusion.add(frozenset({row.iloc[0], row.iloc[1]}))

                    pool_builder = FewShotPoolBuilder(
                        train_path=train_path,
                        OA_source=OA_source,
                        OA_target=OA_target,
                        prompt_function=prompt_function,
                        m_ask_uri_pairs=m_ask_exclusion,
                        sibling_selector=fs_sibling_selector,
                        seed=few_shot_seed,
                        answer_format=answer_format,
                    )
                    pool_data = pool_builder.build_pool()

                    # save pool candidates (JSON) and embeddings (NPY)
                    pool_json = os.path.join(dirpath, f"{task_name}-{oupt_name}-few_shot_pool.json")
                    pool_embs = os.path.join(dirpath, f"{task_name}-{oupt_name}-few_shot_pool_embs.npy")

                    with open(pool_json, 'w') as fp:
                        json.dump(pool_data['candidates'], fp)
                    np_io.save(pool_embs, pool_data['embeddings'])

                    success(f"Built few-shot pool: {len(pool_data['candidates'])} candidates")

                    # save embedding model identifier for Step 3 consistency check
                    pool_meta = {
                        'model': fs_sibling_selector._model.name_or_path
                            if hasattr(fs_sibling_selector._model, 'name_or_path')
                            else 'unknown',
                        'n_candidates': len(pool_data['candidates']),
                        'embed_dim': int(pool_data['embeddings'].shape[1])
                            if pool_data['embeddings'].size > 0 else 0,
                    }
                    pool_meta_path = os.path.join(
                        dirpath,
                        f"{task_name}-{oupt_name}-few_shot_pool_meta.json"
                    )
                    with open(pool_meta_path, 'w') as fp:
                        json.dump(pool_meta, fp)

                # build M_ask pair labels for per-query embedding in Step 3
                info('Building M_ask pair label lookup for hard-similar ...')
                from onto_object import resolve_entity
                pair_labels = {}
                for _, row in m_ask_df.iterrows():
                    src_uri = row[column_source_entity_uri]
                    tgt_uri = row[column_target_entity_uri]
                    pair_key = f"{src_uri}|{tgt_uri}"
                    src_label = tgt_label = ""
                    try:
                        src_ent, _ = resolve_entity(src_uri, OA_source)
                        src_names = src_ent.get_preferred_names()
                        src_label = min(src_names) if src_names else src_uri.rsplit('#', 1)[-1].rsplit('/', 1)[-1]
                    except Exception:
                        src_label = src_uri.rsplit('#', 1)[-1].rsplit('/', 1)[-1]
                    try:
                        tgt_ent, _ = resolve_entity(tgt_uri, OA_target)
                        tgt_names = tgt_ent.get_preferred_names()
                        tgt_label = min(tgt_names) if tgt_names else tgt_uri.rsplit('#', 1)[-1].rsplit('/', 1)[-1]
                    except Exception:
                        tgt_label = tgt_uri.rsplit('#', 1)[-1].rsplit('/', 1)[-1]
                    pair_labels[pair_key] = [src_label, tgt_label]

                labels_path = os.path.join(
                    dirpath,
                    f"{task_name}-{oupt_name}-m_ask_pair_labels.json"
                )
                with open(labels_path, 'w') as fp:
                    json.dump(pair_labels, fp)
                success(f"Saved M_ask pair labels ({len(pair_labels)} pairs)")

            else:
                # static K-element path (hard / random)
                examples = build_few_shot_examples(
                    train_path=train_path,
                    OA_source=OA_source,
                    OA_target=OA_target,
                    prompt_function=prompt_function,
                    m_ask_df=m_ask_df,
                    k=few_shot_k,
                    bidirectional=bidirectional_mode,
                    negative_strategy=negative_strategy,
                    sibling_selector=fs_sibling_selector,
                    seed=few_shot_seed,
                    answer_format=answer_format,
                )
                # save to JSON
                fs_filename = task_name + '-' + oupt_name + '-few_shot_examples.json'
                fs_filepath = os.path.join(dirpath, fs_filename)
                with open(fs_filepath, 'w') as fp:
                    json.dump(examples, fp)
                success(f'Saved {len(examples)} few-shot examples to {fs_filename}')

        except Exception as e:
            warning(f'Few-shot example generation failed: {e}')
            warn('Consultation will proceed in zero-shot mode')

        finally:
            # clean up temporary anchor file
            if _anchor_tmp_path and os.path.isfile(_anchor_tmp_path):
                os.unlink(_anchor_tmp_path)