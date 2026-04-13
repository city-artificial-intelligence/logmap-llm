'''
logmap_llm.pipeline.stage_two
designated subprocess responsible for prompt construction
This is a essentially a stone-alone 'build prompts' script.
The script mixes legacy implementation with the current rework
remaining TODOs include integrating the config loader and paths
module, so that filepath resolution occurs consistently across
both runner.py and this subprocess. Also, see notes in code
regarding remaining work.
'''
import argparse
import os
import sys
import tempfile
import tomllib
import pandas as pd
import json

from functools import partial
from datetime import datetime, timezone

import logmap_llm.oracle.prompts.templates as opb

from logmap_llm.oracle.prompts.templates import (
    set_response_config, 
    set_ontology_domain, 
    get_ontology_domain
)

from logmap_llm.constants import (
    EntityType,
    M_ASK_COLUMNS,
    PAIRS_SEPARATOR,
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
)

from logmap_llm.utils.logging import (
    TeeWriter,
    error,
    warning,
    warn,
    info,
    success
)

from logmap_llm.ontology.sibling_retrieval import (
    SiblingSelector,
    SiblingSelectionStrategy,
)
from logmap_llm.constants import (
    DEFAULT_MAX_SIBLING_CANDIDATES,
)

from logmap_llm.oracle.prompts.few_shot import build_few_shot_examples



def copy_and_coerce_tuple_to_list(xs: tuple) -> list:
    return list(xs)


def get_m_ask_column_names() -> list[str]:
    return copy_and_coerce_tuple_to_list(M_ASK_COLUMNS)


def _resolve_sibling_strategy(configured_strategy: str | None, ontology_domain: str | None) -> SiblingSelectionStrategy:
    '''
    Picks a sibling selection strategy.
    Precedence: explicit config > biomedical-domain auto > generic auto.
    Pooling: SapBERT -> CLS, SBERT -> mean.
    '''
    if configured_strategy is not None:
        return SiblingSelectionStrategy(configured_strategy)
    if ontology_domain and 'biomedical' in ontology_domain.lower():
        return SiblingSelectionStrategy.SAPBERT
    return SiblingSelectionStrategy.SBERT


def _build_sibling_selector(prompts_cfg: dict) -> SiblingSelector | None:
    try:
        strategy = _resolve_sibling_strategy(
            prompts_cfg.get('sibling_strategy'),
            get_ontology_domain(),
        )
        model_override = prompts_cfg.get('sibling_model')
        max_cands = prompts_cfg.get('sibling_max_candidates') or DEFAULT_MAX_SIBLING_CANDIDATES
        
        info(f'Initialising SiblingSelector (strategy={strategy.value}, max_candidates={max_cands}) ... ')
        
        sel = SiblingSelector(
            strategy=strategy,
            model_name_or_path=model_override,
            max_candidates=max_cands,
        )
        success(f'SiblingSelector ready (device: {sel.device})')
        return sel
    
    except Exception as outer_e:
        warning(f'SiblingSelector initialisation failed: {outer_e}')
        warn('Falling back to alphanumeric sibling selection.')
        try:
            return SiblingSelector(strategy=SiblingSelectionStrategy.ALPHANUMERIC)
        except Exception as inner_e:
            warning(f'Even alphanumeric fallback failed: {inner_e}')
            return None



def main():

    parser = argparse.ArgumentParser(
        description="LogMap-LLM: Prompt Building (subprocess)."
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to the TOML configuration file",
    )
    parser.add_argument(
        "--reuse-align",
        action="store_true",
        default=False,
        help="Override config to reuse existing LogMap alignment.",
    )
    parser.add_argument(
        "--reuse-prompts",
        action="store_true",
        default=False,
        help="Override config to reuse existing prompts. Implies --reuse-align.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable owlready2 quadstore caching (parse ontologies from scratch).",
    )
    args = parser.parse_args()

    ###
    # MAIN
    ###

    config_path = args.config

    if not os.path.isfile(config_path):
        error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path, mode="rb") as fp:
        config = tomllib.load(fp)

    # apply CLI overrides (forwarded from parent process)
    if args.reuse_prompts:
        config['pipeline']['build_oracle_prompts'] = 'reuse'
        config['pipeline']['align_ontologies'] = 'reuse'
    elif args.reuse_align:
        config['pipeline']['align_ontologies'] = 'reuse'

    info(f"Configuration loaded from: {config_path}")

    # read ontology domain from config
    ontology_domain = config.get('alignmentTask', {}).get('ontology_domain', None)
    
    if ontology_domain:
        info(f"Detected ontology domain, set to: {ontology_domain}")

    task_name = config['alignmentTask']['task_name']
    onto_src_filepath = config['alignmentTask']['onto_source_filepath']
    onto_tgt_filepath = config['alignmentTask']['onto_target_filepath']

    logmap_outputs_dir_path = config['outputs']['logmap_initial_alignment_output_dirpath']

    ###
    # Initialise timing and logging
    ###

    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # determine output directory for logs and results
    results_dir = config['outputs']['logmapllm_output_dirpath']
    os.makedirs(results_dir, exist_ok=True)

    # start logger
    log_filename = f"pipeline_log_{run_timestamp}.txt"
    log_filepath = os.path.join(results_dir, log_filename)
    original_stdout = sys.stdout
    tee = TeeWriter(log_filepath, original_stdout)
    sys.stdout = tee

    info(f"Pipeline log file: {log_filepath}")
    info('Reusing existing initial LogMap alignment ...')

    # NOTE: we should adopt the paths.py (in pipeline) for managing
    # the file paths ...
    filename = task_name + '-logmap_mappings.txt'
    filepath = os.path.join(logmap_outputs_dir_path, filename)
    mappings = pd.read_csv(filepath, sep=PAIRS_SEPARATOR, header=None)

    info(f'Number of mappings in initial alignment: {len(mappings)}')

    filename = task_name + '-logmap_mappings_to_ask_oracle_user_llm.txt'

    print(f'Loading mappings to ask an Oracle from file: {filename}')
    
    filepath = os.path.join(logmap_outputs_dir_path, filename)

    # handle empty M_ask (in rare cases LogMap may produce no uncertain mappings)
    if os.path.getsize(filepath) == 0:
        warning("M_ask file is empty — no uncertain mappings to build prompts for.")
        warning("Exiting Stage 2 subprocess (nothing to do).")
        sys.stdout = original_stdout
        tee.close()
        sys.exit(0)

    m_ask_df = pd.read_csv(filepath, sep=PAIRS_SEPARATOR, header=None)
    m_ask_df.columns = get_m_ask_column_names()

    oupt_name = config.get('prompts', {}).get('cls_usr_prompt_template_name', 'synonyms_only')
    bidirectional_mode = opb.registry.is_bidirectional(oupt_name)

    # apply response configuration to prompt templates
    # (must happen before any prompt functions are called)
    answer_format = config.get('oracle', {}).get('answer_format', 'true_false')
    response_mode = config.get('oracle', {}).get('response_mode', 'structured')
    
    set_response_config(answer_format, response_mode)
    
    info(f'Response config set to: answer_format={answer_format}, response_mode={response_mode}')

    # apply ontology domain qualifier to prompt templates
    # (must happen before any prompt functions are called)

    # NOTE: (this may not be the best of ideas, since we're modifying global state...)
    # this is _okay_ for now, since the ontology_domain doesn't change for a single
    # isolated python process -- well, actually, this subprocess should have its own
    # dedicated `global _ontology_domain`, just be careful when modifying this later ...

    set_ontology_domain(ontology_domain)
    if ontology_domain:
        info(f'Ontology domain set to: {get_ontology_domain()}')

    ###
    # SIBLING CONTEXT
    # init SiblingSelector 
    # (if template requires sibling context)
    ###

    sibling_selector = None
    prompts_cfg = config.get('prompts', {})
    if opb.registry.requires_siblings(oupt_name):
        sibling_selector = _build_sibling_selector(prompts_cfg)

    ###
    # RESOLVE PROPERTY PROMPTS
    # property prompts are built when the `prop_usr_prompt_template_name`
    # under [prompts] in the config.toml is set.
    ###

    property_prompt_function = None
    prop_template_name = config.get('prompts', {}).get('prop_usr_prompt_template_name', None)

    if prop_template_name is not None:

        resolved_prop_template_entry = opb.registry.get(prop_template_name)

        # TODO: we might look to handle DPROP and OPROPs differently in future
        if resolved_prop_template_entry.entity_type == EntityType.OBJECTPROPERTY:
            property_prompt_function = resolved_prop_template_entry.fn

        if property_prompt_function:
            info(f'Property prompt template: {prop_template_name}')

        else:
            warn(f'Property prompt template "{prop_template_name}" not found — property mappings will be skipped')


    ###
    # RESOLVE INSTANCE PROMPTS
    # instance prompt are built when the `inst_usr_prompt_template_name` 
    # under [prompts] in the config.toml is set 
    ###

    instance_prompt_function = None
    inst_template_name = config.get('prompts', {}).get('inst_usr_prompt_template_name', None)

    if inst_template_name is not None:

        resolved_inst_template_entry = opb.registry.get(inst_template_name)

        if resolved_inst_template_entry.entity_type == EntityType.INSTANCE:
            instance_prompt_function = resolved_inst_template_entry.fn

        if instance_prompt_function:
            info(f'Instance prompt template: {inst_template_name}')

        else:
            warn(f'Instance prompt template "{inst_template_name}" not found — instance mappings will be skipped')


    ###
    # PROMPT BUILD SETUP
    ###

    info('Building fresh Oracle user prompts ...')
    if bidirectional_mode:
        print('A bidirectional template is in use. Constructing two prompts per candidate.')

    ###
    # LOAD ONTOLOGIES
    ###

    info('Loading ontologies ...')
    
    cache_dir = None if args.no_cache else os.path.expanduser('~/.cache/logmap-llm/owlready2')
    
    OA_source, OA_target = opb.load_ontologies(
        onto_src_filepath, onto_tgt_filepath, cache_dir=cache_dir
    )
    
    success('Ontologies loaded.')

    ###
    # PROMPT CONSTRUCTION
    ###

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
        info(f"Number of LLM Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}")

    if config['pipeline']['build_oracle_prompts'] == 'build':
        # save the newly built oracle user prompts to a .json file so they can be reused
        dirpath = config['outputs']['logmapllm_output_dirpath']
        filename = task_name + '-' + oupt_name + '-mappings_to_ask_oracle_user_prompts.json'
        print(f'LLM Oracle user prompts saved to file: {filename}')
        filepath = os.path.join(dirpath, filename)
        with open(filepath, 'w') as fp:
            json.dump(m_ask_oracle_user_prompts, fp)

    ###
    # FEW-SHOT EXAMPLE BUILDING
    ###

    few_shot_k = config.get('few_shot', {}).get('few_shot_k', 0)
    _anchor_tmp_path = None  # track temp file for cleanup

    if few_shot_k > 0:

        train_path = config.get('evaluation', {}).get('train_alignment_path', None)

        if train_path is None or not os.path.isfile(str(train_path)):

            ###
            # No train.tsv available — derive positive examples from LogMap's
            # high-confidence anchors (initial alignment minus M_ask).
            ###

            info("No train.tsv found — deriving few-shot positives from LogMap high-confidence anchors")

            # build M_ask pair set for subtraction
            m_ask_pairs = set()
            for _, row in m_ask_df.iterrows():
                m_ask_pairs.add((row[COL_SOURCE_ENTITY_URI], row[COL_TARGET_ENTITY_URI]))

            # Filter initial alignment to anchors not in M_ask. When instance prompts are active, 
            # filter to INST entities; otherwise filter to CLS entities (class matching).
            # Mixing entity types with the wrong template would produce incoherent prompts.
            # Initial alignment columns: 0=src, 1=tgt, 2=relation, 3=conf, 4=entityType

            # TODO: this implementation is (partly) legacy, where we assume that CLS, PROP and INST
            # prompts are evaluated by seperate python processes. However, _it is the case_ that
            # we need to provide different examples based on whether we're performing CLS, OPROP/DPROP
            # or INST alignment, all of which can occur within the same python process ...
            # _for the time being, we simply assume that for KG track, INST dominates, so we use INST
            # examples, for all other tracks (it is assumed) CLS dominates, so we use CLS examples_
            # The primary implication is that few-shot examples on Conference are unlikely to be reliable.

            # The fix would involve:
            #
            #   (1) Identifying entity type prompts are currently in use during this run: 
            #       (f.P -> ETs : ETs \subseteq { CLS, PROP, INST }).
            #   (2) Construct up to three seperate conversation histories:
            #       (f.ETs -> (f.K -> CH : CH := (msg, resp)_i ... (msg, resp)_N \forall e \in ETs)
            #   (3) In OracleConsultationManager we swicth on each prompts EntityType (if few_shot_k > 0)
            #       to provide the matched conversation history.
            #
            #   DATE OF COMMENT: 12th of April (we'll defer this change until next week)
            #   ... I am cognisant that the refactor is taking longer than I had antisipated ...
            #                                       _(ALWAYS TRIPPLE TIME ESTIAMTES!!)_

            if instance_prompt_function is not None:
                anchor_entity_type = "INST"
            else:
                anchor_entity_type = "CLS"

            info(f"Filtering anchors to {anchor_entity_type} entities")

            anchors = []

            for _, row in mappings.iterrows():

                src, tgt = row.iloc[0], row.iloc[1]
                entity_type = row.iloc[4] if len(row) > 4 else "CLS"
                if entity_type != anchor_entity_type:
                    continue
                if (src, tgt) not in m_ask_pairs:
                    anchors.append((src, tgt))

            if len(anchors) == 0:

                warn(f"No {anchor_entity_type} anchors available for few-shot examples; falling back to zero-shot")
                few_shot_k = 0  # prevent downstream build attempt

            else:
                
                info(f"Anchor pool: {len(anchors)} {anchor_entity_type} pairs (from "
                     f"{len(mappings)} initial - {len(m_ask_pairs)} M_ask)")

                # write to a temporary TSV that FewShotExampleBuilder can read
                anchor_fd, anchor_path = tempfile.mkstemp(suffix='.tsv', prefix='anchors_')
                os.close(anchor_fd)

                with open(anchor_path, 'w') as f:
                    for src, tgt in anchors:
                        f.write(f"{src}\t{tgt}\n")

                _anchor_tmp_path = anchor_path
                train_path = anchor_path

        # build few-shot examples (runs for both train.tsv and anchor-derived paths)

        if few_shot_k > 0 and train_path is not None:

            info(f'Building {few_shot_k} few-shot examples (strategy:'
                 f' "{config.get("few_shot", {}).get("few_shot_negative_strategy", "hard")}) ... ')

            # get the bound prompt function matching the anchor entity type;
            # if instance prompts are active, use the instance template;
            # otherwise use the class template
            
            if instance_prompt_function is not None:
                prompt_function = instance_prompt_function

            else:
                prompt_function = opb.get_oracle_user_prompt_template_function(oupt_name)
                if sibling_selector is not None and opb.registry.requires_siblings(oupt_name):
                    prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

            few_shot_seed = config.get('few_shot', {}).get('few_shot_seed', 42) # DEFAULT_SEED is 42 :)
            negative_strategy = config.get('few_shot', {}).get('few_shot_negative_strategy', 'hard')

            fs_sibling_selector = sibling_selector

            # TODO: implement hard-similar strategy for RAG-based few-shot that mutates message state
            # & think about this in more depth ... if intuition serves, multiple dispatched threads 
            # could mutate each others state; however, I need to check if this is how the GIL works

            if negative_strategy in ('hard', 'hard-similar') and fs_sibling_selector is None:
                fs_sibling_selector = _build_sibling_selector(prompts_cfg)
                if fs_sibling_selector is None:
                    warn('Hard negatives will fall back to random.')

            try:
                ###
                # Static K-element path (hard / random)
                ###
                dirpath = config['outputs']['logmapllm_output_dirpath']
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
                    response_mode=response_mode,
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

if __name__ == "__main__":
    main()
