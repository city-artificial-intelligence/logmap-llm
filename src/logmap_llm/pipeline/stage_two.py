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
import os
import sys
import tempfile
import pandas as pd
import json

from functools import partial

import logmap_llm.oracle.prompts.templates as opb

from logmap_llm.utils.subprocess import subprocess_bootstrap
from logmap_llm.ontology.access import load_ontologies
from logmap_llm.oracle.prompts.templates import (
    set_response_config, 
    set_ontology_domain, 
    get_ontology_domain
)
from logmap_llm.constants import (
    EntityType,
    PromptBuildMode,
    M_ASK_COLUMNS,
    PAIRS_SEPARATOR,
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    DEFAULT_MAX_SIBLING_CANDIDATES,
    DEFAULT_OWLREADY2_CACHE_DIR,
)
from logmap_llm.oracle.prompts.few_shot import (
    build_few_shot_examples,
)
from logmap_llm.ontology.sibling_retrieval import (
    SiblingSelector,
    SiblingSelectionStrategy,
)
from logmap_llm.utils.logging import (
    warning,
    warn,
    success,
    step,
)
from logmap_llm.config.schema import (
    PromptTemplateConfig,
)


###
# START: PRIVATE HELPERS
###

def _copy_and_coerce_tuple_to_list(xs: tuple) -> list:
    return list(xs)


def _get_m_ask_column_names() -> list[str]:
    return _copy_and_coerce_tuple_to_list(M_ASK_COLUMNS)


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


def _build_sibling_selector(prompts_cfg: PromptTemplateConfig) -> SiblingSelector | None:
    try:
        strategy = _resolve_sibling_strategy(
            prompts_cfg.sibling_strategy,
            get_ontology_domain(),
        )
        model_override = prompts_cfg.sibling_model
        max_cands = prompts_cfg.sibling_max_candidates
        
        if max_cands is None:
            max_cands = DEFAULT_MAX_SIBLING_CANDIDATES
        
        step(f'[STEP TWO] Initialising SiblingSelector (strategy={strategy.value}, max_candidates={max_cands}).', important=True)
        
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

###
# END: PRIVATE HELPERS
###


def main():

    ###
    # BOOTSTRAP (THIS SUBPROCESS)
    #############################

    # CONFIG, PATH & FILESYSTEM MANAGEMENT & LOGGING

    step("[STEP 2] Running PROMPT BUILD as an isolated subprocess ... ")
    
    from logmap_llm.pipeline.cli import parse_args
    args = parse_args()

    config, run_paths, tee = subprocess_bootstrap("PROMPT_BUILD_STAGE_TWO", args=args)
    
    config_path = args.config
    
    step(f"[STEP 2] Configuration loaded from: {config_path}")

    initial_mappings_fp = run_paths.logmap_mappings()
    step(f"[STEP 2] Reading initial alignment mappings from file: {initial_mappings_fp}")
    mappings = pd.read_csv(initial_mappings_fp, sep=PAIRS_SEPARATOR, header=None)
    step(f'[STEP 2] Number of mappings in initial alignment: {len(mappings)}')

    m_ask_fp = run_paths.logmap_m_ask()
    m_ask_df = pd.read_csv(m_ask_fp, sep=PAIRS_SEPARATOR, header=None)
    m_ask_df.columns = _get_m_ask_column_names()
    step(f'[STEP 2] Loading mappings to ask an Oracle from: {m_ask_fp}')

    # if logmap produces no uncertain mappings, simply exit:
    if os.path.getsize(m_ask_fp) == 0:
        warning("M_ask file is empty — no uncertain mappings to build prompts for.")
        warning("Exiting Stage 2 subprocess (nothing to do).")
        sys.stdout = tee.original_stdout
        tee.close()
        sys.exit(0)
    


    # NOTE: (this may not be the best of ideas, since we're modifying global state...)
    # this is _okay_ for now, since the ontology_domain doesn't change for a single
    # isolated python process -- well, actually, this subprocess should have its own
    # dedicated `global _ontology_domain`, just be careful when modifying this later ...



    ###
    # CONFIGURE RESPONSE FORMAT FOR PROMPTS
    #######################################
    # apply response configuration to prompt templates
    # must happen before any prompt functions are called
    
    step(f"[STEP 2] Response config set to: answer_format={config.oracle.answer_format}.")
    step(f"[STEP 2] Response config set to: response_mode={config.oracle.response_mode}")
    step(f"[STEP 2] MODIFYING GLOBAL RESPONSE CONFIG STATE FOR TEMPLATES!", important=True)
    set_response_config(config.oracle.answer_format, config.oracle.response_mode)


    ###
    # CONFIGURE ONTOLOGY DOMAIN QUALIFIER
    #####################################
    # apply ontology domain qualifier to prompt templates
    # must happen before any prompt functions are called

    step(f"[STEP 2] Specifying 'ontology_domain' as '{config.alignmentTask.ontology_domain}' ... ", important=True)
    set_ontology_domain(config.alignmentTask.ontology_domain)
    if config.alignmentTask.ontology_domain != get_ontology_domain():
        warning("The 'ontology_domain' specified within the config does not match the global ontology domain state!")


    ###
    # CONFIGURE PROMPT TEMPLATE AND DIRECTION
    #########################################
    # obtain the prompt template name, check whether it is
    # a 'forward only' or a forward+reverse (ie. bidirectional) template

    oupt_name = config.prompts.cls_usr_prompt_template_name
    bidirectional_mode = opb.registry.is_bidirectional(oupt_name)
    step(f"[STEP 2] Using prompt template: {oupt_name} (bidirectional={str(bidirectional_mode)}).", important=True)


    ###
    # CONFIGURE SIBLING CONTEXT (IF REQUIRED)
    #########################################
    # if the selected template requires sibling-based 
    # context then initialise the SiblingSelector 

    sibling_selector = None
    if opb.registry.requires_siblings(oupt_name):
        sibling_selector = _build_sibling_selector(config.prompts)
    if sibling_selector is not None:
        step(f"[STEP 2] SiblingSelector has been initialised (using {sibling_selector.__class__.__name__}).")


    ###
    # RESOLVE PROPERTY PROMPT (IF REQUIRED)
    #######################################
    # if the user has specified a property prompt template within the config
    # ie. when `prop_usr_prompt_template_name` is set under [prompts] in config.toml
    # then we should construct property prompts for use within this experimental run

    property_prompt_function = None
    property_prompt_template_name = config.prompts.prop_usr_prompt_template_name

    if property_prompt_template_name is not None:
        resolved_prop_template_entry = opb.registry.get(property_prompt_template_name)

        # TODO: handle DPROP and OPROPs differently in future
        if resolved_prop_template_entry.entity_type == EntityType.OBJECTPROPERTY:
            property_prompt_function = resolved_prop_template_entry.fn

        if property_prompt_function:
            step(f"[STEP 2] Using PROPERTY PROMPT TEMPLATE: {property_prompt_template_name}", important=True)

        else:
            warning(f"THE PROPERTY PROMPT TEMPLATE '{property_prompt_template_name}' CANNOT BE RESOLVED FROM THE REGISTRY!")
            warning(f"ARE YOU SURE YOU USED THE CORRECT PROPERTY PROMPT TEMPLATE NAME?!")
            warning(f"Property mappings will be skipped!")


    ###
    # RESOLVE INSTANCE PROMPT (IF REQUIRED)
    #######################################
    # if the user has specified an instance prompt template within the config
    # ie. when `inst_usr_prompt_template_name` is set under [prompts] in config.toml
    # then we should construct instance prompts for use within this experimental run

    instance_prompt_function = None
    instance_prompt_template_name = config.prompts.inst_usr_prompt_template_name

    if instance_prompt_template_name is not None:
        resolved_inst_template_entry = opb.registry.get(instance_prompt_template_name)

        if resolved_inst_template_entry.entity_type == EntityType.INSTANCE:
            instance_prompt_function = resolved_inst_template_entry.fn

        if instance_prompt_function:
            step(f"[STEP 2] Using INSTANCE PROMPT TEMPLATE: {instance_prompt_template_name}", important=True)

        else:
            warning(f"THE INSTANCE PROMPT TEMPLATE '{instance_prompt_template_name}' CANNOT BE RESOLVED FROM THE REGISTRY!")
            warning(f"ARE YOU SURE YOU USED THE CORRECT INSTANCE PROMPT TEMPLATE NAME?!")
            warning(f"Instance mappings will be skipped!")


    ###
    # !! END OF BOOTSTRAPPING PROCESS !!
    ###



    ###
    # START: PROMPT BUILD
    #####################

    print()

    step("[STEP 2] STARTING: Building oracle user prompts ... ")
    if bidirectional_mode:
        step("[STEP 2] (bidirectional mode is set) Constructing two prompts per candidate.")


    ###
    # LOAD ONTOLOGIES
    #################

    step("[STEP 2] Loading ontologies ... (CHECKING IF OWLREADY2 CACHE EXISTS) ")
    if args.no_cache:
        warn("The '--no-cache' flag has been specified, skipping cache check.")

    cache_dir = None if args.no_cache else DEFAULT_OWLREADY2_CACHE_DIR

    OA_source, OA_target = load_ontologies(
        config.alignmentTask.onto_source_filepath, 
        config.alignmentTask.onto_target_filepath, 
        cache_dir=cache_dir,
        vocabulary=config.alignmentTask.resolved_vocabulary,
    )

    success("LOADED ONTOLOGIES!\n")


    ###
    # PROMPT CONSTRUCTION
    #####################

    # NOTE: we only support bidirectional_mode for CLS (only) tasks at present (TODO).

    if bidirectional_mode:
        step("[STEP 2] CONSTRUCTING BIDIRECTIONAL TEMPLATES.")
        m_ask_oracle_user_prompts, n_equiv_cands, n_non_equiv_skipped = (
            opb.build_oracle_user_prompts_bidirectional(
                oupt_name, config.alignmentTask.onto_source_filepath, config.alignmentTask.onto_target_filepath, m_ask_df,
                OA_source=OA_source, OA_target=OA_target,
                sibling_selector=sibling_selector
            )
        )
        step(f"[STEP 2] (n_prompts={str(len(m_ask_oracle_user_prompts))})")
        step(f"[STEP 2] (n_equiv_cands={str(n_equiv_cands)})")
        step(f"[STEP 2] (n_non_equiv_skipped={str(n_non_equiv_skipped)})")

    else:
        step("[STEP 2] CONSTRUCTING TEMPLATES.")
        m_ask_oracle_user_prompts = opb.build_oracle_user_prompts(
            oupt_name, config.alignmentTask.onto_source_filepath, config.alignmentTask.onto_target_filepath, m_ask_df,
            OA_source=OA_source, OA_target=OA_target,
            sibling_selector=sibling_selector,
            property_prompt_name=property_prompt_template_name,
            instance_prompt_name=instance_prompt_template_name,
            property_prompt_function=property_prompt_function,
            instance_prompt_function=instance_prompt_function
        )

    if m_ask_oracle_user_prompts is not None:
        step(f"[STEP 2] Number of LLM Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}", important=True)


    # NOTE: LOGIC HERE IS BACKWARDS. DONT BOTHER DOING ANYTHING IF WE ARE NOT IN BUILD MODE.
    # IN FACT. THIS SCRIPT WONT EVEN BE CALLED IF PromptBuildMode IS NOT SET TO BUILD!
    # THIS IS SIMPLY A GUARD? (REALLY NECCESARY?!)


    ###
    # WRITE PROMPTS TO DISK
    #######################

    step(f"[STEP 2] PromptBuildMode is set to: {config.pipeline.build_oracle_prompts}.")

    if config.pipeline.build_oracle_prompts == PromptBuildMode.BUILD:

        step(f"[STEP 2] Saving prompts to disk ... ")

        prompts_json_fp = run_paths.prompts_json()
        with open(prompts_json_fp, 'w') as fp:
            json.dump(m_ask_oracle_user_prompts, fp)

        step(f'[STEP 2] LLM Oracle user prompts saved to file: {prompts_json_fp}')


    ###
    # !! END OF PROMPT CONSTRUCTION PROCESS FOR CONSULTATION !!
    ###



    ###
    # START: FEW-SHOT EXAMPLE BUILDING (IF NECCESARY)
    ###

    _anchor_tmp_path = None
    few_shot_k = config.few_shot.few_shot_k

    if few_shot_k > 0:

        # NOTE: why are we looking into 'evaluation' here...?
        train_path = config.evaluation.train_alignment_path

        if train_path is None or not os.path.isfile(str(train_path)):

            warn("No train.tsv found - deriving few-shot positives from LogMap high-confidence anchors")

            m_ask_pairs = set()

            for _idx, row in m_ask_df.iterrows():
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

            step(f"[STEP 2] Filtering anchors to {anchor_entity_type} entities.")

            anchors = []

            for _idx, row in mappings.iterrows():
                src, tgt = row.iloc[0], row.iloc[1]
                entity_type = row.iloc[4] if len(row) > 4 else "CLS"
                if entity_type != anchor_entity_type:
                    continue
                if (src, tgt) not in m_ask_pairs:
                    anchors.append((src, tgt))

            if len(anchors) == 0:
                warning(f"No {anchor_entity_type} anchors available for few-shot examples !! falling back to zero-shot !! ")
                few_shot_k = 0  # prevent downstream build attempt

            else:
                
                step(f"[STEP 2] Anchor pool: {len(anchors)} {anchor_entity_type} pairs (from {len(mappings)} initial - {len(m_ask_pairs)} M_ask)")
                # writes to a temporary TSV that FewShotExampleBuilder can read
                anchor_fd, anchor_path = tempfile.mkstemp(suffix='.tsv', prefix='anchors_')
                os.close(anchor_fd)
                with open(anchor_path, 'w') as f:
                    for src, tgt in anchors:
                        f.write(f"{src}\t{tgt}\n")
                _anchor_tmp_path = anchor_path
                train_path = anchor_path

        # build few-shot examples (runs for both train.tsv and anchor-derived paths)

        if few_shot_k > 0 and train_path is not None:

            step(f"[STEP 2] Building {few_shot_k} few-shot examples (strategy: {config.few_shot.few_shot_negative_strategy})", important=True)

            # get the bound prompt function matching the anchor entity type;
            # if instance prompts are active, use the instance template;
            # otherwise use the class template
            
            if instance_prompt_function is not None:
                prompt_function = instance_prompt_function

            else:
                prompt_function = opb.get_oracle_user_prompt_template_function(oupt_name)
                if sibling_selector is not None and opb.registry.requires_siblings(oupt_name):
                    prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

            few_shot_seed = config.few_shot.few_shot_seed # default: 42
            negative_strategy = config.few_shot.few_shot_negative_strategy # default: hard

            fs_sibling_selector = sibling_selector

            # TODO: implement hard-similar strategy for RAG-based few-shot that mutates message state
            # & think about this in more depth ... if intuition serves, multiple dispatched threads 
            # could mutate each others state; however, I need to check if this is how the GIL works

            if negative_strategy in ('hard', 'hard-similar') and fs_sibling_selector is None:
                fs_sibling_selector = _build_sibling_selector(config.prompts)
                if fs_sibling_selector is None:
                    warning('Hard negatives will fall back to random.')

            try:
            
            ###
            # Static K-element path (hard / random)
            #######################################

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
                    answer_format=config.oracle.answer_format,
                    response_mode=config.oracle.response_mode,
                )

                few_shot_examples_json_fp = run_paths.few_shot_json()
                with open(few_shot_examples_json_fp, 'w') as fp:
                    json.dump(examples, fp)

                step(f"[STEP 2] SAVED {len(examples)} few-shot examples to {str(few_shot_examples_json_fp)}")

            except Exception as e:
                warning(f'Few-shot example generation failed: {e}')
                warn('Consultation will proceed in zero-shot mode')

            finally:
                # clean up temporary anchor file
                if _anchor_tmp_path and os.path.isfile(_anchor_tmp_path):
                    os.unlink(_anchor_tmp_path)

if __name__ == "__main__":
    main()
