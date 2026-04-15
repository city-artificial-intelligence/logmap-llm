"""
logmap_llm.pipeline.orchestration

Ported from jd-extended, see:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/pipeline_steps.py

With some modifications (ready for future branches/features).
"""
from __future__ import annotations

import sys
import os
import subprocess
import json
import pandas as pd

from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.contracts import (
    AlignmentResult,
    PromptBuildResult,
    OracleResult,
    RefinementResult,
    EvaluationResult,
)
from logmap_llm.constants import (
    PAIRS_SEPARATOR,
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
    RefinementStrategy,
)
from logmap_llm.utils.data import (
    normalise_prediction_column,
    filter_accepted_predictions,
)
from logmap_llm.utils.logging import (
    fatal,
    critical,
    warning,
    warn,
    step,
    success,
)



# NOTE: Do NOT import oracle_prompt_building or onto_access here.
# Those modules transitively import owlready2, which cannot coexist
# with JPype in the same process.  Prompt building runs in a subprocess.



# PRIVATE HELPERS
#################



def _detect_bidirectional(prompts: dict) -> bool:
    """
    Detect whether prompts were built in bidirectional mode; bidirectional prompt keys 
    have a direction suffix, specifically: 'src_uri|tgt_uri|REVERSE', whereas standard
    keys are just: 'src_uri|tgt_uri'
    """
    if not prompts:
        return False
    return any(key.count(PAIRS_SEPARATOR) >= 2 for key in prompts)



def _validate_prompt_keys(prompts: dict, bidirectional: bool, template_name: str = "") -> None:
    """
    Validates that the prompt key format matches the consultation mode.
    """
    if not prompts:
        return
    
    keys_have_direction = _detect_bidirectional(prompts)
    
    if bidirectional and not keys_have_direction:
        raise ValueError(
            f"Bidirectional template '{template_name}' selected but prompts   "
            f"lack direction keys (eg. '...|REVERSE'). The prompt JSON was    "
            f"likely built with a non-bidirectional template. Rebuild prompts "
            f"or select a matching template."
        )
    if not bidirectional and keys_have_direction:
        raise ValueError(
            f"Standard template '{template_name}' selected but prompts have  "
            f"direction keys (eg. '...|REVERSE'). The prompt JSON was likely "
            f"built with a bidirectional template. Rebuild prompts or select "
            f"a matching template."
        )



def _kg_refine_in_python(ctx: PipelineContext, oracle_result: OracleResult) -> None:
    """
    Produce refined alignment for KG track in Python.

    Computes: refined = { initial - M_ask } U { m \in M_ask : oracle(m) = True }

    Prior to patching, LogMap's Java refinement crashes with NullPointerExceptions for instance matching.
    Additionally, the alignment stage does take a while (eg. when a KG has > 1m instances); this can be
    offset through by-passing the initial aligning (with --reuse-align). However, the refinement stage 
    (similarly) takes a long time; as such, this provides an optional means to quickly calculate an
    approximation to what the refinement ought to look like, but lacks final conflict resolution.
    Though, for KGs and instance matching, I am not sure how impactful the conflict resolution is 
    anyway; since LogMap was not originally designed for this specific use case.

    NOTE: since patching LogMap, the new build appears to work as intended.

    TODO: _kg_refine_in_python should handle both pipe and tab separators, or use the same format detection 
    logic as evaluate.py (updated note -- 14th April -- can't quite recall what this TODO was for, exactly;
    revisit later).
    """
    from logmap_llm.constants import (
        COL_SOURCE_ENTITY_URI,
        COL_TARGET_ENTITY_URI,
        COL_RELATION,
        COL_CONFIDENCE,
        COL_ENTITY_TYPE,
    )

    # load initial alignment
    initial_path = ctx.run_paths.logmap_mappings()
    initial_df = pd.read_csv(initial_path, sep=PAIRS_SEPARATOR, header=None)
    initial_df.columns = [
        COL_SOURCE_ENTITY_URI,
        COL_TARGET_ENTITY_URI,
        COL_RELATION,
        COL_CONFIDENCE,
        COL_ENTITY_TYPE
    ][:len(initial_df.columns)]

    # build set of m_ask pairs
    preds = oracle_result.predictions
    m_ask_pairs = set()
    
    for _, row in preds.iterrows():
        m_ask_pairs.add((str(row.iloc[0]), str(row.iloc[1])))

    # retain rows disjoint from m_ask (ie. the initial mapping)
    keep_mask = initial_df.apply(
        lambda row: (str(row.iloc[0]), str(row.iloc[1])) not in m_ask_pairs,
        axis=1
    )
    retained = initial_df[keep_mask]

    # m_ask entries where oracle predicted true
    oracle_accepted = filter_accepted_predictions(preds)

    # take the first N columns matching the initial alignment format (combine)
    ncols = len(initial_df.columns)
    accepted_subset = oracle_accepted.iloc[:, :ncols].copy()
    accepted_subset.columns = initial_df.columns

    refined = pd.concat([retained, accepted_subset], ignore_index=True)

    # write to the refined output directory
    output_path = ctx.run_paths.refined_mappings_tsv()
    os.makedirs(ctx.run_paths.refined_dir, exist_ok=True)
    refined.to_csv(output_path, sep='\t', header=False, index=False)

    step(f"[Step 4] KG refined alignment: {len(refined)} mappings "
         f"({len(retained)} retained + {len(accepted_subset)} oracle-accepted)")



def align(ctx: PipelineContext) -> AlignmentResult:
    """
    Step 1: Pass the neccesary files to LogMap to perform an initial alignment and get M_ask.
    `import bridging as br` allows for mapping LogMap IO between java and python (and vice versa)
    """
    import logmap_llm.bridging as br

    step("[Step 1] Align ontologies and obtain mappings to ask an Oracle")

    match ctx.cfg.pipeline.align_ontologies:

        case AlignMode.ALIGN:

            step("[Step 1] Performing fresh initial LogMap alignment")
            ctx.logmap.perform_alignment()
            return AlignmentResult(
                m_ask_df=br.java_mappings_2_python(
                    ctx.logmap.get_mappings_for_llm()
                ),
                mappings=br.java_mappings_2_python(
                    ctx.logmap.get_mappings()
                ),
            )

        case AlignMode.REUSE:

            step(f"[Step 1] Loading mappings from file: {ctx.run_paths.logmap_m_ask()}")
            return AlignmentResult(
                m_ask_df=br.load_m_ask_from_file(
                    ctx.run_paths.logmap_m_ask()
                ),
                mappings=pd.read_csv(
                    ctx.run_paths.logmap_mappings(),
                    sep=PAIRS_SEPARATOR,
                    header=None,
                ),
            )

        case AlignMode.BYPASS:
            warn("Bypassing initial LogMap alignment")
            return AlignmentResult()

        case _:
            fatal(f"config: align_ontologies param not recognised: {ctx.cfg.pipeline.align_ontologies}")

    fatal(f"unable to match on: {ctx.cfg.pipeline.align_ontologies}")



def prompt_build(ctx: PipelineContext, initial_alignment: AlignmentResult) -> PromptBuildResult:
    """
    Step 2: Build user prompts for oracle consultation.
    In BUILD mode, dispatches to pipeline/stage_two.py as a subprocess (isolates JVMs & owlready2).
    """
    step("[Step 2] Build user prompts for oracle consultation")

    match ctx.cfg.pipeline.build_oracle_prompts:

        case PromptBuildMode.BUILD:
            
            step("[Step 2] Building fresh oracle user prompts via subprocess")
            
            if not ctx.config_path:
                fatal(f"Stage 2 in BUILD mode without a valid config path.") # @raises

            cmd = [
                sys.executable, "-m", "logmap_llm.pipeline.stage_two",
                "--config", ctx.config_path,
            ]

            proc = subprocess.run(cmd, capture_output=False)
            if proc.returncode != 0:
                fatal(f"Stage 2 subprocess failed with return code {proc.returncode}") # @raises

            # read the prompts back from the JSON file
            prompts_path = ctx.run_paths.prompts_json()
            
            if not prompts_path.exists():
                warning(f"Prompts file not found: {prompts_path}")
                return PromptBuildResult()
            
            with open(prompts_path) as fp:
                prompts = json.load(fp)
            
            bidirectional = _detect_bidirectional(prompts)
            return PromptBuildResult(
                prompts=prompts,
                bidirectional=bidirectional,
            )

        case PromptBuildMode.REUSE:

            step(f"[Step 2] Loading LLM oracle user prompts from: {ctx.run_paths.prompts_json()}")

            with open(ctx.run_paths.prompts_json()) as fp:
                prompts = json.load(fp)

            bidirectional = _detect_bidirectional(prompts)

            _validate_prompt_keys(
                prompts, bidirectional,
                template_name=ctx.cfg.prompts.cls_usr_prompt_template_name,
            )
            return PromptBuildResult(
                prompts=prompts,
                bidirectional=bidirectional,
            )

        case PromptBuildMode.BYPASS:

            warn("Bypassing use of LLM oracle user prompts")
            return PromptBuildResult()

        case _:
            fatal(f"config: build_oracle_prompts param not recognised: {ctx.cfg.pipeline.build_oracle_prompts}")    

    fatal(f"unable to match on: {ctx.cfg.pipeline.build_oracle_prompts}")



def consult_oracle(ctx: PipelineContext, initial_alignment: AlignmentResult, prompt_build_result: PromptBuildResult) -> OracleResult:
    """
    Step 3: Consult Oracle for mappings to ask.
    """
    import logmap_llm.oracle.prompts.developer as dp
    import logmap_llm.oracle.consultation as oc

    step("[Step 3] Consult Oracle for mappings to ask")

    # PROMPT TEMPLATE SWITCHING LOGIC
    #################################
    # TODO: neatly rework

    cls_dev_prompt_text = dp.get_developer_prompt(
        name=ctx.cfg.prompts.cls_dev_prompt_template_name,
        answer_format=ctx.cfg.oracle.answer_format,
        response_mode=ctx.cfg.oracle.response_mode,
    )

    developer_prompt_map = {}

    prop_dev_prompt_text = None
    if ctx.cfg.prompts.prop_usr_prompt_template_name:
        prop_dev_prompt_text = dp.get_developer_prompt(
            ctx.cfg.prompts.prop_dev_prompt_template_name,
            answer_format=ctx.cfg.oracle.answer_format,
            response_mode=ctx.cfg.oracle.response_mode,
        )
        developer_prompt_map["OPROP"] = prop_dev_prompt_text
        developer_prompt_map["DPROP"] = prop_dev_prompt_text

    inst_dev_prompt_text = None
    if ctx.cfg.prompts.inst_usr_prompt_template_name:
        inst_dev_prompt_text = dp.get_developer_prompt(
            ctx.cfg.prompts.inst_dev_prompt_template_name,
            answer_format=ctx.cfg.oracle.answer_format,
            response_mode=ctx.cfg.oracle.response_mode,
        )
        developer_prompt_map["INST"] = inst_dev_prompt_text

    if len(developer_prompt_map.keys()) == 0:
        developer_prompt_map = None

    match ctx.cfg.pipeline.consult_oracle:

        case ConsultMode.CONSULT:
            
            if prompt_build_result.prompts is None or len(prompt_build_result.prompts) == 0:

                warn("[Step 3] No prompts available — skipping oracle consultation")
                return OracleResult() 
            
            else:

                # FEW-SHOT-HANDLER
                ##################
                # TODO: neatly rework

                if ctx.cfg.few_shot.few_shot_k > 0:

                    few_shot_fp = ctx.run_paths.few_shot_json()
                    
                    if os.path.isfile(few_shot_fp):

                        with open(few_shot_fp) as fp:
                            few_shot_examples = [tuple(pair) for pair in json.load(fp)]
                        success(f"Loaded {len(few_shot_examples)} few-shot examples from {few_shot_fp}")

                    else:

                        warn(f"few_shot_k={ctx.cfg.few_shot.few_shot_k} but examples file not found: {few_shot_fp}")
                        warn("Falling back to zero-shot")
                        few_shot_examples = None

                else:

                    few_shot_examples = None

                # END: FEW-SHOT-HANDLER

                oracle_kwargs = dict(
                    m_ask_prompts=prompt_build_result.prompts,
                    m_ask_init_alignment_df=initial_alignment.m_ask_df,
                    oracle_cfg=ctx.cfg.oracle,
                    developer_prompt_text=cls_dev_prompt_text,
                    developer_prompt_map=developer_prompt_map,
                    few_shot_examples=few_shot_examples,
                    **ctx.cfg.oracle.consult_kwargs,
                )
                
                if prompt_build_result.bidirectional:
                    step(f"[Step 3] Consulting LLM oracle (bidirectional) with model: {ctx.cfg.oracle.model_name}")
                    oracle_predictions_df = oc.consult_oracle_bidirectional(**oracle_kwargs)
                else:
                    step(f"[Step 3] Consulting LLM oracle with model: {ctx.cfg.oracle.model_name}")
                    oracle_predictions_df = oc.consult_oracle_for_mappings_to_ask(**oracle_kwargs)

                oracle_result = OracleResult(
                    predictions=oracle_predictions_df,
                    oracle_params=oracle_kwargs,
                    bidirectional=prompt_build_result.bidirectional,
                )
                oracle_result.predictions.to_csv(
                    ctx.run_paths.predictions_csv(), na_rep="nan"
                ) # ensure round-trip-ability under pd.read_csv^
                return oracle_result

        case ConsultMode.REUSE:

            step(f"[Step 3] Loading LLM oracle predictions from: {ctx.run_paths.predictions_csv()}")
            predictions_df = pd.read_csv(ctx.run_paths.predictions_csv())
            normalise_prediction_column(predictions_df) # pred.str_T/F/Y/N -> bool
            return OracleResult(predictions=predictions_df)

        case ConsultMode.LOCAL:

            step(f"[Step 3] Local oracle predictions from: {ctx.cfg.oracle.local_oracle_predictions_dirpath}")
            return OracleResult(
                local_dir=ctx.cfg.oracle.local_oracle_predictions_dirpath,
                predictions=None,
            )

        case ConsultMode.BYPASS:

            warn("Bypassing oracle consultations")
            return OracleResult()

        case _:
            fatal(f"config: consult_oracle param not recognised: {ctx.cfg.pipeline.consult_oracle}")

    fatal(f"unable to match on: {ctx.cfg.pipeline.consult_oracle}")



def refine_alignment(ctx: PipelineContext, oracle_result: OracleResult) -> RefinementResult:
    """
    Step 4: Refine alignment using oracle mapping predictions.
    """
    import logmap_llm.bridging as br

    step("[Step 4] Refine alignment using oracle mapping predictions")

    ctx.logmap.set_output_dir(ctx.run_paths.refined_dir)

    match ctx.cfg.pipeline.refine_alignment:

        case RefineMode.REFINE:

            if not oracle_result.has_predictions and not oracle_result.is_local:
                
                warn("[Step 4] No oracle predictions — skipping refinement (initial alignment is final)")
                
                import shutil
                
                # moves the initial mappings to the refined dir
                src = ctx.run_paths.logmap_mappings_tsv()
                dst = ctx.run_paths.refined_mappings_tsv()

                if src.exists():
                    os.makedirs(ctx.run_paths.refined_dir, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                    step(f"[Step 4] Initial alignment copied to refined dir: {dst.name}")

                return RefinementResult()

            # KG track: there exists a bug in the LogMap KG track execution path
            # For now, we bypass LogMap's Java refinement (since it crashes with NPEs 
            # on instance mappings) and produce refined alignment in Python
            # Note: the version of LogMap shipped with this repository can actually
            # refine the KG track via LogMap (and can therefore resolve any remaining
            # mappings associated with logical conflicts -- which is reccomended)
            # However, we cannot be certain that other users have the latest changes
            # (for example, if they use a custom LogMap version built from src)
            # therefore, we also include the PYTHON_SET_UNION bypass (see below):

            if (ctx.cfg.pipeline.refinement_strategy == RefinementStrategy.PYTHON_SETUNION and oracle_result.has_predictions):

                step("[Step 4] Python-based refinement (set-union bypass)")
                _kg_refine_in_python(ctx, oracle_result)
                return RefinementResult()

            if oracle_result.has_predictions:
                
                step("[Step 4] Refining initial LogMap alignment with LLM Oracle predictions")
                preds_java = br.python_oracle_mapping_predictions_2_java(
                    oracle_result.predictions
                )
                step(f"[Step 4] Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}")
                ctx.logmap.refine_alignment(preds_java)
                mappings_java = ctx.logmap.get_mappings()
                step(f"[Step 4] Number of mappings in LogMap refined alignment: {len(mappings_java)}")
                return RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            elif oracle_result.is_local:
                
                step("[Step 4] Refining initial LogMap alignment with local Oracle predictions")
                ctx.logmap.refine_alignment(str(oracle_result.local_dir))
                mappings_java = ctx.logmap.get_mappings()
                step(f"[Step 4] Number of mappings in LogMap refined alignment: {len(mappings_java)}")
                return RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            else:
                critical("Check your config to ensure 'refine_alignment' is set appropriately!")
                critical("The oracle response is empty and no local predictions file is specified.")
                fatal(
                    "Oracle payload passed to `refine_alignment` is either "
                    "malformed or is empty.",
                    IOError,
                )

        case RefineMode.BYPASS:

            warn("Bypassing alignment refinement")
            return RefinementResult()

        case _:
            fatal(f"config: refine_alignment param not recognised: {ctx.cfg.pipeline.refine_alignment}")

    fatal(f"unable to match on: {ctx.cfg.pipeline.refine_alignment}")



def evaluate(ctx) -> EvaluationResult:
    """
    Step 5: Evaluate the refined alignment.
    Similarly to step two (build_prompts), the evaluate method (which uses the DeepOnto evaluator
    by default -- which requires the use of another JVM) dispatches to logmap_llm.evaluation.harness
    as a subprocess; again, this is for JVM isolation (due to DeepOnto); however, we should note that
    this is not !! strictly !! neccesary, since we also implement our own evalution under the
        CustomEvaluationEngine class in logmap_llm.evaluation.engines.custom
    The architectural design decision to use extendable evaluation engines is based on the knowledge
    that actually multiple evaluators exist, eg. 'MELT', 'HOBBIT', 'SEALS', 'DeepOnto', and so forth
        (users should be able to extend the EvaluationEngine class and make use of their preffered 
         evaluator of choice; many of which do utilise another JVM instance, so isolating the evaluate
         procedure as a subprocess is _arguably_ _fairly_ sensible, though, not required for instances
         that use the CustomEvaluationEngine -- worth noting)
    Note: Evaluation can be skipped by setting 'evaluate = false' under '[evaluation]' in config.toml
    """
    step("[Step 5] Evaluation")

    if not ctx.cfg.evaluation.evaluate:
        warn("Evaluation disabled in config")
        return EvaluationResult()

    if not ctx.config_path:
        warning("No config path available for evaluation subprocess")
        return EvaluationResult()

    cmd = [
        sys.executable, "-m", "logmap_llm.evaluation.harness",
        "--config", ctx.config_path,
    ]

    step("[Step 5] Running evaluation subprocess")
    
    proc = subprocess.run(cmd, capture_output=False)

    if proc.returncode != 0:
        warning(f"Evaluation subprocess exited with code {proc.returncode}")

    eval_results_path = os.path.join(
        ctx.cfg.outputs.logmapllm_output_dirpath,
        "evaluation_results.json",
    )
    
    if os.path.exists(eval_results_path):
        with open(eval_results_path) as fp:
            results_dict = json.load(fp)
        
        success(f"Evaluation results loaded from: {eval_results_path}")
        return EvaluationResult(results=results_dict)

    return EvaluationResult()
