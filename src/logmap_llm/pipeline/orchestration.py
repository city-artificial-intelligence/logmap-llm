"""
logmap_llm.pipeline.orchestration

Ported from jd-extended, see:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/pipeline_steps.py

With some modifications (ready for future branches/features).

TODO: add appropriate comments & documentation
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



def align(ctx: PipelineContext) -> AlignmentResult:
    """
    Docstring for align

    `import bridging as br` allows for mapping LogMap IO between java and python (and vice versa)

    :param ctx: Description
    :type ctx: PipelineContext
    :return: Description
    :rtype: AlignmentResult
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

    return result



# TODO: migrate to utils py
def _detect_bidirectional(prompts: dict) -> bool:
    """
    Detect whether prompts were built in bidirectional mode.
    Bidirectional prompt keys have a direction suffix: `src_uri|tgt_uri|REVERSE`.
    Standard keys have just: `src_uri|tgt_uri`.
    """
    if not prompts:
        return False
    return any(key.count(PAIRS_SEPARATOR) >= 2 for key in prompts)



# TODO: migrate to utils py
def _validate_prompt_keys(prompts: dict, bidirectional: bool, template_name: str = "") -> None:
    """
    Validate that prompt key format matches the consultation mode.
    """
    if not prompts:
        return
    
    keys_have_direction = _detect_bidirectional(prompts)
    
    if bidirectional and not keys_have_direction:
        raise ValueError(
            f"Bidirectional template '{template_name}' selected but prompts "
            f"lack direction keys (e.g. '...|REVERSE'). The prompt JSON was "
            f"likely built with a non-bidirectional template. Rebuild prompts "
            f"or select a matching template."
        )
    if not bidirectional and keys_have_direction:
        raise ValueError(
            f"Standard template '{template_name}' selected but prompts have "
            f"direction keys (e.g. '...|REVERSE'). The prompt JSON was likely "
            f"built with a bidirectional template. Rebuild prompts or select "
            f"a matching template."
        )




def prompt_build(ctx: PipelineContext, initial_alignment: AlignmentResult) -> PromptBuildResult:
    """
    Step 2: Build user prompts for oracle consultation.

    In BUILD mode, dispatches to pipeline/stage_two.py as a subprocess to isolate owlready2 from JPype.

    Parameters
    ----------
    ctx : PipelineContext
    initial_alignment : AlignmentResult
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


    # TODO: migrate to runner.py
    if result.n_prompts > 0:

        if result.bidirectional:

            n_candidates = result.n_prompts // 2

            success(f"Number of LLM oracle user prompts: {result.n_prompts} ({n_candidates} candidates x 2 directions)")
        else:
            success(f"Number of LLM oracle user prompts: {result.n_prompts}")

    return result



def consult_oracle(ctx: PipelineContext, initial_alignment: AlignmentResult, prompt_build_result: PromptBuildResult) -> OracleResult:
    """
    Step 3: Consult Oracle for mappings to ask.

    Parameters
    ----------
    ctx : PipelineContext
    initial_alignment : AlignmentResult
    prompt_build_result : PromptBuildResult
    """
    import logmap_llm.oracle.prompts.developer as dp
    import logmap_llm.oracle.consultation as oc

    step("[Step 3] Consult Oracle for mappings to ask")

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
                result = OracleResult() 
                
                # NOTE: it would be nice if you could break from a switch in Python ...
                # (prefer guard clauses over if-else; could look to return TODO)
            
            else:

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

                result = OracleResult(
                    predictions=oracle_predictions_df,
                    oracle_params=oracle_kwargs,
                    bidirectional=prompt_build_result.bidirectional,
                )

                if result.predictions is not None:
                    result.predictions.to_csv(ctx.run_paths.predictions_csv(), na_rep="nan")    # ensure round-trip-ability
                    success(f"Oracle predictions saved to: {ctx.run_paths.predictions_csv()}")  # under pd.read_csv

        case ConsultMode.REUSE:

            step(f"[Step 3] Loading LLM oracle predictions from: {ctx.run_paths.predictions_csv()}")
            predictions_df = pd.read_csv(ctx.run_paths.predictions_csv())
            normalise_prediction_column(predictions_df) # pred.str_T/F/Y/N -> bool
            return OracleResult(predictions=predictions_df)

        case ConsultMode.LOCAL:

            step(f"[Step 3] Local oracle predictions from: {ctx.cfg.oracle.local_oracle_predictions_dirpath}")
            result = OracleResult(
                local_dir=ctx.cfg.oracle.local_oracle_predictions_dirpath,
                predictions=None,
            )

        case ConsultMode.BYPASS:

            warn("Bypassing oracle consultations")
            result = OracleResult()

        case _:
            fatal(f"config: consult_oracle param not recognised: {ctx.cfg.pipeline.consult_oracle}")


    # TODO: migrate to runner.py
    if result.prediction_summary() is not None:
        success(f"\n\n{result.prediction_summary()}")
    else:
        warning("There is NO ORACLE PREDICTION SUMMARY available")

    return result



def _kg_refine_in_python(ctx: PipelineContext, oracle_result: OracleResult) -> None:
    """
    Produce refined alignment for KG track in Python.

    Computes: refined = { initial - M_ask } U { m ∈ M_ask : oracle(m) = True }

    LogMap's Java refinement crashes with NullPointerExceptions when
    processing instance mappings (the bulk of the KG track), so we
    bypass it entirely.

    TODO: we need to test the new LogMap build by actually refining it, the NPE _should_ have disappeared 
    a this point.

    TODO: _kg_refine_in_python should handle both pipe and tab separators, or use the same format detection 
    logic as evaluate.py.
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



def refine_alignment(ctx: PipelineContext, oracle_result: OracleResult) -> RefinementResult:
    """
    Step 4: Refine alignment using oracle mapping predictions.

    Parameters
    ----------
    ctx : PipelineContext
    oracle_result : OracleResult
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

                result = RefinementResult()
                return result

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
                result = RefinementResult()
                return result

            if oracle_result.has_predictions:
                
                step("[Step 4] Refining initial LogMap alignment with LLM Oracle predictions")
                preds_java = br.python_oracle_mapping_predictions_2_java(
                    oracle_result.predictions
                )
                step(f"[Step 4] Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}")
                ctx.logmap.refine_alignment(preds_java)
                mappings_java = ctx.logmap.get_mappings()
                step(f"[Step 4] Number of mappings in LogMap refined alignment: {len(mappings_java)}")
                result = RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            elif oracle_result.is_local:
                
                step("[Step 4] Refining initial LogMap alignment with local Oracle predictions")
                ctx.logmap.refine_alignment(str(oracle_result.local_dir))
                mappings_java = ctx.logmap.get_mappings()
                step(f"[Step 4] Number of mappings in LogMap refined alignment: {len(mappings_java)}")
                result = RefinementResult(
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
            result = RefinementResult()

        case _:
            fatal(f"config: refine_alignment param not recognised: {ctx.cfg.pipeline.refine_alignment}")

    return result



def evaluate(ctx) -> EvaluationResult:
    """
    Step 5: Evaluate the refined alignment
    Dispatches to evaluation/evaluate.py as a subprocess (for JVM isolation due to DeepOnto)
    """
    step("[Step 5] Evaluation")

    if not ctx.cfg.evaluation.evaluate:
        warn("Evaluation disabled in config")
        return EvaluationResult()

    if not ctx.config_path:
        warning("No config path available for evaluation subprocess")
        return EvaluationResult()

    cmd = [
        sys.executable, "-m", "logmap_llm.evaluation.evaluate",
        "--config", ctx.config_path,
    ]

    step("[Step 5] Running evaluation subprocess")
    
    proc = subprocess.run(cmd, capture_output=False)

    if proc.returncode != 0:
        warning(f"Evaluation subprocess exited with code {proc.returncode}")

    # try to load the results
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
