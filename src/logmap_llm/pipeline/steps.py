'''
Ported from jd-extended, see:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/pipeline_steps.py

With some modifications (ready for future branches/features).

TODO: add appropriate comments & documentation
'''

from __future__ import annotations

import sys
import os
import subprocess
import json
import pandas as pd

from logmap_llm.log_utils import (
    fatal,
    critical,
    warning,
    step,
    success,
)
from logmap_llm.pipeline.paths import (
    PipelinePaths
)
from logmap_llm.pipeline.types import (
    AlignmentResult,
    PromptBuildResult,
    OracleResult,
    RefinementResult,
    EvaluationResult, # NOTE: stub at present.
)
from logmap_llm.constants import (
    PAIRS_SEPARATOR,
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
)


# NOTE: We DO NOT import oracle_prompt_building or onto_access here.
# Those modules transitively import owlready2, which has seemed to cause
# issues with JPype in the same process; prompt building runs in a subprocess.



def align(ctx) -> AlignmentResult:
    """
    Docstring for align

    `import bridging as br` allows for mapping LogMap IO between java and python (and vice versa)

    :param ctx: Description
    :type ctx: PipelineContext
    :return: Description
    :rtype: AlignmentResult
    """
    import logmap_llm.bridging as br

    step("Align ontologies and obtain mappings to ask an Oracle")

    match(ctx.cfg.pipeline.align_ontologies):

        case AlignMode.ALIGN:
            step("Performing fresh initial LogMap alignment")
            ctx.logmap.perform_alignment()
            result = AlignmentResult(
                m_ask_df=br.java_mappings_2_python(
                    ctx.logmap.get_mappings_for_llm()
                ),
                mappings=br.java_mappings_2_python(
                    ctx.logmap.get_mappings()
                )
            )

        case AlignMode.REUSE:
            step(f"Loading mappings to ask an oracle from file: {ctx.run_paths.logmap_m_ask()}")
            result = AlignmentResult(
                m_ask_df=br.load_m_ask_from_file(
                    ctx.run_paths.logmap_m_ask()
                ),
                mappings=pd.read_csv(
                    ctx.run_paths.logmap_mappings(),
                    sep=PAIRS_SEPARATOR,
                    header=None
                )
            )

        case AlignMode.BYPASS:
            warning("Bypassing initial LogMap alignment")
            result = AlignmentResult()

        case _:
            fatal(f"config: align_ontologies param not recognised: {ctx.cfg.pipeline.align_ontologies}")

    if result.n_mappings > 0:
        success(f"Number of mappings within the initial alignment: {result.n_mappings}")
        success(f"Number of mappings within M_ask: {result.n_m_ask}")

    return result


def prompt_build(ctx, initial_alignment: AlignmentResult) -> PromptBuildResult:
    """
    NOTE: this is getting messy, let's consider some refinements.
    """
    
    step("Build user prompts for oracle consultation")

    match ctx.cfg.pipeline.build_oracle_prompts:

        case PromptBuildMode.BUILD:
            
            step("Building fresh oracle user prompts (delegation to subprocess)")
            
            # NOTE: the prompt builder subprocess reads the config TOML 
            # so we pass it the original config path (the runner stores it on ctx)
            config_path = getattr(ctx, '_config_path', None)

            if not config_path:
                # raises ValueError
                # fatal("No config path available for subprocess dispatch")
                # or opt for a warning: (TODO)
                warning("No config path available for subprocess dispatch -- skipping")
            
            cmd = [
                sys.executable, "-m", "logmap_llm.pipeline.subp_prompt_builder",
                "--config", config_path,
            ]
            
            if ctx.track:
                cmd.extend(["--track", ctx.track])
        
            # NOTE: FROM MOST RECENT PIPELINE (TODO)
            # if args.reuse_align:
            #     build_script_cmd.append('--reuse-align')
            # if args.reuse_prompts:
            #     build_script_cmd.append('--reuse-prompts')
            # if track:
            #     build_script_cmd.extend(['--track', track])
            # if args.no_cache:
            #     build_script_cmd.append('--no-cache')
            # build_script_proc = subprocess.Popen(
            #     build_script_cmd,
            #     env=build_script_env,
            #     text=True,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.STDOUT,
            # )

            proc = subprocess.run(cmd, capture_output=False)
            if proc.returncode != 0:
                fatal(f"Stage 2 subprocess failed with return code {proc.returncode}")
            
            prompts_path = ctx.run_paths.prompts_json()
            
            if prompts_path.exists():
                with open(prompts_path, 'r') as fp:
                    result = PromptBuildResult(prompts=json.load(fp))
                success(f"LLM oracle user prompts loaded from: {prompts_path}")
            else:
                warning(f"Prompts file not found: {prompts_path}")
                result = PromptBuildResult()

        case PromptBuildMode.REUSE:
            step(f"Loading LLM oracle user prompts from file: {str(ctx.run_paths.prompts_json())}")
            with open(ctx.run_paths.prompts_json(), 'r') as fp:
                result = PromptBuildResult(
                    prompts=json.load(fp)
                )

        case PromptBuildMode.BYPASS:
            warning(f"Bypassing use of LLM oracle user prompts")
            result = PromptBuildResult()

        case _:
            fatal(f"config: build_oracle_prompts param not recognised: {ctx.cfg.pipeline.build_oracle_prompts}")

    if result.n_prompts > 0:
        success(f"Number of LLM oracle user prompts: {result.n_prompts}")

    return result


"""

# ORIGINAL VERSION (STAGE TWO):

def prompt_build(ctx, initial_alignment: AlignmentResult) -> PromptBuildResult:

    step("Build user prompts for oracle consultation")

    match(ctx.cfg.pipeline.build_oracle_prompts):

        case PromptBuildMode.BUILD:
            step("Building fresh oracle user prompts")
            result = PromptBuildResult(
                prompts=opb.build_oracle_user_prompts(
                    ctx.cfg.oracle.oracle_user_prompt_template_name,
                    ctx.cfg.alignmentTask.onto_source_filepath,
                    ctx.cfg.alignmentTask.onto_target_filepath,
                    initial_alignment.m_ask_df
                )
            )
            with open(ctx.run_paths.prompts_json(), 'w') as fp:
                json.dump(result.prompts, fp)
            success(f"LLM oracle user prompts saved to file: {str(ctx.run_paths.prompts_json())}")

        case PromptBuildMode.REUSE:
            step(f"Loading LLM oracle user prompts from file: {str(ctx.run_paths.prompts_json())}")
            with open(ctx.run_paths.prompts_json(), 'r') as fp:
                result = PromptBuildResult(
                    prompts=json.load(fp)
                )

        case PromptBuildMode.BYPASS:
            warning(f"Bypassing use of LLM oracle user prompts")
            result = PromptBuildResult()

        case _:
            fatal(f"config: build_oracle_prompts param not recognised: {ctx.cfg.pipeline.build_oracle_prompts}")

    if result.n_prompts > 0:
        success(f"Number of LLM oracle user prompts: {result.n_prompts}")
    
    return result

"""

        


def consult_oracle(ctx, initial_alignment: AlignmentResult, prompt_build_result: PromptBuildResult) -> OracleResult:

    """
    Docstring for consult_oracle

    :param ctx: Description
    :type ctx: PipelineContext
    :param initial_alignment: Description
    :type initial_alignment: AlignmentResult
    :param prompt_build_result: Description
    :type prompt_build_result: PromptBuildResult
    :return: Description
    :rtype: OracleResult
    """

    import logmap_llm.oracle.prompts.developer as dp
    import logmap_llm.oracle.consultation as oc

    step("Consult Oracle for mappings to ask")

    dev_prompt_template = dp.get_developer_prompt(ctx.cfg.oracle.oracle_dev_prompt_template_name)

    match(ctx.cfg.pipeline.consult_oracle):

        case ConsultMode.CONSULT:
            step(f"Consulting LLM oracle with model: {ctx.cfg.oracle.model_name}")
            oracle_predictions_df, oracle_params_dict = oc.consult_oracle_for_mappings_to_ask(
                m_ask_oracle_user_prompts=prompt_build_result.prompts,
                api_key=ctx.cfg.oracle.openrouter_apikey,
                model_name=ctx.cfg.oracle.model_name,
                max_workers=ctx.cfg.oracle.max_workers,
                m_ask_df=initial_alignment.m_ask_df,
                base_url=ctx.cfg.oracle.base_url,
                enable_thinking=ctx.cfg.oracle.enable_thinking,
                interaction_style=ctx.cfg.oracle.interaction_style,
                developer_prompt_text=dev_prompt_template
            )
            result = OracleResult(
                predictions=oracle_predictions_df,
                oracle_params=oracle_params_dict
            )
            if result.predictions is not None:
                result.predictions.to_csv(ctx.run_paths.predictions_csv())
                success(f"Oracle predictions for 'mappings to ask' saved to file: {ctx.run_paths.predictions_csv()}")

        case ConsultMode.REUSE:
            step(f'Loading LLM oracle predictions for the mappings_to_ask from file: {ctx.run_paths.predictions_csv()}')
            result = OracleResult(
                predictions=pd.read_csv(
                    ctx.run_paths.predictions_csv()
                )
            )

        case ConsultMode.LOCAL:
            step(f'Local oracle prediction .csv file(s) will be loaded from directory: {ctx.cfg.oracle.local_oracle_predictions_dirpath}')
            result = OracleResult(
                local_dir=ctx.cfg.oracle.local_oracle_predictions_dirpath,
                predictions=None
            )

        case ConsultMode.BYPASS:
            warning('Bypassing oracle consultations')
            result = OracleResult()

        case _:
            fatal(f"config: consult_oracle param not recognised: {ctx.cfg.pipeline.consult_oracle}")

    if result.prediction_summary() is not None:
        success(f"\n\n{result.prediction_summary()}")
    else:
        warning("There is NO ORACLE PREDICTION SUMMARY available")

    return result




def refine_alignment(ctx, oracle_result: OracleResult) -> RefinementResult:

    """
    Docstring for refine_alignment

    `import bridging as br` allows for mapping LogMap IO between java and python (and vice versa)

    :param ctx: Description
    :type ctx: PipelineContext
    :param oracle_result: Description
    :type oracle_result: OracleResult
    :return: Description
    :rtype: RefinementResult
    """

    import logmap_llm.bridging as br

    step("refine alignment using oracle mapping predictions")

    ctx.logmap.set_output_dir(ctx.run_paths.refined_dir)

    match(ctx.cfg.pipeline.refine_alignment):

        case RefineMode.REFINE:

            if oracle_result.has_predictions:
                step("Refining initial LogMap alignment with LLM Oracle predictions")
                preds_java = br.python_oracle_mapping_predictions_2_java(oracle_result.predictions)
                step(f'Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}')
                ctx.logmap.refine_alignment(preds_java)
                mappings_java = ctx.logmap.get_mappings()
                step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}')
                result = RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            elif oracle_result.is_local:
                step("Refining initial LogMap alignment with local Oracle predictions")
                ctx.logmap.refine_alignment(str(oracle_result.local_dir))
                mappings_java = ctx.logmap.get_mappings()
                step(f'Number of mappings in LogMap refined alignment: {len(mappings_java)}')
                result = RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            else:
                critical("Check your config to ensure 'refine_alignment' is set appropriately!")
                critical("The oracle response is empty and no local predictions file is specified.")
                critical("This could result in a problem!")
                # JD: probably an IO Error is most appropriate here?
                fatal(f"Oracle payload passed to `refine_alignment` is either malformed or is empty.", IOError)

        case RefineMode.BYPASS:
            warning("Bypassing alignment refinement")
            result = RefinementResult()

        case _:
            fatal(f"config: refine_alignment param not recognised: {ctx.cfg.pipeline.refine_alignment}")

    return result



def evaluate(ctx) -> EvaluationResult:
    """
    Evaluate the refined alignment
    NOTE: stub.
    """
    step("Evaluation [stub has been run]")
    return EvaluationResult()
