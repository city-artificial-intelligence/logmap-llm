"""
logmap_llm.pipeline.runner

The main entry point for the LogMap-LLM pipeline. Conceptually, the execution 
flow can be understood as two overarching phases:

    (1) bootstrapping - where we collect experimental execution information
        from the provided config and initialise the neccesary modules / components
        that will be reused throughout the pipeline. These are all bundled into a
        'PipelineContext' and are passed between neach major phase of the pipeline.
    (2) Pipeline Phase 1 (init align) through -> phase 5 (evaluation) + then reporting

    BOOTSTRAPPING
    -------------
    1. parse arguments from the CLI (importantly looking for --config),
        validate the config (pydantic) & load it as a LogMapLLMConfig
    2. pass the LogMapLLMConfig to the PipelinePaths module (a static
        constructor); this is the module that ensures consistent filepath
        usage throughout the pipeline, regardless of the caller & ensures
        that the neccesary (uniquely pathed) directories exist for the
        experiments we intend to run (see large comment in misc.py)
    3. Start JVM ready for use \w LogMap & initialise the LogMap bridging
        interface (this has access to certain exposed java methods in LogMap,
        such as performAlignment, that we need to be able to call from python)
    4. begin recording the output (logging) - see utils.logging
    5. pass the LogMapLLMConfig, PipelinePaths, LogMapInterface and the config 
        to a PipelineContext (ctx) constructor -- you can think this (ctx) as 
        useful global state, only in this case, its attached to a context obj
        that is passed around between different pipeline components. 

    PIPELINE PHASES
    ---------------

        1. align - obtains an initial alignment between the input ontologies
            and produces a set of 'mappings to ask' (or M_ask) for oracle consultation
         - Takes PipelineContext as input, and produces M_ask and all mappings as output

        2. prompt_build - spawns a subprocess that is responsible for building prompts
            from the previously described M_ask file; it uses an ontology-driven prompt
            building approach in combination with owlready2 (as an ontology-based data
            access layer -- see ontology.access.py and ontology.object.py). This phase
            is also responsible for constructing the 'few-shot' examples; these prompts
            and examples are then saved to disk. The subprocess is used to avoid issues
            between running multiple JVMs and owlready2 instances concurrently
         - Takes PipelineContext and align output as input, and produces prompts as output

        3. consult_oracle - leverages oracle.consultation (which, in turn leverages
            oracle.manager) to consult an LLM either via OpenRouter or locally (or on
            a local network -- tested inferrence backends include vLLM and SGLang)
            about mappings that LogMap was uncertain of, these are generally binary
            classification 'questions' (generated in the prior step) that include
            contextual information regarding entities within the ontologies.
         - Takes PipelineContext and prompt_build output as input, and produces predictions
            as output

        4. refinement - passes the predictions (answers) from the oracle consultation
            back to LogMap, so that LogMap can conduct any remaining conflict resolution
            and produce a 'refined alignment' which can then either be: (1) used in 
            downstream applications, or (2) compared agaisnt a reference alignment
            to obtain a set of scores (in the next [optional] phase)
         - Takes PipelineContext and predictions as input, produces a refined alignment
            as output

        5. evaluate - an optional stage that allows for the use of an EvaluationEngine
            (provided under logmap_llm.evaluation.engines) -- the final output alignment
            -- the refined alignment -- is passed to the evaluation harness, which will
            compute the following metrics: global precision, recall and F1 + oracle
            precison, recall, F1, sensitivity, specificity and Youdens J index; it also
            provides a breakdown of TP/FP/TN/[FN] and can also be used for evaluating
            refined alignments agsisnt partial gold standard reference alignments, such
            is the case in OAEI 2025 KG track (uses the semantics specified by the track)
         - Takes the PipelineContext and refined alignment as input, produces scores (noted
            above) as output, as well as saving them to disk for further inspection and 
            stratification.
"""
from __future__ import annotations

import sys
import os
import os.path
import time
from datetime import datetime, timezone
from argparse import Namespace

from logmap_llm.utils.logging import TeeWriter, info, step, success, critical, warning
from logmap_llm.config.loader import load_and_validate_config, print_config_summary
from logmap_llm.interface import start_jvm, LogMapInterface
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.orchestration import (
    align,
    prompt_build,
    consult_oracle,
    refine_alignment,
    evaluate,
)
from logmap_llm.pipeline.contracts import TimingRecord
from logmap_llm.pipeline.reporting import (
    print_timing_summary,
    print_experimental_parameters,
    write_results_file,
)


def main(args: Namespace | None = None) -> int:
    """
    Begin a full pipeline run.
    """

    # BOOTSTRAPPING
    ###############

    if args is None:
        from logmap_llm.pipeline.cli import parse_args
        args = parse_args()

    expr_run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # config is specified via the `--config` CLI argument, by specifying an absolute
    # or relative filepath, this overrides the default

    cfg = load_and_validate_config(
        args.config,
        reuse_align=args.reuse_align,
        reuse_prompts=args.reuse_prompts,
    )

    # RESOLVING FILE PATHS
    #######################
    # PipelinePaths (paths.py) manages (mappings, results, few-shot example, etc) file path
    # locations; this is esp. useful when running multiple LogMap-LLM python processes concurrently.
    # The paths in use are constructed by concatenating the task name, the prompt name, and an
    # optional suffix (e.g., for few-shot, we might use fs8_hard to mean 'few-shot', '8 examples', 
    # 'using hard contrastive example strategy'). paths.py is purposfully meant to be extended /
    # extendable and the suffix can be determined either when writing a config manually, OR
    # at the process orchestration level (ie. when you use an orchestrator to manage multiple 
    # LogMap-LLM pipelines in parallel) - we simply specify a 'feature set' -> 'suffix' mapping
    # thereby ensuring that filesystem artifacts remain (theoretically) isolated between each process.
    
    run_paths:PipelinePaths = PipelinePaths.from_config(cfg)

    if not run_paths.create_base_dirs():
        critical("Experimental directories CANNOT BE created OR DO NOT already exist!")

    # logging (see utils.logging)
    tee_branch_out:TeeWriter = TeeWriter(
        str(run_paths.run_log(expr_run_timestamp)),
        sys.stdout,
    )
    sys.stdout = tee_branch_out

    try:

        print_config_summary(cfg)
        
        info(f"Summary of File Paths:\n\n{run_paths.summary()}\n\n")

        # if logmap_dirpath is unset, it assumes the official logmap directory lives within
        # the project it assumes the official logmap directory lives within the project
        # root (or wherever you run this pipeline from) - required for LogMap params + java-deps

        logmap_dirpath = os.path.join(os.getcwd(), 'logmap')
        if cfg.alignmentTask.logmap_parameters_dirpath:
            logmap_dirpath = cfg.alignmentTask.logmap_parameters_dirpath

        start_jvm(logmap_dir=logmap_dirpath) # ready for LogMap

        logmap = LogMapInterface.create_from_cfg(cfg, logmap_dirpath)

        logmap.set_output_dir(run_paths.initial_dir) # required by config

        # the ontology domain is an optional nice-ity, it allows the prompts to specifically
        # state which types of ontologies are being aligned (biomedical, conference, kgs, etc)

        if cfg.alignmentTask.ontology_domain:
            info(f"Ontology domain: {cfg.alignmentTask.ontology_domain}\n")

        # at this level of abstration, the code is more-or-less self-documenting
        # however, it is helpful to review config/schema.py and pipeline/context.py 
        # to understand the what is passed between each phase via PipelineContext.

        pipeline_ctx = PipelineContext(cfg, run_paths, logmap, config_path=args.config)


        # PIPELINE PHASES
        #################
        # the pipeline phases (as described in the module doc-string) follow the pattern:
        # (1) start timer, (2) execute phase (imported from orchestration.py) (3) stop timer
        # (4) print completion message (5) continue to next phase

        info("LogMap-LLM pipeline starting.")

        timing = TimingRecord()


        # ALIGN
        #######

        align_start_time = time.time()

        align_result = align(pipeline_ctx)

        timing.align_seconds = time.time() - align_start_time
    
        if align_result.n_mappings > 0:
            success(f"Number of mappings within the initial alignment: {align_result.n_mappings}")
            success(f"Number of mappings within M_ask: {align_result.n_m_ask}")


        # PROMPT BUILD
        ##############

        prompt_build_start_time = time.time()

        prompt_build_result = prompt_build(pipeline_ctx, align_result)

        timing.prompt_build_seconds = time.time() - prompt_build_start_time

        if prompt_build_result.n_prompts > 0 and not prompt_build_result.bidirectional:
            success(f"Number of LLM oracle user prompts: {prompt_build_result.n_prompts}")

        if prompt_build_result.n_prompts > 0 and prompt_build_result.bidirectional:
            success_suffix = f" ({prompt_build_result.n_prompts // 2} candidates x 2 directions)"
            success(f"Number of LLM oracle user prompts: {prompt_build_result.n_prompts}{success_suffix}")


        # CONSULT ORACLE
        ################

        console_oracle_start_time = time.time()

        oracle_result = consult_oracle(
            pipeline_ctx, align_result, prompt_build_result
        )

        timing.consult_seconds = time.time() - console_oracle_start_time

        if oracle_result.prediction_summary() is not None:
            for summary_message in oracle_result.prediction_summary(return_list=True):
                success(summary_message)
            success(f"Oracle predictions saved to: {run_paths.predictions_csv()}")


        # REFINE ALIGNMENT
        ##################

        refine_alignment_start_time = time.time()

        refinement_result = refine_alignment(pipeline_ctx, oracle_result)

        timing.refine_seconds = time.time() - refine_alignment_start_time

        success("Refinement stage ending.") # TODO: check success \w cond


        # EVALUATE
        ##########

        evaluate_start_time = time.time()

        eval_result = evaluate(pipeline_ctx)

        timing.evaluate_seconds = time.time() - evaluate_start_time

        success("Evaluation stage ending.") # TODO: check success \w cond


        # REPORTING
        ###########

        all_task_times = [
            timing.align_seconds,
            timing.prompt_build_seconds,
            timing.consult_seconds,
            timing.refine_seconds,
            timing.evaluate_seconds,
        ]

        timing.total_seconds = sum(task_time for task_time in all_task_times if task_time is not None)

        n_consultations = (
            prompt_build_result.n_prompts
            if prompt_build_result.n_prompts > 0
            else 0
        )

        print_timing_summary(timing, n_consultations)

        print_experimental_parameters(
            cfg, oracle_result, prompt_build_result, timing
        )

        
        # WRITE FINAL RESULTS TO DISK
        #############################

        results_path = run_paths.run_results(expr_run_timestamp)

        write_results_file(
            results_path,
            cfg, timing,
            eval_result,
            oracle_result,
            prompt_build_result,
        )

        if refinement_result.n_refined_mappings > 0:
            success(f"Total refined mappings: {refinement_result.n_refined_mappings}")

        success("LogMap-LLM session ending")


    finally:
        sys.stdout = tee_branch_out.original_stdout
        tee_branch_out.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
