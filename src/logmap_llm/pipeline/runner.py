"""
logmap_llm.pipeline.runner — main pipeline entry point
orchestrates the five pipeline steps

essentially ported from:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/logmap_llm.py
"""
from __future__ import annotations

import sys
import os
import os.path
import time
from datetime import datetime, timezone
from argparse import Namespace

from logmap_llm.utils.logging import TeeWriter, info, step, success, critical
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
    
    """
    if args is None:
        from logmap_llm.pipeline.cli import parse_args
        args = parse_args()

    expr_run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # MANAGE (LOAD & VALIDATE) CONFIG

    config_path = args.config

    cfg = load_and_validate_config(
        config_path,
        reuse_align=args.reuse_align,
        reuse_prompts=args.reuse_prompts
    )

    # LOAD (FILE) PATH LOCATIONS FROM `cfg`

    run_paths = PipelinePaths.from_config(cfg)

    os.makedirs(run_paths.output_dir, exist_ok=True)
    os.makedirs(run_paths.initial_dir, exist_ok=True)
    os.makedirs(run_paths.refined_dir, exist_ok=True)

    # START LOGGING OUTPUT TO BOTH STD::OUT AND LOG FILE

    tee_branch_out = TeeWriter(
        str(run_paths.run_log(expr_run_timestamp)),
        sys.stdout
    )
    sys.stdout = tee_branch_out

    try:
        
        # PRINT EXP PARAMS:
        # recursively parses the loaded `cfg` and displays parameters
        # outputs all file path locations constructed for the exp run
        # (useful for log files, tracability and reproducibility)

        print_config_summary(cfg)
        
        info(f"\n\nSummary of File Paths:\n\n{run_paths.summary()}\n\n")

        # INITIALISE LOGMAP
        # LogMaps `parameters.txt` file should be present in the dir
        # specified under `alignmentTask.logmap_parameters_dirpath`
        # in config, otherwise the script assumes `LogMap` dir is in cwd

        if cfg.alignmentTask.logmap_parameters_dirpath:
            logmap_dirpath = cfg.alignmentTask.logmap_parameters_dirpath
        else:
            logmap_dirpath = os.path.join(os.getcwd(), 'logmap')

        # functional call to jpype (required for LogMap & bridging)
        start_jvm(logmap_dir=logmap_dirpath)
        
        # simple API for interfacing with LogMap via python
        logmap = LogMapInterface(cfg, logmap_dirpath)
        logmap.set_output_dir(run_paths.initial_dir)

        # resolve track from CLI override or config
        track = args.track or cfg.alignmentTask.track

        # build pipeline context
        pipeline_ctx = PipelineContext(cfg, run_paths, logmap, track=track)

        # store config path for subprocess dispatch
        pipeline_ctx._config_path = args.config  # type: ignore[attr-defined]

        # START LOGMAP SESSION (PIPELINE)
        # The pipeline consists of five steps:
        #   1. obtains an initial alignment (via LogMap) for the given task
        #      and identifies mappings that LogMap is highly uncertain of
        #      producing both an initial alignment and the M_ask mapping set
        #   2. consumes the output (M_ask) mappings, these used to construct
        #      prompts for an LLM which may include structural context. The
        #      `prompt_build` step then produces these prompts as output.
        #   3. `consult_oracle` then consumes the output prompts, by passing
        #      them to an LLM. The LLM responds by indicating whether it
        #      judges the alignment in question as correct. The set of LLM
        #      responses produce the output at this step.
        #   4. the LLM oracle output mappings are then consumed by the refine
        #      ment step, which LogMap then integrates into its final alignment.
        #      The final alignment is saved to disk, and is read by an evaluation
        #      process in the next step.
        #   5. Evaluation occurs by comparing the refined alignment to the
        #      reference alignment. Experimental results are saved to disk
        #      and streamed to std::out (and therefore also logged). These
        #      results are then later read by scripts for plotting and results
        #      table construction for analysis.

        info("LogMap-LLM pipeline starting.")

        timing = TimingRecord()

        # TODO: can't we wrap these in a `TimedTransaction` class or something?

        # STEP ONE: obtain initial alignment

        t0 = time.time()

        align_result = align(pipeline_ctx)
        
        timing.align_seconds = time.time() - t0

        # STEP TWO: consume uncertain mappings to produce M_ask prompts

        t0 = time.time()

        prompt_build_result = prompt_build(pipeline_ctx, align_result)
        
        timing.prompt_build_seconds = time.time() - t0

        # STEP THREE: consult oracle regarding M_ask to obtain mapping adjustments (responses)

        t0 = time.time()

        oracle_result = consult_oracle(pipeline_ctx, align_result, prompt_build_result)

        timing.consult_seconds = time.time() - t0

        # STEP FOUR: consume oracle LLM responses about M_ask to refine alignment 
        
        t0 = time.time()

        refinement_result = refine_alignment(pipeline_ctx, oracle_result)

        timing.refine_seconds = time.time() - t0

        # STEP FIVE: run the evaluation procedure, report and log metrics

        t0 = time.time()

        eval_result = evaluate(pipeline_ctx)

        timing.evaluate_seconds = time.time() - t0

        # (STEP SIX) REPORTING: compute total
        
        timing.total_seconds = sum(
            t for t in [
                timing.align_seconds,
                timing.prompt_build_seconds,
                timing.consult_seconds,
                timing.refine_seconds,
                timing.evaluate_seconds,
            ]
            if t is not None
        )

        # REPORTING: timing summary

        n_consultations = (
            prompt_build_result.n_prompts
            if prompt_build_result.n_prompts > 0
            else 0
        )

        print_timing_summary(timing, n_consultations)

        # (STEP SEVEN) REPORTING: experimental parameters

        print_experimental_parameters(cfg, oracle_result, prompt_build_result, timing)

        # (STEP EIGHT) REPORTING: write results file

        results_path = run_paths.run_results(expr_run_timestamp)

        write_results_file(
            results_path, cfg, timing, eval_result, oracle_result,
            prompt_build_result,
        )

        step(f"[Summary] Total refined mappings: {refinement_result.n_refined_mappings}\n")

        success("LogMap-LLM session ending")

    finally:
        sys.stdout = tee_branch_out.original_stdout
        tee_branch_out.close()

    return 0


if __name__ == "__main__":
    # NOTE: argparse moved to cli.py
    sys.exit(main())
