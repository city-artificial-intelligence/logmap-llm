"""
logmap_llm.pipeline.reporting — post-pipeline reporting utilities
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from logmap_llm.log_utils import info, step
from logmap_llm.config.schema import LogMapLLMConfig
from logmap_llm.pipeline.types import (
    TimingRecord,
    EvaluationResult,
    OracleResult,
    PromptBuildResult,
)


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def format_timing_value(val: float | str | None) -> str:
    """Format a timing value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    return format_duration(val)


def classify_endpoint(base_url: str | None) -> str:
    """Classify an LLM endpoint based on its base URL."""
    if base_url is None:
        return "OpenRouter (default)"
    url = base_url.lower()
    if "openrouter" in url:
        return "OpenRouter"
    if "localhost" in url or "127.0.0.1" in url:
        return "Local (vLLM/SGLang)"
    return f"Custom ({base_url})"


def get_gpu_info() -> str:
    """Attempt to get GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "N/A"


def print_timing_summary(timing: TimingRecord, n_consultations: int) -> None:
    """Print a human-readable timing summary (Step 6 equivalent)."""
    step("[Step 6] Timing Summary")
    print()
    print(f"  Alignment          : {format_duration(timing.align_seconds)}")
    print(f"  Prompt building    : {format_duration(timing.prompt_build_seconds)}")
    print(f"  Oracle consultation: {format_duration(timing.consult_seconds)}")
    if n_consultations > 0 and timing.consult_seconds:
        per_consult = timing.consult_seconds / n_consultations
        print(f"    Per consultation : {format_duration(per_consult)}")
    print(f"  Refinement         : {format_duration(timing.refine_seconds)}")
    print(f"  Evaluation         : {format_duration(timing.evaluate_seconds)}")
    print(f"  Total              : {format_duration(timing.total_seconds)}")
    print()


def print_experimental_parameters(
    cfg: LogMapLLMConfig,
    oracle_result: OracleResult,
    prompt_result: PromptBuildResult,
    timing: TimingRecord,
) -> None:
    """Print experimental parameters (Step 7 equivalent)."""
    step("[Step 7] Experimental Parameters")
    print()
    print(f"  Task name           : {cfg.alignmentTask.task_name}")
    print(f"  Model               : {cfg.oracle.model_name}")
    print(f"  Endpoint            : {classify_endpoint(cfg.oracle.base_url)}")
    print(f"  Prompt template     : {cfg.oracle.oracle_user_prompt_template_name}")
    print(f"  Developer prompt    : {cfg.oracle.oracle_dev_prompt_template_name}")
    print(f"  N prompts           : {prompt_result.n_prompts}")
    if oracle_result.oracle_params:
        params = oracle_result.oracle_params
        print(f"  Interaction style   : {params.get('interaction_style', 'N/A')}")
        print(f"  Temperature         : {params.get('temperature', 'N/A')}")
        print(f"  Top-p               : {params.get('top_p', 'N/A')}")
        print(f"  Max workers         : {params.get('max_workers', 'N/A')}")
    print(f"  GPU                 : {get_gpu_info()}")
    print()


def write_results_file(
    filepath: Path,
    cfg: LogMapLLMConfig,
    timing: TimingRecord,
    eval_result: EvaluationResult,
    oracle_result: OracleResult,
    prompt_result: PromptBuildResult,
) -> None:
    """Write experiment results to a JSON file (Step 8 equivalent)."""
    results = {
        "task_name": cfg.alignmentTask.task_name,
        "model_name": cfg.oracle.model_name,
        "prompt_template": cfg.oracle.oracle_user_prompt_template_name,
        "developer_prompt": cfg.oracle.oracle_dev_prompt_template_name,
        "endpoint": classify_endpoint(cfg.oracle.base_url),
        "n_prompts": prompt_result.n_prompts,
        "oracle_params": oracle_result.oracle_params,
        "timing": {
            "align_seconds": timing.align_seconds,
            "prompt_build_seconds": timing.prompt_build_seconds,
            "consult_seconds": timing.consult_seconds,
            "refine_seconds": timing.refine_seconds,
            "evaluate_seconds": timing.evaluate_seconds,
            "total_seconds": timing.total_seconds,
        },
        "evaluation": eval_result.metrics,
    }

    with open(filepath, 'w') as fp:
        json.dump(results, fp, indent=2, default=str)

    step(f"[Step 8] Results written to: {filepath}")
