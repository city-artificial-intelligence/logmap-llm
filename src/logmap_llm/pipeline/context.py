"""
logmap_llm.pipeline.context — provides the pipeline execution context

(essentially) ported from:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/pipeline_steps.py

NOTE: This is quite similar to what you encounter in training pipelines
it's not atypical to see a huge 'config' or parameters object to be shared
between each stage in a training pipeline (think: wandb, etc).

NOTE: When this context object grows or needs to reach deeply nested 
modules, consider whether those modules should receive specific fields 
or should simply accept full context.
"""
from __future__ import annotations

from dataclasses import dataclass
from logmap_llm.config.schema import LogMapLLMConfig
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.interface import LogMapInterface


@dataclass
class PipelineContext:
    """Shared context threaded through all pipeline steps."""
    cfg: LogMapLLMConfig
    run_paths: PipelinePaths
    logmap: LogMapInterface
    config_path: str | None = None
