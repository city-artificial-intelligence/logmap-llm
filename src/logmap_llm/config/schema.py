from __future__ import annotations

import warnings
from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional

from logmap_llm.constants import (
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
)


class AlignmentTaskConfig(BaseModel):
    """Configuration for the alignment task"""
    task_name: str
    onto_source_filepath: str
    onto_target_filepath: str
    generate_extended_mappings_to_ask_oracle: bool = False
    logmap_parameters_dirpath: str = ""
    track: str | None = None


class OracleConfig(BaseModel):
    """Configuration for the oracle"""
    openrouter_apikey: str = ""
    model_name: str
    oracle_dev_prompt_template_name: str = "class_equivalence"
    oracle_user_prompt_template_name: str
    local_oracle_predictions_dirpath: str = ""
    # Optional LLM parameters
    base_url: Optional[str] = None
    enable_thinking: Optional[bool] = None
    interaction_style: Optional[Literal["auto", "openrouter", "vllm"]] = None
    max_completion_tokens: Optional[int] = None
    failure_tolerance: Optional[int] = None
    max_workers: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    reasoning_effort: Optional[str] = None
    # Few-shot parameters
    few_shot_k: int = 0
    few_shot_seed: int = 42
    few_shot_negative_strategy: str = "hard"

    @model_validator(mode="before")
    @classmethod
    def migrate_openrouter_model_name(cls, data: dict) -> dict:
        """Accept legacy ``openrouter_model_name`` key with a deprecation warning."""
        if isinstance(data, dict) and "openrouter_model_name" in data:
            if "model_name" not in data:
                warnings.warn(
                    "Config key 'openrouter_model_name' is deprecated; "
                    "use 'model_name' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                data["model_name"] = data.pop("openrouter_model_name")
            else:
                data.pop("openrouter_model_name")
        return data


class OutputsConfig(BaseModel):
    """Configuration for output directory paths"""
    logmapllm_output_dirpath: str
    logmap_initial_alignment_output_dirpath: str
    logmap_refined_alignment_output_dirpath: str


class PipelineConfig(BaseModel):
    """Configuration for pipeline step modes"""
    align_ontologies: AlignMode = AlignMode.ALIGN
    build_oracle_prompts: PromptBuildMode = PromptBuildMode.BUILD
    consult_oracle: ConsultMode = ConsultMode.CONSULT
    refine_alignment: RefineMode = RefineMode.REFINE


class EvaluationConfig(BaseModel):
    """Configuration for the optional evaluation step"""
    evaluate: bool = False
    reference_alignment_path: Optional[str] = None
    train_alignment_path: Optional[str] = None
    test_cands_path: Optional[str] = None
    metrics: list[str] | str = Field(default_factory=lambda: ["global", "oracle"])
    force_custom_eval: bool = False
    jvm_memory: str = "8g"

    @model_validator(mode="after")
    def normalise_metrics(self):
        """Accept comma-separated string or list for metrics."""
        if isinstance(self.metrics, str):
            self.metrics = [m.strip() for m in self.metrics.split(",")]
        return self


class LogMapLLMConfig(BaseModel):
    """Top-level configuration schema for LogMap-LLM"""
    alignmentTask: AlignmentTaskConfig
    oracle: OracleConfig
    outputs: OutputsConfig
    pipeline: PipelineConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    class Config:
        """Allow extra fields at the top level for forward compatibility"""
        extra = "ignore"


def validate_config(config_dict: dict) -> LogMapLLMConfig:
    """Validate the config dict and return a typed config object"""
    return LogMapLLMConfig.model_validate(config_dict)
