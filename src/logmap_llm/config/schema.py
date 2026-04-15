from __future__ import annotations

from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, ClassVar
from logmap_llm.constants import (
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
    RefinementStrategy,
    BinaryOutputFormat,
    BinaryOutputFormatWithReasoning,
    YesNoOutputFormat,
    YesNoOutputFormatWithReasoning,
    RESPONSE_FORMAT_FOR_ANSWER,
    ResponseModes,
    AnswerFormat,
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_RESPONSE_MODE,
)
from logmap_llm.utils.logging import warn


class AlignmentTaskConfig(BaseModel):
    task_name: str
    onto_source_filepath: str
    onto_target_filepath: str
    generate_extended_mappings_to_ask_oracle: bool = False
    logmap_parameters_dirpath: str = ""
    ontology_domain: str | None = None


class PromptTemplateConfig(BaseModel):
    '''
    For controlling which templates are used during oracle consultation.
    We assume that every track includes class-based alignment. Therefore,
    we provide a default configuration for this under both the cls_dev and
    cls_user prompt templates using the `class_equivalence` cls dev/system 
    prompt and the `synonyms_only` cls user prompt.
    '''
    cls_dev_prompt_template_name: str            = "class_equivalence"
    cls_usr_prompt_template_name: str            = "synonyms_only"
    prop_dev_prompt_template_name: Optional[str] = "property_equivalence"
    prop_usr_prompt_template_name: Optional[str] = None
    inst_dev_prompt_template_name: Optional[str] = "instance_equivalence"
    inst_usr_prompt_template_name: Optional[str] = None

    # (strategy) select from 'alphanumeric', 'shortest_label', 'sapbert', 'sbert'
    # defaults to 'sapbert' when ontology_domain := 'biomedical', 'sbert' otherwise
    sibling_strategy: Optional[str] = None

    # optional override of the embedding model; only used when sibling_strategy 
    # is 'sapbert' or 'sbert'; when None, the default checkpoint is used
    sibling_model: Optional[str] = None

    # the cost cap on the candidate sibling set prior to ranking
    #  None -> uses DEFAULT_MAX_SIBLING_CANDIDATES from constants.py
    sibling_max_candidates: Optional[int] = None


class FewShotConfig(BaseModel):
    '''
    For controlling the few-shot prompting mode during oracle consultation.
    By default `few_shot_k` is set to 0, which does not trigger any few-shot
    prompting mechanisms.
    '''
    few_shot_k: int                 = 0
    few_shot_seed: int              = 42
    few_shot_negative_strategy: str = "hard" # hard negatives produce near-miss contrastive examples (hand,random)


class OracleConfig(BaseModel):
    model_name: str        # <-- REQUIRED (provide in the config.toml)
    api_key: str = "EMPTY" # use 'EMPTY' for vLLM or the actual API key for your selected service
    
    # LLM parameters
    base_url: Optional[str] = "https://openrouter.ai/api/v1" # "http://localhost:8000/v1" (for vLLM & SGLang)
    supports_chat_template_kwargs: Optional[bool] = None     #
    failure_tolerance: Optional[int] = None                  #
    
    max_workers: Optional[int]            = 24               # Reccomended: start \w number of phsyical cores - 8
    enable_thinking: Optional[bool]       = False            # toggles thinking mode on/off for supported models
    max_completion_tokens: Optional[int]  = 32768 # 1000     # when reasoning, CoT may expend >> max_completion_tokens
    temperature: Optional[float]          = 0.0              # for _as deterministic as possible_ set to 0.0
    top_p: Optional[float]                = 1.0              # sampling (interacts \w temperature)
    reasoning_effort: Optional[str]       = "minimal"        # default minimal (check supported model docs for settings)
    local_oracle_predictions_dirpath: str = ""               #

    interaction_style: Optional[Literal["auto", "openrouter", "vllm"]] = "auto"

    '''
    Note, at present there is some _delicate_ global state in templates.py that depends on
    DEFAULT_ANSWER_FORMAT and DEFAULT_RESPONSE_MODE from constants.py, this is why we use
    these defaults here, just to ensure that the schema is aligned to the specified defaults
    when these values may be changed by a user. TODO: prepare a more appropriate implementation.
    '''
    answer_format: Literal[AnswerFormat.TRUE_FALSE, AnswerFormat.YES_NO]  = DEFAULT_ANSWER_FORMAT    # 'true_false'
    response_mode: Literal[ResponseModes.STRUCTURED, ResponseModes.PLAIN] = DEFAULT_RESPONSE_MODE    # 'structured'

    @property
    def response_format(self) -> BinaryOutputFormat | BinaryOutputFormatWithReasoning | YesNoOutputFormat | YesNoOutputFormatWithReasoning | None:
        if self.response_mode not in (ResponseModes.STRUCTURED, ResponseModes.PLAIN):
            raise ValueError("The specified `response_mode` is not supported.")
        if self.response_mode == "plain":
            return None
        return RESPONSE_FORMAT_FOR_ANSWER[
            (self.answer_format, self.enable_thinking)
        ]

    _NON_KWARG_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "local_oracle_predictions_dirpath",
    })

    @property
    def consult_kwargs(self) -> dict:
        kwargs_dict = self.model_dump(exclude=self._NON_KWARG_FIELDS)
        kwargs_dict["response_format"] = self.response_format
        return kwargs_dict

    @model_validator(mode="before")
    @classmethod
    def migrate_openrouter_model_name(cls, data: dict) -> dict:
        """Accept legacy `openrouter_model_name` key with a deprecation warning."""
        if isinstance(data, dict) and "openrouter_model_name" in data:
            if "model_name" not in data:
                warn("Config key 'openrouter_model_name' is deprecated; use 'model_name' instead.")
                data["model_name"] = data.pop("openrouter_model_name")
            else:
                data.pop("openrouter_model_name") # silently drops (both present)
        return data
    
    @model_validator(mode="before")
    @classmethod
    def migrate_openrouter_apikey(cls, data: dict) -> dict:
        """Accept legacy `openrouter_apikey` key with a deprecation warning."""
        if isinstance(data, dict) and "openrouter_apikey" in data:
            if "api_key" not in data:
                warn("Config key `openrouter_apikey` is deprecated; use `api_key` instead.")
                data["api_key"] = data.pop("openrouter_apikey")
            else:
                data.pop("openrouter_apikey") # silently drops (both present)
        return data


class OutputsConfig(BaseModel):
    """Configuration for output directory paths."""
    logmapllm_output_dirpath: str
    logmap_initial_alignment_output_dirpath: str
    logmap_refined_alignment_output_dirpath: str


class PipelineConfig(BaseModel):
    """Configuration for pipeline step modes."""
    align_ontologies: AlignMode = AlignMode.ALIGN
    build_oracle_prompts: PromptBuildMode = PromptBuildMode.BUILD
    consult_oracle: ConsultMode = ConsultMode.CONSULT
    refine_alignment: RefineMode = RefineMode.REFINE
    refinement_strategy: RefinementStrategy = RefinementStrategy.LOGMAP


class EvaluationConfig(BaseModel):
    """Configuration for the optional evaluation step."""
    evaluate: bool = False
    reference_alignment_path: Optional[str] = None
    train_alignment_path: Optional[str] = None
    test_cands_path: Optional[str] = None
    metrics: list[str] | str = Field(default_factory=lambda: ["global", "oracle"])
    force_custom_eval: bool = True                  #
    partial_reference: bool = False                 # for kg track: true
    stratified_by_entity_type: bool = False
    stratified_class_property: bool = False
    jvm_memory: str = "8g"

    @model_validator(mode="after")
    def normalise_metrics(self):
        """Accept comma-separated string or list for metrics."""
        if isinstance(self.metrics, str):
            self.metrics = [m.strip() for m in self.metrics.split(",")]
        return self


class LogMapLLMConfig(BaseModel):
    """
    Top-level configuration schema for LogMap-LLM.
    Allows for validation of the entire TOML config structure on construction.
    NOTE (TODO): should we not make all these default factories?
    """
    alignmentTask: AlignmentTaskConfig
    oracle: OracleConfig
    prompts: PromptTemplateConfig = Field(default_factory=PromptTemplateConfig)
    few_shot: FewShotConfig = Field(default_factory=FewShotConfig)
    outputs: OutputsConfig
    pipeline: PipelineConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    class Config:
        """Allow extra fields at the top level for forward compatibility."""
        extra = "ignore"


def validate_config(config_dict: dict) -> LogMapLLMConfig:
    """
    Validate the config dict and return a typed config object.
    """
    return LogMapLLMConfig.model_validate(config_dict)
