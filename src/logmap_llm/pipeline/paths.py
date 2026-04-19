"""
logmap_llm.pipeline.paths
"""
from __future__ import annotations

import os

from pathlib import Path
from logmap_llm.config.schema import LogMapLLMConfig


class PipelinePaths:
    """
    Manages pipeline artefact paths.

    Naming convention:
    - artefacts:      output_dir  / f"{task_name}-{oupt_name}-{suffix}"
    - initial aligns: initial_dir / f"{task_name}-logmap_mappings.txt"
    - m_ask:          initial_dir / f"{task_name}-logmap_mappings_to_ask_oracle_user_llm.txt"
    - refined aligns: refined_dir / f"{task_name}-logmap_mappings.tsv"
    """

    def __init__(
        self,
        output_dir: str | Path,
        initial_dir: str | Path,
        refined_dir: str | Path,
        task_name: str,
        oupt_name: str,
    ):
        self.output_dir = Path(output_dir)
        self.initial_dir = Path(initial_dir)
        self.refined_dir = Path(refined_dir)
        self.task_name = task_name
        self.oupt_name = oupt_name

    @classmethod
    def from_config(cls, cfg: LogMapLLMConfig) -> PipelinePaths:
        """Construct from a validated LogMapLLMConfig"""
        return cls(
            output_dir=cfg.outputs.logmapllm_output_dirpath,
            initial_dir=cfg.outputs.logmap_initial_alignment_output_dirpath,
            refined_dir=cfg.outputs.logmap_refined_alignment_output_dirpath,
            task_name=cfg.alignmentTask.task_name,
            oupt_name=cfg.prompts.cls_usr_prompt_template_name,
        )

    def _artifact(self, suffix: str) -> Path:
        return self.output_dir / f"{self.task_name}-{self.oupt_name}-{suffix}"

    def prompts_json(self) -> Path:
        return self._artifact("mappings_to_ask_oracle_user_prompts.json")

    def predictions_csv(self) -> Path:
        return self._artifact("mappings_to_ask_with_oracle_predictions.csv")

    def few_shot_json(self) -> Path:
        return self._artifact("few_shot_examples.json")

    def eval_json(self) -> Path:
        return self.output_dir / "evaluation_results.json"

    def run_log(self, timestamp: str) -> Path:
        return self.output_dir / f"pipeline_log_{timestamp}.txt"

    def subprocess_log(self, timestamp: str, subprocess_name: str = "UNSET") -> Path:
        return self.output_dir / f"subprocess_{subprocess_name}_{timestamp}.txt"

    def run_results(self, timestamp: str) -> Path:
        return self.output_dir / f"expr_run_results_{timestamp}.txt"

    def logmap_mappings(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings.txt"
    
    def logmap_mappings_tsv(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings.tsv"

    def logmap_m_ask(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings_to_ask_oracle_user_llm.txt"

    def refined_mappings_tsv(self) -> Path:
        return self.refined_dir / f"{self.task_name}-logmap_mappings.tsv"

    def create_base_dirs(self) -> bool:
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.initial_dir, exist_ok=True)
        os.makedirs(self.refined_dir, exist_ok=True)
        return (
            os.path.exists(self.output_dir) 
            and os.path.exists(self.initial_dir) 
            and os.path.exists(self.refined_dir)
        )

    def summary(self) -> str:
        """Human-readable summary of resolved artifact paths"""
        lines = [
            f"  Task                : {self.task_name}",
            f"  Prompt template     : {self.oupt_name}",
            f"  Output directory    : {self.output_dir}",
            f"  Initial align dir   : {self.initial_dir}",
            f"  Refined align dir   : {self.refined_dir}",
            f"  Prompts JSON        : {self.prompts_json().name}",
            f"  Predictions CSV     : {self.predictions_csv().name}",
            f"  Few-shot JSON       : {self.few_shot_json().name}",
        ]
        return "\n".join(lines)
