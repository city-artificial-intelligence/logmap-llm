'''
Ported from:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/pipeline_utils.py

TODO: this needs to be well-documented.
'''

from pathlib import Path
from logmap_llm.config.schema import LogMapLLMConfig


class PipelinePaths:
    """
    Manages pipeline artefact paths.

    Uses the following naming convention:

    - artefacts      : output_dir / f"{self.task_name}-{self.oupt_name}-{suffix}" / * files
        ->  LogMap-LLM outputs (prompts, predictions, logs, eval)

    - initial aligns : initial_dir / f"{self.task_name}-logmap_mappings.txt"
        ->  LogMap's initial alignment outputs

    - m_ask          : initial_dir / f"{self.task_name}-logmap_mappings_to_ask_oracle_user_llm.txt"
        ->  m_ask is bundled with inital alignment files)
    
    - refined aligns : refined_dir / f"{self.task_name}-logmap_mappings.tsv"
        ->  LogMap's refined alignment outputs

    Does not perform I/O. callers read/write the returned Paths directly.
    """
    
    def __init__(self, output_dir: str | Path, initial_dir: str | Path, refined_dir: str | Path, task_name: str, oupt_name: str):
        self.output_dir = Path(output_dir)
        self.initial_dir = Path(initial_dir)
        self.refined_dir = Path(refined_dir)
        self.task_name = task_name
        self.oupt_name = oupt_name

    @classmethod
    def from_config(cls, cfg: LogMapLLMConfig) -> "PipelinePaths":
        return cls(
            output_dir=cfg.outputs.logmapllm_output_dirpath,
            initial_dir=cfg.outputs.logmap_initial_alignment_output_dirpath,
            refined_dir=cfg.outputs.logmap_refined_alignment_output_dirpath,
            task_name=cfg.alignmentTask.task_name,
            oupt_name=cfg.oracle.oracle_user_prompt_template_name,
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

    def run_results(self, timestamp: str) -> Path:
        return self.output_dir / f"expr_run_results_{timestamp}.txt"

    def logmap_mappings(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings.txt"

    def logmap_m_ask(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings_to_ask_oracle_user_llm.txt"

    def refined_mappings_tsv(self) -> Path:
        return self.refined_dir / f"{self.task_name}-logmap_mappings.tsv"
    
    def summary(self) -> str:
        """
        human-readable summary of resolved artifact paths
        """
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