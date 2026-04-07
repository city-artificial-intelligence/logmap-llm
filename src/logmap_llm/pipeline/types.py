'''
logmap_llm.pipeline.types - specifies typed result containers for pipeline steps.

Ported from:
    
    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/pipeline_types.py


'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class AlignmentResult:
    """
    Docstring for AlignmentResult
    """
    m_ask_df: pd.DataFrame | None = None
    mappings: pd.DataFrame | None = None
    
    @property
    def n_mappings(self) -> int:
        return len(self.mappings) if self.mappings is not None else 0
    
    @property
    def n_m_ask(self) -> int:
        return self.m_ask_df.shape[0] if self.m_ask_df is not None else 0


@dataclass
class PromptBuildResult:
    """
    Docstring for PromptBuildResult
    """
    prompts: dict | None = None
    bidirectional: bool = False         # NOTE: stubbed for 'future' feature/s
    
    @property
    def n_prompts(self) -> int:
        return len(self.prompts) if self.prompts is not None else 0
    # TODO: consider how we need to modify the logic here for bidirectional
    # counts when bidirection == true.


@dataclass
class OracleResult:
    """
    Docstring for OracleResult
    """
    predictions: pd.DataFrame | None = None
    oracle_params: dict = field(default_factory=dict)
    local_dir: str | Path | None = None

    @property
    def has_predictions(self) -> bool:
        return self.predictions is not None

    @property
    def is_local(self) -> bool:
        return self.local_dir is not None

    def prediction_summary(self) -> str | None:
        """
        Docstring for prediction_summary

        :param self: Description
        :return: Description
        :rtype: str | None
        """
        if self.predictions is None:
            return None
        preds = self.predictions['Oracle_prediction']
        n = len(preds)
        n_err = int(sum(preds == 'error'))
        n_true = int(sum(preds == True))
        n_false = int(sum(preds == False))
        w = len(str(n))
        lines = [
            f"Mappings to ask an Oracle : {n}",
            f"LLM Oracle consultations  : {n - n_err}",
            f"Predicted True            : {str(n_true).rjust(w)}",
            f"Predicted False           : {str(n_false).rjust(w)}",
            f"Consultation failures     : {str(n_err).rjust(w)}",
        ]
        return "\n".join(lines)


@dataclass
class RefinementResult:
    """
    Docstring for RefinementResult
    """
    refined_mappings: pd.DataFrame | None = None

    @property
    def n_refined_mappings(self) -> int:
        return self.refined_mappings.shape[0] if self.refined_mappings is not None else 0
    
    @property
    def completed(self) -> bool:
        return self.refined_mappings is not None


@dataclass
class EvaluationResult:
    """
    Docstring for EvaluationResult

    NOTE: stub (at present)
    """
    metrics: dict = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return len(self.metrics) > 0


@dataclass
class TimingRecord:
    """
    Timing information for pipeline steps
    """
    align_seconds: float | None = None
    prompt_build_seconds: float | None = None
    consult_seconds: float | None = None
    refine_seconds: float | None = None
    evaluate_seconds: float | None = None
    total_seconds: float | None = None

