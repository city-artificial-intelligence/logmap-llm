from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class AlignmentResult:
    """Result of the alignment step."""
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
    """Result of the prompt building step."""
    prompts: dict | None = None
    bidirectional: bool = False

    @property
    def n_prompts(self) -> int:
        return len(self.prompts) if self.prompts is not None else 0


@dataclass
class OracleResult:
    """Result of the oracle consultation step (Step 3)."""
    predictions: pd.DataFrame | None = None
    oracle_params: dict = field(default_factory=dict)
    local_dir: str | Path | None = None
    bidirectional: bool = False

    @property
    def has_predictions(self) -> bool:
        return self.predictions is not None

    @property
    def is_local(self) -> bool:
        return self.local_dir is not None

    @property
    def n_consultations(self) -> int:
        """Number of idv. LLM calls"""
        if self.predictions is None:
            return 0
        n = len(self.predictions)
        return n * 2 if self.bidirectional else n

    def prediction_summary(self, return_list=False) -> str | None:
        """Human-readable summary of oracle predictions"""
        if self.predictions is None:
            return None

        preds: pd.Series = self.predictions['Oracle_prediction']

        n_preds   = len(preds)
        counts    = preds.value_counts()
        n_true    = int(counts.get(True,      0))
        n_false   = int(counts.get(False,     0))
        n_error   = int(counts.get('error',   0))
        n_skipped = int(counts.get('skipped', 0))

        successful_consults = n_preds - n_error - n_skipped
        consultation_suffix = "(unidirectional consultations)"

        if self.bidirectional:
            successful_consults *= 2
            consultation_suffix = f"({successful_consults // 2} x 2)"

        padding_width = len(str(max(n_preds, successful_consults)))
        pad = lambda x: str(x).rjust(padding_width)

        output_lines = [
            f"Mappings to ask an Oracle  : {pad(n_preds)}",
            f"Predicted True             : {pad(n_true)}",
            f"Predicted False            : {pad(n_false)}",
            f"Consultation failures      : {pad(n_error)}",
            f"LLM Oracle consultations   : {pad(successful_consults)} {consultation_suffix}",
        ]

        if n_skipped > 0:
            output_lines.append(f"Skipped (no prompt)        : {pad(n_skipped)}")

        if self.bidirectional:
            output_lines.append("")
            output_lines.append("Bidirectional details (aggregated via logical AND):")
            output_lines.append(f"  Equivalence (both True)    : {pad(n_true)}")
            output_lines.append(f"  Not equivalent (>=1 False) : {pad(n_false)}")

        return output_lines if return_list else "\n".join(output_lines)


@dataclass
class RefinementResult:
    """Result of the alignment refinement step."""
    refined_mappings: pd.DataFrame | None = None

    @property
    def n_refined_mappings(self) -> int:
        return self.refined_mappings.shape[0] if self.refined_mappings is not None else 0

    @property
    def completed(self) -> bool:
        return self.refined_mappings is not None


@dataclass
class EvaluationResult:
    """Result of the evaluation step."""
    metrics: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return len(self.metrics) > 0 or len(self.results) > 0


@dataclass
class TimingRecord:
    """Timing information for pipeline steps."""
    align_seconds: float | None = None
    prompt_build_seconds: float | None = None
    consult_seconds: float | None = None
    refine_seconds: float | None = None
    evaluate_seconds: float | None = None
    total_seconds: float | None = None
