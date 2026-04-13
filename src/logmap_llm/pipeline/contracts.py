from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class AlignmentResult:
    """Result of the alignment step (Step 1)."""
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
    """Result of the prompt building step (Step 2)."""
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
        """Number of individual LLM calls made."""
        if self.predictions is None:
            return 0
        n = len(self.predictions)
        # in bidirectional mode, double the count:
        return n * 2 if self.bidirectional else n

    def prediction_summary(self) -> str | None:
        """Human-readable summary of oracle predictions."""
        if self.predictions is None:
            return None
        
        preds = self.predictions['Oracle_prediction']

        n = len(preds)
        n_err = int(sum(preds == 'error'))
        n_true = int(sum(preds == True))
        n_false = int(sum(preds == False))
        n_skipped = int(sum(preds == 'skipped'))
        
        padding_width = len(str(n))

        output_lines = [f"Mappings to ask an Oracle : {n}"]
        
        if self.bidirectional:
            output_lines.append(f"LLM Oracle consultations : {(n - n_err - n_skipped) * 2}"
                                f"({n - n_err - n_skipped} candidates x 2 directions)")
        else:
            output_lines.append(f"LLM Oracle consultations : {n - n_err - n_skipped}")

        output_lines.extend([
            f"Predicted True           : {str(n_true).rjust(padding_width)}",
            f"Predicted False          : {str(n_false).rjust(padding_width)}",
            f"Consultation failures    : {str(n_err).rjust(padding_width)}",
        ])

        if n_skipped > 0:
            output_lines.append(f"Skipped (no prompt)      : {str(n_skipped).rjust(padding_width)}")

        if self.bidirectional:
            output_lines.append("")
            output_lines.append("Bidirectional details (aggregated via logical AND):")
            output_lines.append(f"  Equivalence (both True)     : {n_true}")
            output_lines.append(f"  Not equivalent (>=1 False)  : {n_false}")

        return "\n".join(output_lines)


@dataclass
class RefinementResult:
    """Result of the alignment refinement step (Step 4)."""
    refined_mappings: pd.DataFrame | None = None

    @property
    def n_refined_mappings(self) -> int:
        return self.refined_mappings.shape[0] if self.refined_mappings is not None else 0

    @property
    def completed(self) -> bool:
        return self.refined_mappings is not None


@dataclass
class EvaluationResult:
    """Result of the evaluation step (Step 5)."""
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
