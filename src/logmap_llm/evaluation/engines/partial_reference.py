"""
logmap_llm.evaluation.engines.partial_reference
Evaluation engine for alignment tasks that use a partial gold standard (ie. KG track)
"""
from __future__ import annotations

from pathlib import Path

from logmap_llm.evaluation.engines.base import EvaluationEngine
from logmap_llm.evaluation.io import load_mapping_pairs
from logmap_llm.evaluation.metrics import (
    compute_kg_partial_prf,
    compute_oracle_metrics,
    classify_uri_entity_type,
)


class PartialReferenceEvaluationEngine(EvaluationEngine):
    """
    Evaluation engine for tasks with a partial gold standard reference.
    """

    def name(self) -> str:
        return "partial_reference"


    def compute_global(self, system_path: Path, reference_path: Path, train_reference_path: Path | None = None, **options) -> dict:
        """
        Compute global metrics using partial-GS semantics.
        """
        if train_reference_path is not None:
            raise ValueError(
                "PartialReferenceEvaluationEngine does not accept "
                "train_reference_path: a partial gold standard does not "
                "split into train and test references."
            )

        system = load_mapping_pairs(system_path)
        reference = load_mapping_pairs(reference_path)
        return compute_kg_partial_prf(system, reference)


    def compute_oracle(self, predictions: list[dict], reference_pairs: set[tuple[str, str]], **options) -> dict:
        """
        Compute oracle discrimination metrics with partial-GS scoping.
        """
        return compute_oracle_metrics(
            predictions,
            reference_pairs,
            partial_reference=True,
        )


    def compute_stratified_global(self, system_path: Path, reference_path: Path, **options) -> dict | None:
        ...


    def supports(self, metric_name: str) -> bool:
        if metric_name == "stratified_global":
            return True
        return super().supports(metric_name)