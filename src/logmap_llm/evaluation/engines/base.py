from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class EvaluationEngine(ABC):

    @abstractmethod
    def name(self) -> str:
        """
        Return a short identifier for this engine.
        Used in the 'source' field of returned metric dicts.
        """
        ...

    @abstractmethod
    def compute_global(
        self,
        system_path: Path,
        reference_path: Path,
        train_reference_path: Path | None = None,
        **options,
    ) -> dict:
        """
        Compute global precision, recall, and F1 for an alignment.
        """
        ...

    @abstractmethod
    def compute_oracle(
        self,
        predictions: list[dict],
        reference_pairs: set[tuple[str, str]],
        **options,
    ) -> dict:
        """
        Compute oracle discrimination metrics for a set of predictions.
        """
        ...

    ###
    # OPTIONAL METHODS
    ###

    def compute_stratified_global(self, system_path: Path, reference_path: Path, **options) -> dict | None:
        """
        Optional: compute per-entity-type global metrics.
        """
        return None

    def compute_ranking(self, test_cands_path: Path, oracle_predictions_path: Path, **options) -> dict | None:
        """
        Optional: compute local ranking metrics (MRR, Hits@K).
        NOTE: this is a stub for BioML-based ranking (ie. \w onto embs).
        """
        return None

    ###
    # CAPABILITY QUERY
    ##################
    # check to see what 'this' engine supports.
    ###

    def supports(self, metric_name: str) -> bool:
        """
        Return True if this engine can compute the named metric.
        """
        return metric_name in {"global", "oracle"}
    
