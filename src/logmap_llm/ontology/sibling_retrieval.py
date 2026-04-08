"""
logmap_llm.ontology.sibling_retrieval — SapBERT/SBERT-based sibling selection

Selects the most relevant sibling concepts using SapBERT embedding similarity.
Siblings are ranked by cosine similarity to the concept's own label.
The returned siblings are the concept's nearest peers.

Dependencies:
    
    - transformers, torch (for SapBERT [CLS] embeddings)
    - sentence-transformers (for all-MiniLM-L12-v2 fallback)

"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

if TYPE_CHECKING:
    from logmap_llm.ontology.entities import ClassEntity

logger = logging.getLogger(__name__)

DEFAULT_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

DEFAULT_GENERAL_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


# TODO: it would be far more efficient to pre-compute the embeddings.
# rather than re-compute embeddings across each pipeline run (!!!)

# NOTE: SiblingSelector is essentially a retriever.


class SiblingSelector:
    """
    Select the most relevant siblings using embedding similarity.

    Uses [CLS] token embeddings (not mean pooling) following SapBERT's training procedure.
    Embeddings are L2-normalised so dot product equals cosine similarity.

    Embeddings are cached by entity IRI across a pipeline run.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.
    
    batch_size : int
        Batch size for encoding.
    
    max_length : int
        Maximum token length.
    """
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_SAPBERT_MODEL,
        batch_size: int = 64,
        max_length: int = 64,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading SiblingSelector model: {model_name_or_path}")
        logger.info(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        self.model.eval()

        self.batch_size = batch_size
        self.max_length = max_length
        self._cache: dict[str, np.ndarray] = {}

    def _embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single text string using [CLS] token.
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        norm = np.linalg.norm(cls_emb)
        if norm > 0:
            cls_emb = cls_emb / norm

        return cls_emb


    def _embed(self, label: str, iri: str | None = None) -> np.ndarray:
        """
        Embed a label, using cache if IRI is provided.
        """
        if iri and iri in self._cache:
            return self._cache[iri]
        emb = self._embed_single(label)
        if iri:
            self._cache[iri] = emb

        return emb


    def select_siblings(
        self,
        entity: ClassEntity,
        max_count: int = 2,
        max_candidates: int = 50,
    ) -> list[tuple[str, float]]:
        """
        Returns up to max_count sibling labels ranked by similarity,
        ie. a list of (sibling_preferred_label, cosine_similarity) tuples
        ranked by desc co-sim.
        """
        # get siblings via direct parent's children minus self
        all_siblings = set()
        for parent in entity.get_direct_parents():
            for child in parent.get_direct_children():
                if child != entity:
                    all_siblings.add(child)
        # limit candidates
        all_siblings = list(all_siblings)[:max_candidates]

        if not all_siblings:
            return []
        if len(all_siblings) <= max_count:
            return [(self._get_label(s), 1.0) for s in all_siblings]

        entity_label = self._get_label(entity)
        entity_iri = entity.annotation.get("uri")
        entity_emb = self._embed(entity_label, entity_iri)

        scored = []
        for sib in all_siblings:
            sib_label = self._get_label(sib)
            sib_iri = sib.annotation.get("uri")
            sib_emb = self._embed(sib_label, sib_iri)
            sim = float(entity_emb @ sib_emb) # dot product
            scored.append((sib_label, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_count]

    @staticmethod
    def _get_label(entity: ClassEntity) -> str:
        names = entity.get_preferred_names()
        return min(names) if names else str(entity.thing_class.name)
