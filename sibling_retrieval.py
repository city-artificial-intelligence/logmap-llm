'''
sibling retrieval module 

selects the most relevant sibling concepts using SapBERT embedding similarity.
used by multiple oracle prompt templates

dependencies:
    
    - transformers (pip install transformers)
    - torch (pip install torch)
    - SapBERT model: cambridgeltl/SapBERT-from-PubMedBERT-fulltext (~440MB, downloads automatically on first use)

Note: SapBERT is NOT a sentence-transformer model, it uses [CLS] token embeddings \w standard HF BERT model.
      if you try and instanciate it as a sentence-transformer model, it will 'work', but this is actually a
      silent failure, since it will initialise a default sentence-transformer model and will likely go unnoticed
'''

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from onto_object import OntologyEntryAttr

logger = logging.getLogger(__name__)

# default SapBERT model identifier on HuggingFace Hub
DEFAULT_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# general-purpose model for non-biomedical tracks (e.g. Conference)
DEFAULT_GENERAL_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


class SiblingSelector:
    """
    select the most relevant siblings using SapBERT embedding similarity

    For a concept C with parent P, retrieves P's other children (siblings of C), 
    embeds each with SapBERT, and returns the top-k most similar to C by cosine similarity.
    The most similar siblings are the concept'snearest peers — the concepts most likely to be 
    confused with it which is maximally informative for disambiguation in the prompt.

    SapBERT uses **[CLS] token** embeddings (not mean pooling), following the training procedure 
    described in Liu et al. (2021).  Embeddings are L2-normalised so dot product equals cosine similarity.

    Embeddings are cached by entity IRI across a pipeline run so that a concept appearing in multiple 
    m_ask pairs is only embedded once

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model identifier or local path.
        Defaults to cambridgeltl/SapBERT-from-PubMedBERT-fulltext.
    batch_size : int
        Batch size for encoding multiple labels at once.
    max_length : int
        Maximum token length for the tokenizer.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_SAPBERT_MODEL,
        batch_size: int = 64,
        max_length: int = 128,
        pooling: str = "cls",
    ):
        from transformers import AutoTokenizer, AutoModel

        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got '{pooling}'")

        logger.info("Loading model: %s (pooling: %s)", model_name_or_path, pooling)

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModel.from_pretrained(model_name_or_path).to(self._device)
        self._model.eval()

        self._batch_size = batch_size
        self._max_length = max_length
        self._pooling = pooling
        self._cache: dict[str, np.ndarray] = {}

        logger.info(
            "Model loaded (device: %s, pooling: %s). Embedding cache initialised.",
            self._device, self._pooling,
        )


    @property
    def device(self) -> torch.device:
        """the device the model is running on"""
        return self._device

    # Embedding:

    def _embed_single(self, label: str, iri: str | None = None) -> np.ndarray:
        """
        embed a single label string, using cache if IRI is provided
        returns an L2-normalised embedding as a float32 numpy array
        uses [CLS] pooling for SapBERT or mean pooling for general models
        """
        if iri and iri in self._cache:
            return self._cache[iri]

        inputs = self._tokenizer(
            [label],
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        if self._pooling == "cls":
            # [CLS] token embedding (first token of last hidden state)
            emb = outputs.last_hidden_state[:, 0, :]
        else:
            # mean pooling (average over non-padding tokens)
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            emb = sum_embeddings / sum_mask

        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        emb = emb[0].cpu().numpy().astype(np.float32)

        if iri:
            self._cache[iri] = emb
        return emb

    def _embed_batch(
        self, labels: list[str], iris: list[str | None] | None = None
    ) -> list[np.ndarray]:
        """Embed a batch of labels, returning L2-normalised [CLS] embeddings.

        Uses the cache for any labels whose IRIs are already known.
        Uncached labels are batched together for efficient GPU inference.
        """
        if iris is None:
            iris = [None] * len(labels)

        results: list[np.ndarray | None] = [None] * len(labels)
        to_encode: list[tuple[int, str]] = []  # (original_index, label)

        # check cache first
        for i, (label, iri) in enumerate(zip(labels, iris)):
            if iri and iri in self._cache:
                results[i] = self._cache[iri]
            else:
                to_encode.append((i, label))

        # batch-encode uncached labels
        if to_encode:
            uncached_labels = [lbl for _, lbl in to_encode]
            uncached_embs = self._encode_labels(uncached_labels)

            for (orig_idx, _), emb in zip(to_encode, uncached_embs):
                results[orig_idx] = emb
                iri = iris[orig_idx]
                if iri:
                    self._cache[iri] = emb

        return results  # type: ignore[return-value]

    def _encode_labels(self, labels: list[str]) -> list[np.ndarray]:
        """Batch-encode a list of labels into L2-normalised embeddings."""
        all_embs: list[np.ndarray] = []

        for i in range(0, len(labels), self._batch_size):
            batch = labels[i : i + self._batch_size]

            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            if self._pooling == "cls":
                embs = outputs.last_hidden_state[:, 0, :]
            else:
                # mean pooling
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                embs = sum_embeddings / sum_mask

            embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
            all_embs.extend(embs.cpu().numpy().astype(np.float32))

        return all_embs


    def select_siblings(
        self,
        entity: OntologyEntryAttr,
        max_count: int = 2,
        max_candidates: int = 50,
    ) -> list[tuple[str, float]]:
        """
        return up to *max_count* sibling labels ranked by SapBERT similarity

        Parameters
        ----------
        entity : OntologyEntryAttr
            The concept whose siblings to select.
        max_count : int
            Maximum number of siblings to return.
        max_candidates : int
            Maximum number of raw siblings to retrieve from the ontology
            before ranking.  Caps the cost for concepts with very many
            siblings.

        Returns
        -------
        list of (sibling_preferred_label, cosine_similarity) tuples,
        sorted by descending similarity.  Empty list if the concept
        has no parents or no siblings.
        """
        all_siblings = entity.get_siblings(max_count=max_candidates)
        if not all_siblings:
            return []

        # if fewer siblings than max_count, no ranking needed
        if len(all_siblings) <= max_count:
            return [(self._get_label(s), 1.0) for s in all_siblings]

        # embedding
        entity_label = self._get_label(entity)
        entity_iri = entity.annotation.get("uri")
        entity_emb = self._embed_single(entity_label, entity_iri)

        # batch siblings
        sib_labels = [self._get_label(s) for s in all_siblings]
        sib_iris = [s.annotation.get("uri") for s in all_siblings]
        sib_embs = self._embed_batch(sib_labels, sib_iris)

        # score by cosine similarity (dot x of L2-normalised vectors)
        scored: list[tuple[str, float]] = []
        for label, emb in zip(sib_labels, sib_embs):
            sim = float(np.dot(entity_emb, emb))
            scored.append((label, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:max_count]

    @property
    def cache_size(self) -> int:
        """number of embeddings currently cached"""
        return len(self._cache)

    @staticmethod
    def _get_label(entity: OntologyEntryAttr) -> str:
        """extract a single preferred label string from an entity"""
        names = entity.get_preferred_names()
        return min(names) if names else str(entity.thing_class.name)