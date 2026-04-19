"""
logmap_llm.ontology.sibling_retrieval

Sibling selection with varying strategies; selects the top-k most "appropriate" siblings

Use in:
  
  (1) prompt context construction (hierarchical / sibling-context templates), and
  (2) few-shot hard-negative generation (contrastive near-miss examples)

NOTE: for hierarchical context construction we have direct access to parents ...
but we could use HiT_OnT : cls -> concept_emb; rank by s(C \sqsubseteq D) to 
obtain a set of plausible parents, take the set difference between the actual
parents, then use this to construct a near-miss for hierarchical context
.. its a bit complex like (and computationally demanding, since we use 
hyperbolic embs), but it could be worth exploring if near-miss case proves
very helpful ... though, the primary purpose would be for subsumption-based
ranking on BioML tasks (TODO) (<-- an aside)

The selection algorithm:

    1. Given C, gather direct parents of C: parents(C) [multi-inheritance aware]
    2. Union direct children of every parent into candidate set C^HAT_ch
    3. Sibling set S = C^HAT_ch \ {C}
    4. If |S| > max_candidates, slice S to max_candidates (cost cap).
    5. If |S| <= k, short-circuit and return all (score = 1.0).
    6. Otherwise rank S by the configured strategy and return top-k.

Four strategies are available, exposed via SiblingSelectionStrategy:

    ALPHANUMERIC      — sort by rdfs:label, ascending (naive baseline).
    SHORTEST_LABEL    — sort by len(rdfs:label), ascending (naive baseline).
    SAPBERT           — embed labels with SapBERT (CLS pooling, biomedical).
    SBERT             — embed labels with all-MiniLM-L12-v2 (mean pooling, generic).

The two embedding strategies share the same nearest-by-cosine ranking; they
differ only in the underlying encoder; embedding models are loaded lazily,
therefore, naive strategies pay no transformer-load cost.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

from logmap_llm.constants import (
    DEFAULT_SAPBERT_MODEL,
    DEFAULT_GENERAL_MODEL,
    DEFAULT_MAX_SIBLING_CANDIDATES,
    DEFAULT_TOP_K,
    VERBOSE,
    VERY_VERBOSE,
)
from logmap_llm.utils.logging import (
    debug,
    warn,
    warning,
    info,
    success,
    critical,
)
from logmap_llm.ontology.object import ClassEntity



class SiblingSelectionStrategy(str, Enum):
    """Strategy for ranking the candidate sibling set."""
    ALPHANUMERIC   = "alphanumeric"
    SHORTEST_LABEL = "shortest_label"
    SAPBERT        = "sapbert"
    SBERT          = "sbert"

    @property
    def is_embedding_based(self) -> bool:
        return self in (SiblingSelectionStrategy.SAPBERT, SiblingSelectionStrategy.SBERT)


###
# HELPERS
###


def _get_label(entity: ClassEntity) -> str:
    """
    Return the entity's preferred label, or a fragment of its IRI as fallback.
    NOTE: multiple preferred terms may be, for example: "Disease" and "Disease (SEMANTIC_TAG)"
    so applying min to the set of preferred terms may be _a little hacky_, but works for now
    """
    names = entity.get_preferred_names()
    return min(names) if names else str(entity.thing_class.name)


def _gather_siblings(entity: ClassEntity) -> set:
    """
    Multi-inheritance-aware sibling gathering:
        S = { cup_[p \in parents(C)] children(p) } \ { C }
    """
    siblings: set = set()
    for parent in entity.get_direct_parents():
        for child in parent.get_direct_children():
            if child != entity:
                siblings.add(child)
    return siblings


###
# MAIN (MODULE/CLASS)
###


class SiblingSelector:
    """
    Top-k sibling selection with a pluggable scoring strategy.
    
    usage (interface):

        selector = SiblingSelector(strategy=SiblingSelectionStrategy.SAPBERT)
        ranked   = selector.select_siblings(entity, max_count=2)
        # -> list[(label, score)] sorted by descending score

    embedding strategies cache embeddings by entity IRI across the run; 
    this helps when the same class appears in multiple M_ask candidates
    """

    def __init__(
        self,
        strategy: SiblingSelectionStrategy | str = SiblingSelectionStrategy.SAPBERT,
        model_name_or_path: str | None = None,
        max_candidates: int = DEFAULT_MAX_SIBLING_CANDIDATES,
        batch_size: int = 64,
        max_length: int = 64,
    ):
        self.strategy = SiblingSelectionStrategy(strategy)
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.max_length = max_length

        # embedding-only state (loaded lazily)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._pooling: str | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._model_name_or_path: str | None = None

        if self.strategy.is_embedding_based:
            self._init_embedding_backend(model_name_or_path)

        elif model_name_or_path is not None:
            warn(f"model_name_or_path={model_name_or_path!r} ignored: ")
            warn(f"  strategy {self.strategy.value} does not use an embedding model.\n")
        
        info(f"SiblingSelector ready ... with: ")
        info(f"  strategy={self.strategy.value}, max_candidates={self.max_candidates}.\n")


    ###
    # EMBEDDING BACKEND (lazy)
    ###

    def _init_embedding_backend(self, model_name_or_path: str | None) -> None:

        # imports kept local so that the naive strategies do not pay the
        # torch/transformers import cost when the user picks them (can be quite heavy)

        if VERBOSE:
            debug(f"(_init_embedding_backend) Importing torch; and transformers (AutoTokenizer, AutoModel).")

        import torch  # noqa: WPS433
        from transformers import AutoTokenizer, AutoModel  # noqa: WPS433

        if model_name_or_path is None:
            
            model_name_or_path = (
                DEFAULT_SAPBERT_MODEL
                if self.strategy is SiblingSelectionStrategy.SAPBERT
                else DEFAULT_GENERAL_MODEL
            )

        # pooling is implied by strategy:
        #   (1) SapBERT was trained with a contrastive [CLS] objective
        #   (2) sentence-transformers were trained with mean pooling
        #
        # mismatched pooling silently degrades embedding quality
        # we do not let the caller override this (as was previously supported)
        #   (if anything, belongs in expert config)

        self._pooling = "cls" if self.strategy is SiblingSelectionStrategy.SAPBERT else "mean"
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        info(f"Loading embedding backend: model={model_name_or_path} pooling={self._pooling} device={self._device}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self._model = AutoModel.from_pretrained(model_name_or_path).to(self._device)
        self._model.eval()
        self._model_name_or_path = model_name_or_path



    @property
    def device(self):
        """For backwards compatibility with previous logging in stage_two."""
        return self._device if self._device is not None else "cpu (no model loaded)"


    def _pool(self, last_hidden_state, attention_mask):
        if self._pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts and L2-normalise rows."""

        if VERBOSE and VERY_VERBOSE:
            debug(f"(_embed_batch) Importing torch.")

        import torch  # noqa: WPS433
        all_rows: list[np.ndarray] = []
        
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])
            arr = pooled.cpu().numpy()
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            all_rows.append(arr / norms)

        return np.vstack(all_rows) if all_rows else np.zeros((0, 0), dtype=np.float32)


    def _embed_with_cache(self, label: str, iri: str | None) -> np.ndarray:
        if iri is not None and iri in self._cache:
            return self._cache[iri]
        emb = self._embed_batch([label])[0]
        if iri is not None:
            self._cache[iri] = emb
        return emb


    ###
    # SELECTION
    ###


    def select_siblings(
        self,
        entity: ClassEntity,
        max_count: int = DEFAULT_TOP_K,
        max_candidates: int | None = None,
    ) -> list[tuple[ClassEntity, float]]:
        """
        Return up to max_count sibling ClassEntity objs ranked by the configured strategy
        max_candidates overrides the per-call cost cap (default: this inst max_candidates)
        note that every result is a (ClassEntity, score) tuple ranked by descending score
        callers that require labels only (ie. template fns) must call _get_label(class_entity)

        TODO: props and insts are !NOT! supported (at present); since the 'sibling' notion
        is (for the most part) class-hierarchy specific, ie. property hierarchies are often
        shallow and underspecified, and instances would require a separate type-aware walk: 
        ie. instance -> types -> sibling types -> instances of sibling types; also, this 
        would be unlikely to yield 'good' results, so for inst resolution, we will get better
        results by simply ranking all instances with a matching type identity ... for props
        we could probably do the same ... both extensions are planned (but not yet implemented)
        """
        if not isinstance(entity, ClassEntity):
            critical(f"Sibling selection for properties and instances is not yet implemented; see the docstring for the planned approach.")
            raise NotImplementedError(f"SiblingSelector.select_siblings is (at present) for CLS only (received {type(entity).__name__}).")

        cap = max_candidates if max_candidates is not None else self.max_candidates

        if VERBOSE and VERY_VERBOSE:
            debug(f"(select_siblings) max_candidates set to {str(cap)} and max_count set to {str(max_count)}.")

        # cls -> direct parent/s -> unionised children
        # siblings = list(_gather_siblings(entity)) # PREVIOUSLY NON DETERMINISTIC (# Fix:)
        siblings = sorted(_gather_siblings(entity), key=_get_label)
        if not siblings:
            return []
        if len(siblings) > cap:
            siblings = siblings[:cap]

        # dont rank if |S| < k
        if len(siblings) <= max_count:
            return [(sib, 1.0) for sib in siblings]

        # otherwise, rank by employed strategy
        return self._rank(entity, siblings, max_count)


    ###
    # RANKING STRATEGIES
    ###

    def _rank(self, entity: ClassEntity, siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        if self.strategy is SiblingSelectionStrategy.ALPHANUMERIC:
            return self._rank_alphanumeric(siblings, k)
        if self.strategy is SiblingSelectionStrategy.SHORTEST_LABEL:
            return self._rank_shortest_label(siblings, k)
        # embedding strategies share an implementation
        return self._rank_by_embedding(entity, siblings, k)


    @staticmethod
    def _rank_alphanumeric(siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        ranked = sorted(siblings, key=_get_label)
        return [(sib, 1.0) for sib in ranked[:k]]


    @staticmethod
    def _rank_shortest_label(siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        ranked = sorted(siblings, key=lambda s: (len(_get_label(s)), _get_label(s)))
        return [(sib, 1.0) for sib in ranked[:k]]


    def _rank_by_embedding(self, entity: ClassEntity, siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        """
        collects labels + IRIs and batches the uncaches entities in a single forward pass; then score 
        the entity emb @ sib emb (for all sib embs), maximise score (sort DESC) & return top-k as 
        (ClassEntity, score) pairs (so callers can look siblings up by IRI)
        """
        entity_label = _get_label(entity)
        entity_iri = entity.iri
        entity_emb = self._embed_with_cache(entity_label, entity_iri)

        sib_payload: list[tuple[ClassEntity, str, str | None]] = []
        for sib in siblings:
            sib_payload.append(
                (sib, _get_label(sib), sib.iri)
            )

        uncached_idx = []
        for idx, (_sib, _label, iri) in enumerate(sib_payload):
            if iri is None or iri not in self._cache:
                uncached_idx.append(idx)
        
        if uncached_idx:
            uncached_labels = [sib_payload[idx][1] for idx in uncached_idx]
            embs = self._embed_batch(uncached_labels)
            for slot, row in zip(uncached_idx, embs):
                _sib, _label, iri = sib_payload[slot]
                if iri is not None:
                    self._cache[iri] = row

        scored: list[tuple[ClassEntity, float]] = []
        
        for sib, label, iri in sib_payload:
            if iri is not None and iri in self._cache:
                emb = self._cache[iri]
            else:
                # no IRI fallback, re-embeds single label
                emb = self._embed_batch([label])[0]
            # score := dot product entity_emb \cdot sib_emb
            # ^^^^ cosine-sim (since embs are L_2 norm'd)
            scored.append((sib, float(entity_emb @ emb)))

        # maximise by score (sort by DESC), then return top-k
        scored.sort(key=lambda lt: lt[1], reverse=True)

        return scored[:k]