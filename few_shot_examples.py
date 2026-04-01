"""
few-shot example generation for LLM oracle consultation

Builds few-shot examples as (user_prompt, assistant_response) pairs that are injected into 
the OpenAI messages list between the developer prompt and the real query. The advantage of
using a stateless API/protocol is that we can manage the context most effectively in this way.

Examples are formatted using the same template function as the main query to ensure format consistency.

For BioML:
Positive examples are sourced from train.tsv (known-correct equivalence mappings). 
Negative examples are generated via sibling replacement (hard) -- i.e., creating a plausible near-miss
    or by applying a column-swap (random) -- this is controlled by configuration (TOML) file
"""

import math
import random
import logging

import numpy as np
import pandas as pd

from onto_access import OntologyAccess
from onto_object import (
    OntologyEntryAttr,
    ClassNotFoundError,
    PropertyEntryAttr,
    PropertyNotFoundError,
    InstanceEntryAttr
)

logger = logging.getLogger(__name__)

####
# ANSWER FORMATS
####

# JSON response strings that match BinaryOutputFormat schema (true_false):
ANSWER_TRUE = '{"answer": true}'
ANSWER_FALSE = '{"answer": false}'

# JSON response strings that match YesNoOutputFormat schema (yes_no):
ANSWER_YES = '{"answer": "Yes"}'
ANSWER_NO = '{"answer": "No"}'

# match config to lookup answer format

_POSITIVE_ANSWER = {
    "true_false": ANSWER_TRUE,
    "yes_no": ANSWER_YES
}

_NEGATIVE_ANSWER = {
    "true_false": ANSWER_FALSE,
    "yes_no": ANSWER_NO
}

# maximum attempts to sample a non-colliding pair before giving up
# TODO: this should be a configurable parameter in config (TOML)
MAX_SAMPLE_RETRIES = 50


# TODO: build test cases for this
#       Actually, build test cases for this entire repo!
#       A test suite would be quite helpful...
class FewShotExampleBuilder:
    """
    builds few-shot examples for LLM oracle prompting

    Parameters
    ----------
    train_path : str
        Path to train.tsv (OAEI format: tab-separated, cols 0-1 are URIs).
    OA_source : OntologyAccess
        Pre-loaded source ontology.
    OA_target : OntologyAccess
        Pre-loaded target ontology.
    prompt_function : callable
        Bound template function matching signature
        ``(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str``.
    m_ask_uri_pairs : set of frozenset
        Direction-agnostic exclusion set built from M_ask. Any pair whose
        frozenset({src_uri, tgt_uri}) is in this set will not be sampled.
    negative_strategy : str
        ``"hard"`` for sibling-replacement negatives (requires sibling_selector),
        ``"random"`` for column-swap negatives.
    sibling_selector : SiblingSelector or None
        Required when negative_strategy is ``"hard"``. Used to find the most
        confusable sibling of the target concept.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_path: str,
        OA_source: OntologyAccess,
        OA_target: OntologyAccess,
        prompt_function: callable,
        m_ask_uri_pairs: set,
        negative_strategy: str = "hard", # default is hard
        sibling_selector=None,
        seed: int = 42,
        answer_format: str = "true_false", # TODO: should we modify default to yes_no?
    ):                                     # ^^^ this could be introducing issues (TODO: check this)
        self.OA_source = OA_source
        self.OA_target = OA_target
        self.prompt_function = prompt_function
        self._m_ask_exclusion = m_ask_uri_pairs
        self.negative_strategy = negative_strategy
        self.sibling_selector = sibling_selector
        self.rng = random.Random(seed)

        # answer strings matching the structured output schema
        self._answer_positive = _POSITIVE_ANSWER.get(answer_format, ANSWER_TRUE)
        self._answer_negative = _NEGATIVE_ANSWER.get(answer_format, ANSWER_FALSE)

        if negative_strategy == "hard" and sibling_selector is None:
            logger.warning(
                "negative_strategy='hard' but no sibling_selector provided; "
                "will fall back to 'random' for all negatives."
            )

        # load train.tsv — OAEI format, tab-separated, no header
        # ...may be 3 columns: (src, tgt, relation) or 5 columns (uri, uri, =, score, CLS)
        self._train_df = pd.read_csv(train_path, sep='\t', header=None, usecols=[0, 1])
        self._train_df.columns = ['src_uri', 'tgt_uri']

        # pre-filter: remove any train pairs that collide with M_ask (prevent leakage)
        before_count = len(self._train_df)
        self._train_df = self._train_df[
            self._train_df.apply(
                lambda r: not self._is_in_m_ask(r['src_uri'], r['tgt_uri']),
                axis=1,
            )
        ].reset_index(drop=True)
        removed = before_count - len(self._train_df)
        if removed > 0:
            logger.info(
                f"Removed {removed} train pairs that overlap with M_ask "
                f"({len(self._train_df)} remaining)"
            )

        if len(self._train_df) == 0:
            raise ValueError("No train pairs remaining after M_ask exclusion. Cannot build few-shot examples.")

        # shuffled index for sampling without replacement within a run
        self._indices = list(range(len(self._train_df)))
        self.rng.shuffle(self._indices)
        self._sample_cursor = 0

    def _is_in_m_ask(self, src_uri: str, tgt_uri: str) -> bool:
        """check if a URI pair (direction-agnostic) is in M_ask"""
        return frozenset({src_uri, tgt_uri}) in self._m_ask_exclusion

    def _next_train_pair(self) -> tuple[str, str]:
        """get the next train pair, cycling if exhausted"""
        if self._sample_cursor >= len(self._indices):
            self._sample_cursor = 0
        idx = self._indices[self._sample_cursor]
        self._sample_cursor += 1
        row = self._train_df.iloc[idx]
        return row['src_uri'], row['tgt_uri']

    def _resolve_single_entity(self, uri: str, onto: OntologyAccess):
        """
        resolve a single URI to a class, property, or instance entity

        Tries class -> property -> instance, mirroring onto_object.resolve_entity().
        Returns the entity object or None.
        """
        # first try class (most common for Bio-ML/Anatomy)
        try:
            return OntologyEntryAttr(uri, onto)
        except ClassNotFoundError:
            pass

        # then try property (Conference track)
        try:
            return PropertyEntryAttr(uri, onto)
        except (PropertyNotFoundError, Exception):
            pass

        # then try instance (KG track)
        try:
            ind = onto.getIndividualByURI(uri)
            if ind is not None:
                context = onto.getInstanceContext(uri)
                return InstanceEntryAttr(context=context, uri=uri, onto=onto)
            # fallback: check rdflib graph for DBkWik-style instances
            if hasattr(onto, 'hasSubjectInGraph') and onto.hasSubjectInGraph(uri):
                context = onto.getInstanceContext(uri)
                return InstanceEntryAttr(context=context, uri=uri, onto=onto)
        except Exception as e:
            logger.debug(f"Instance resolution failed for {uri}: {e}")

        return None

    def _resolve_entities(
        self, src_uri: str, tgt_uri: str
    ) -> tuple[OntologyEntryAttr, OntologyEntryAttr] | None:
        """
        resolve URI pair to entity objects (class, property, or instance)
        tries class resolution first, then property, then instance as fallback (needed for KG track). 
        
        Returns None if either URI cannot be resolved
        """
        src = self._resolve_single_entity(src_uri, self.OA_source)
        if src is None:
            logger.debug(f"Could not resolve source {src_uri}")
            return None

        tgt = self._resolve_single_entity(tgt_uri, self.OA_target)
        if tgt is None:
            logger.debug(f"Could not resolve target {tgt_uri}")
            return None

        return src, tgt

    def _format_example(
        self,
        src_entity: OntologyEntryAttr,
        tgt_entity: OntologyEntryAttr,
        label: bool,
    ) -> tuple[str, str]:
        """
        format a single example using the template function

        Returns (user_prompt_string, assistant_response_string)
        """
        prompt = self.prompt_function(src_entity, tgt_entity)
        answer = self._answer_positive if label else self._answer_negative
        return (prompt, answer)

    def _make_positive_example(self) -> tuple[str, str] | None:
        """
        sample and format a positive example from train.tsv
        Returns (user_prompt, assistant_answer) or None if no resolvable pair found <= MAX_SAMPLE_RETRIES
        """
        for _ in range(MAX_SAMPLE_RETRIES):
            src_uri, tgt_uri = self._next_train_pair()
            entities = self._resolve_entities(src_uri, tgt_uri)
            if entities is not None:
                return self._format_example(entities[0], entities[1], label=True)
        return None

    def _make_hard_negative(self) -> tuple[str, str] | None:
        """
        generate a hard negative via sibling replacement

        Takes a positive pair (C, D), replaces D with its most confusable sibling (by SapBERT similarity to C)
        If no sibling is found or sibling_selector is unavailable, falls back to random negative
        """
        if self.sibling_selector is None:
            return self._make_random_negative()

        for _ in range(MAX_SAMPLE_RETRIES):
            src_uri, tgt_uri = self._next_train_pair()
            entities = self._resolve_entities(src_uri, tgt_uri)
            if entities is None:
                continue

            src_entity, tgt_entity = entities

            # get siblings of the target concept
            try:
                tgt_parent_names = tgt_entity.get_direct_parents()
                if not tgt_parent_names:
                    continue

                # use the first parent to find siblings
                siblings = tgt_entity.get_siblings()
                if not siblings:
                    continue

                # pick a sibling that isn't the target itself and isn't in M_ask
                candidate_siblings = []
                for sib_name in siblings:
                    # resolve sibling URI — siblings are returned as name strings,
                    # we need the URI; use the target ontology to look up by name
                    sib_class = self.OA_target.getClassByName(sib_name)
                    if sib_class is None:
                        continue
                    sib_uri = sib_class.iri
                    if sib_uri == tgt_uri:
                        continue
                    if self._is_in_m_ask(src_uri, sib_uri):
                        continue
                    candidate_siblings.append(sib_uri)

                if not candidate_siblings:
                    continue

                # pick the first candidate 
                # Note: SiblingSelector already ranks by semantic similarity when available
                neg_tgt_uri = candidate_siblings[0]
                neg_entities = self._resolve_entities(src_uri, neg_tgt_uri)
                if neg_entities is not None:
                    return self._format_example(neg_entities[0], neg_entities[1], label=False)

            except Exception as e:
                logger.debug(f"Hard negative generation failed: {e}")
                continue

        # fallback to random negative if hard negative couldn't be generated
        logger.info("Hard negative generation exhausted retries; falling back to random")
        return self._make_random_negative()

    def _make_random_negative(self) -> tuple[str, str] | None:
        """
        generate a random negative via column-swap:
        Samples two different positive pairs and crosses their targets: (C1, D2) forms the negative
        """
        for _ in range(MAX_SAMPLE_RETRIES):
            src_uri_1, _ = self._next_train_pair()
            _, tgt_uri_2 = self._next_train_pair()

            # ensure we didn't accidentally create a real positive (kind of important!)
            if self._is_in_m_ask(src_uri_1, tgt_uri_2):
                continue

            entities = self._resolve_entities(src_uri_1, tgt_uri_2)
            if entities is not None:
                return self._format_example(entities[0], entities[1], label=False)

        return None

    def _make_negative_example(self) -> tuple[str, str] | None:
        """generate a negative example using the configured strategy"""
        if self.negative_strategy == "hard":
            return self._make_hard_negative()
        else:
            return self._make_random_negative()

    def build_examples(
        self, k: int, bidirectional: bool = False
    ) -> list[tuple[str, str]]:
        """
        build k few-shot examples

        Parameters
        ----------
        k : int
            Number of examples. 0 returns an empty list.
        bidirectional : bool
            If True, uses mod-2 interleaving: odd-indexed examples are a new
            pair in the primary direction, even-indexed are the same pair
            reversed. Unique pairs sampled = ceil(k / 2).

        Returns
        -------
        list of (user_prompt_string, assistant_response_string) tuples.
        The first example is always positive.
        """
        # TODO: consider simplifying this (REFACTOR)
        if k <= 0:
            return []

        examples = []

        if not bidirectional:
            # standard: first example positive, then alternate
            for i in range(k):
                if i % 2 == 0:
                    ex = self._make_positive_example()
                else:
                    ex = self._make_negative_example()

                if ex is None:
                    logger.warning(
                        f"Could not generate example {i+1}/{k}; "
                        f"stopping at {len(examples)} examples"
                    )
                    break
                examples.append(ex)
        else:
            # bidirectional mod-2 interleaving (complicates this slightly)
            # where each unique pair produces up to 2 examples (forward + reverse)
            n_unique_pairs = math.ceil(k / 2)

            for pair_idx in range(n_unique_pairs):
                is_positive = (pair_idx % 2 == 0)

                # sample the pair and build forward example
                if is_positive:
                    # get a resolvable positive pair
                    pair_example = None
                    for _ in range(MAX_SAMPLE_RETRIES):
                        src_uri, tgt_uri = self._next_train_pair()
                        entities = self._resolve_entities(src_uri, tgt_uri)
                        if entities is not None:
                            pair_example = entities
                            break
                else:
                    # get a negative pair
                    pair_example = None
                    if self.negative_strategy == "hard" and self.sibling_selector is not None:
                        # for hard negatives in bidirectional, we need the resolved entities to reverse them
                        for _ in range(MAX_SAMPLE_RETRIES):
                            src_uri, tgt_uri = self._next_train_pair()
                            entities = self._resolve_entities(src_uri, tgt_uri)
                            if entities is None:
                                continue
                            # try to get a sibling replacement
                            tgt_entity = entities[1]
                            try:
                                siblings = tgt_entity.get_siblings()
                                if siblings:
                                    for sib_name in siblings:
                                        sib_class = self.OA_target.getClassByName(sib_name)
                                        if sib_class is None:
                                            continue
                                        if sib_class.iri == tgt_uri:
                                            continue
                                        if self._is_in_m_ask(src_uri, sib_class.iri):
                                            continue
                                        neg_entities = self._resolve_entities(
                                            src_uri, sib_class.iri
                                        )
                                        if neg_entities is not None:
                                            pair_example = neg_entities
                                            break
                            except Exception:
                                pass
                            if pair_example is not None:
                                break
                    # fallback to random if hard failed
                    if pair_example is None:
                        for _ in range(MAX_SAMPLE_RETRIES):
                            src_uri_1, _ = self._next_train_pair()
                            _, tgt_uri_2 = self._next_train_pair()
                            if self._is_in_m_ask(src_uri_1, tgt_uri_2):
                                continue
                            entities = self._resolve_entities(src_uri_1, tgt_uri_2)
                            if entities is not None:
                                pair_example = entities
                                break

                if pair_example is None:
                    logger.warning(
                        f"Could not generate bidirectional pair {pair_idx+1}; "
                        f"stopping at {len(examples)} examples"
                    )
                    break

                src_ent, tgt_ent = pair_example

                # forward direction
                fwd_prompt = self.prompt_function(src_ent, tgt_ent)
                fwd_answer = self._answer_positive if is_positive else self._answer_negative
                examples.append((fwd_prompt, fwd_answer))

                if len(examples) >= k:
                    break

                # reverse direction (same pair, swapped)
                rev_prompt = self.prompt_function(tgt_ent, src_ent)
                rev_answer = self._answer_positive if is_positive else self._answer_negative
                examples.append((rev_prompt, rev_answer))

                if len(examples) >= k:
                    break

        return examples







def build_few_shot_examples(
    train_path: str,
    OA_source: OntologyAccess,
    OA_target: OntologyAccess,
    prompt_function: callable,
    m_ask_df,
    k: int,
    bidirectional: bool = False,
    negative_strategy: str = "hard",
    sibling_selector=None,
    seed: int = 42,
    answer_format: str = "true_false",
) -> list[tuple[str, str]]:
    """
    convenience function to build few-shot examples in one call

    Parameters
    ----------
    train_path : str
        Path to train.tsv.
    OA_source, OA_target : OntologyAccess
        Pre-loaded ontologies.
    prompt_function : callable
        Bound template function.
    m_ask_df : pd.DataFrame
        The M_ask DataFrame (used to build the exclusion set).
    k : int
        Number of few-shot examples.
    bidirectional : bool
        Whether to use mod-2 interleaving for bidirectional templates.
    negative_strategy : str
        "hard" or "random".
    sibling_selector : SiblingSelector or None
        Required for hard negatives.
    seed : int
        Random seed.
    answer_format : str
        ``'true_false'`` or ``'yes_no'`` — controls the JSON answer strings
        in the few-shot assistant responses.

    Returns
    -------
    list of (user_prompt, assistant_response) string tuples.
    """
    if k <= 0:
        return []

    # build direction-agnostic exclusion set from M_ask
    m_ask_exclusion = set()
    for _, row in m_ask_df.iterrows():
        m_ask_exclusion.add(frozenset({row.iloc[0], row.iloc[1]}))

    builder = FewShotExampleBuilder(
        train_path=train_path,
        OA_source=OA_source,
        OA_target=OA_target,
        prompt_function=prompt_function,
        m_ask_uri_pairs=m_ask_exclusion,
        negative_strategy=negative_strategy,
        sibling_selector=sibling_selector,
        seed=seed,
        answer_format=answer_format,
    )

    return builder.build_examples(k=k, bidirectional=bidirectional)















# ----------------------------------------------------------------
# EXPERIMENTAL: Per-query few-shot support (hard-similar strategy)
# ----------------------------------------------------------------

# hard-similar aims to maximise the positive cosine similarity to find the most appropriate 
# positive example (in a RAG-like manner) since many positives may simply be irrelevant 
# to the current uncertain mapping.

def _get_entity_label(entity) -> str:
    """extract a single preferred label string from any entity type"""
    names = entity.get_preferred_names()
    if names:
        return min(names)
    # fallback for entities with no preferred name
    if hasattr(entity, 'thing_class') and hasattr(entity.thing_class, 'name'):
        return str(entity.thing_class.name)
    if hasattr(entity, 'uri'):
        # use the URI fragment as a last resort
        uri = entity.uri
        return uri.rsplit('#', 1)[-1].rsplit('/', 1)[-1]
    return str(entity)


class FewShotPoolBuilder:
    """
    build a candidate pool for per-query few-shot selection

    Unlike FewShotExampleBuilder (which produces a static K-element list), this builds 
    all viable candidates with pre-computed embeddings and optional hard negatives.
    Specifically: this enables per-query selection at consultation time.

    Parameters
    ----------
    train_path : str
        Path to train.tsv or anchor-derived TSV.
    OA_source, OA_target : OntologyAccess
        Pre-loaded ontologies.
    prompt_function : callable
        Bound template function.
    m_ask_uri_pairs : set of frozenset
        Direction-agnostic exclusion set from M_ask.
    sibling_selector : SiblingSelector
        Used for hard-negative generation AND for computing embeddings.
    seed : int
        Random seed.
    answer_format : str
        'true_false' or 'yes_no'.
    """

    # TODO: refactor to reduce the amount of duplicated code

    def __init__(
        self,
        train_path: str,
        OA_source: OntologyAccess,
        OA_target: OntologyAccess,
        prompt_function: callable,
        m_ask_uri_pairs: set,
        sibling_selector,
        seed: int = 42,
        answer_format: str = "true_false", # TODO: see prior comment, change to yes_no default?
    ):
        self.OA_source = OA_source
        self.OA_target = OA_target
        self.prompt_function = prompt_function
        self._m_ask_exclusion = m_ask_uri_pairs
        self.sibling_selector = sibling_selector
        self.rng = random.Random(seed)

        self._answer_positive = _POSITIVE_ANSWER.get(answer_format, ANSWER_TRUE)
        self._answer_negative = _NEGATIVE_ANSWER.get(answer_format, ANSWER_FALSE)

        # load train/anchor pairs
        self._train_df = pd.read_csv(
            train_path, sep='\t', header=None, usecols=[0, 1]
        )
        self._train_df.columns = ['src_uri', 'tgt_uri']

        # pre-filter M_ask collisions
        before_count = len(self._train_df)
        self._train_df = self._train_df[
            self._train_df.apply(
                lambda r: frozenset({r['src_uri'], r['tgt_uri']}) not in m_ask_uri_pairs,
                axis=1,
            )
        ].reset_index(drop=True)
        removed = before_count - len(self._train_df)
        if removed > 0:
            logger.info(
                f"Pool builder: removed {removed} train pairs overlapping M_ask "
                f"({len(self._train_df)} remaining)"
            )

        # reuse the same resolution helper as FewShotExampleBuilder
        self._resolver = FewShotExampleBuilder.__new__(FewShotExampleBuilder)
        self._resolver.OA_source = OA_source
        self._resolver.OA_target = OA_target

    def build_pool(self) -> dict:
        """
        build the full candidate pool

        Returns
        -------
        dict with keys:
            'candidates': list of dicts, each containing:
                'src_uri': str
                'tgt_uri': str
                'positive_prompt': str
                'positive_answer': str
                'negative_prompt': str or None
                'negative_answer': str or None
                'pair_label': str  (concatenated labels for embedding)
            'embeddings': np.ndarray of shape (n_candidates, embed_dim)
        """
        candidates = []
        embeddings_list = []

        for _, row in self._train_df.iterrows():
            src_uri, tgt_uri = row['src_uri'], row['tgt_uri']

            # resolve entities
            entities = self._resolver._resolve_entities(src_uri, tgt_uri)
            if entities is None:
                continue

            src_entity, tgt_entity = entities

            # format positive prompt
            try:
                positive_prompt = self.prompt_function(src_entity, tgt_entity)
            except Exception as e:
                logger.debug(f"Pool: prompt formatting failed for {src_uri}: {e}")
                continue

            # attempt hard negative
            negative_prompt = None
            negative_answer = None
            try:
                if self.sibling_selector is not None and hasattr(tgt_entity, 'get_siblings'):
                    siblings = tgt_entity.get_siblings()
                    if siblings:
                        for sib in siblings:
                            sib_name = sib if isinstance(sib, str) else (
                                min(sib.get_preferred_names()) if hasattr(sib, 'get_preferred_names') else str(sib)
                            )
                            sib_class = self.OA_target.getClassByName(sib_name)
                            if sib_class is None:
                                continue
                            if sib_class.iri == tgt_uri:
                                continue
                            if frozenset({src_uri, sib_class.iri}) in self._m_ask_exclusion:
                                continue
                            neg_entities = self._resolver._resolve_entities(
                                src_uri, sib_class.iri
                            )
                            if neg_entities is not None:
                                negative_prompt = self.prompt_function(
                                    neg_entities[0], neg_entities[1]
                                )
                                negative_answer = self._answer_negative
                                break
            except Exception as e:
                logger.debug(f"Pool: hard negative failed for {src_uri}: {e}")

            # compute pair label and embedding
            src_label = _get_entity_label(src_entity)
            tgt_label = _get_entity_label(tgt_entity)
            pair_label = f"{src_label} {tgt_label}"

            embedding = self.sibling_selector._embed_single(pair_label)

            candidates.append({
                'src_uri': src_uri,
                'tgt_uri': tgt_uri,
                'positive_prompt': positive_prompt,
                'positive_answer': self._answer_positive,
                'negative_prompt': negative_prompt,
                'negative_answer': negative_answer,
                'pair_label': pair_label,
            })
            embeddings_list.append(embedding)

        logger.info(
            f"Pool built: {len(candidates)} candidates "
            f"({sum(1 for c in candidates if c['negative_prompt'] is not None)} "
            f"with hard negatives)"
        )

        embeddings = np.stack(embeddings_list) if embeddings_list else np.empty((0, 0))

        return {
            'candidates': candidates,
            'embeddings': embeddings,
        }


class FewShotPool:
    """
    per-query few-shot example selector

    Loaded from a serialised pool at consultation time (Step 3).
    For each query, ranks candidates by embedding similarity and returns 
    the K most similar examples with alternating positive/negative.

    Parameters
    ----------
    candidates : list of dict
        Each dict has keys: src_uri, tgt_uri, positive_prompt,
        positive_answer, negative_prompt, negative_answer, pair_label.
    embeddings : np.ndarray of shape (n_candidates, embed_dim)
        L2-normalised embeddings for each candidate pair.
    sibling_selector : SiblingSelector
        Used to embed query pairs at selection time.
    k : int
        Number of few-shot examples to select per query.
    seed : int
        Random seed for tie-breaking / random negative fallback.
    answer_format : str
        'true_false' or 'yes_no'.
    """

    def __init__(
        self,
        candidates: list,
        embeddings: np.ndarray,
        sibling_selector,
        k: int,
        seed: int = 42,
        answer_format: str = "true_false", # default change to yes_no? (TODO)
    ):
        self.candidates = candidates
        self.embeddings = embeddings
        self.sibling_selector = sibling_selector
        self.k = k
        self.rng = random.Random(seed)

        self._answer_positive = _POSITIVE_ANSWER.get(answer_format, ANSWER_TRUE)
        self._answer_negative = _NEGATIVE_ANSWER.get(answer_format, ANSWER_FALSE)

        # pre-compute frozensets for leakage exclusion
        self._candidate_pair_sets = [
            frozenset({c['src_uri'], c['tgt_uri']}) for c in candidates
        ]

    def select_for_query(
        self,
        query_src_label: str,
        query_tgt_label: str,
        query_src_uri: str,
        query_tgt_uri: str,
    ) -> list[tuple[str, str]]:
        """
        select K few-shot examples most similar to the query pair

        Returns
        -------
        list of (user_prompt, assistant_response) tuples.
        Alternates positive/negative. First example is always positive.

        Leakage prevention: candidates whose frozenset({src_uri, tgt_uri})
        matches the query pair are excluded.
        """
        if len(self.candidates) == 0 or self.k <= 0:
            return []

        # compute query embedding
        query_text = f"{query_src_label} {query_tgt_label}"
        query_emb = self.sibling_selector._embed_single(query_text)

        # compute similarities (dot product of L2-normalised vectors)
        similarities = self.embeddings @ query_emb

        # build exclusion set for the query
        query_pair = frozenset({query_src_uri, query_tgt_uri})

        # rank candidates by descending similarity, excluding query pair
        scored = []
        for idx, sim in enumerate(similarities):
            if self._candidate_pair_sets[idx] == query_pair:
                continue
            scored.append((idx, float(sim)))

        scored.sort(key=lambda x: x[1], reverse=True)

        # select top ceil(K/2) positive candidates
        n_positives = math.ceil(self.k / 2)
        selected_indices = [idx for idx, _ in scored[:n_positives]]

        # build examples: alternate positive / negative
        examples = []
        for candidate_idx in selected_indices:
            cand = self.candidates[candidate_idx]

            # positive example
            examples.append((cand['positive_prompt'], cand['positive_answer']))
            if len(examples) >= self.k:
                break

            # negative example (hard if available, else skip)
            if cand['negative_prompt'] is not None:
                examples.append((cand['negative_prompt'], cand['negative_answer']))
            else:
                # random negative fallback: pick a random candidate's positive as a mismatched negative
                fallback_indices = [
                    i for i in range(len(self.candidates))
                    if i != candidate_idx
                    and self._candidate_pair_sets[i] != query_pair
                ]
                if fallback_indices:
                    rand_idx = self.rng.choice(fallback_indices)
                    rand_cand = self.candidates[rand_idx]
                    examples.append((
                        rand_cand['positive_prompt'],
                        self._answer_negative,
                    ))

            if len(examples) >= self.k:
                break

        return examples[:self.k]