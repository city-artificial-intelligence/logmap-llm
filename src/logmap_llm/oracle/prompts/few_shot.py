"""
logmap_llm.oracle.prompts.few_shot — Few-shot example generation.

Builds few-shot examples as (user_prompt, assistant_response) pairs for 
injection between the developer prompt and the real query.

Formatted using the same template function as the main query.

Positive examples from train.tsv; negatives via sibling 
replacement (hard) or column-swap (random).
"""
from __future__ import annotations

import math
import random
import logging

import pandas as pd

from logmap_llm.ontology.access import OntologyAccess
from logmap_llm.ontology.entities import (
    ClassEntity,
    ClassNotFoundError,
    PropertyEntity,
    PropertyNotFoundError,
)

logger = logging.getLogger(__name__)

# NOTE: constants probably belong in constants.py

ANSWER_TRUE = '{"answer": true}'
ANSWER_FALSE = '{"answer": false}'
ANSWER_YES = '{"answer": "Yes"}'
ANSWER_NO = '{"answer": "No"}'

_POSITIVE_ANSWER = {"true_false": ANSWER_TRUE, "yes_no": ANSWER_YES}
_NEGATIVE_ANSWER = {"true_false": ANSWER_FALSE, "yes_no": ANSWER_NO}

# TODO: MAX_SAMPLE_RETRIES should be configurable

MAX_SAMPLE_RETRIES = 50


class FewShotExampleBuilder:
    """
    Builds few-shot examples for LLM oracle prompting.

    Parameters
    ----------
    train_path : str
        Path to train.tsv (tab-separated, cols 0-1 are URIs).
    
    OA_source : OntologyAccess
    
    OA_target : OntologyAccess
    
    prompt_function : callable
        Bound template function: (src_entity, tgt_entity) -> str.
    
    m_ask_uri_pairs : set of frozenset
        Direction-agnostic exclusion set from M_ask.
    
    negative_strategy : str
        "hard" for sibling-replacement, "random" for column-swap.
    
    sibling_selector : SiblingSelector or None
        Required when negative_strategy is "hard".
    
    seed : int
    
    answer_format : str
        "true_false" or "yes_no".
    """
    def __init__(
        self,
        train_path: str,
        OA_source: OntologyAccess,
        OA_target: OntologyAccess,
        prompt_function: callable,
        m_ask_uri_pairs: set,
        negative_strategy: str = "hard",    # TODO: default negative strategy set to hard (?)
        sibling_selector=None,
        seed: int = 42,
        answer_format: str = "true_false",
    ):
        self.OA_source = OA_source
        self.OA_target = OA_target
        self.prompt_function = prompt_function
        self.m_ask_uri_pairs = m_ask_uri_pairs
        self.negative_strategy = negative_strategy
        self.sibling_selector = sibling_selector
        self.answer_format = answer_format

        self._pos_answer = _POSITIVE_ANSWER.get(answer_format, ANSWER_TRUE)
        self._neg_answer = _NEGATIVE_ANSWER.get(answer_format, ANSWER_FALSE)

        self._rng = random.Random(seed)

        # load training pairs
        self._train_df = pd.read_csv(
            train_path, sep='\t', header=None, usecols=[0, 1]
        )
        self._train_pairs = list(
            zip(self._train_df.iloc[:, 0], self._train_df.iloc[:, 1])
        )
        self._rng.shuffle(self._train_pairs)
        self._train_idx = 0
        logger.info(f"FewShotExampleBuilder: {len(self._train_pairs)} training pairs, strategy={negative_strategy}, format={answer_format}")


    def _next_train_pair(self) -> tuple[str, str]:
        """
        Returns the next training pair, cycling through the shuffled list.
        """
        pair = self._train_pairs[self._train_idx % len(self._train_pairs)]
        self._train_idx += 1
        return pair


    def _is_in_m_ask(self, src_uri: str, tgt_uri: str) -> bool:
        return frozenset({src_uri, tgt_uri}) in self.m_ask_uri_pairs


    def _resolve_entities(self, src_uri: str, tgt_uri: str):
        """
        Resolves two URIs to entity objects. Returns (src, tgt) or None.
        """
        try:
            src = ClassEntity(src_uri, self.OA_source)
            tgt = ClassEntity(tgt_uri, self.OA_target)
            return (src, tgt)
        except (ClassNotFoundError, PropertyNotFoundError, Exception):
            # TODO: reconsider the appropriateness of handling this exception in this manner...
            # NOTE: `_resolve_entity`, `resolve_entity` and `_resolve_entities` are all different fns
            # acorss the codebase (or are from different feature branches), the naming convention 
            # could be confusing and should be reconsidered (TODO).
            return None


    def _format_example(self, src_entity, tgt_entity, label: bool) -> tuple[str, str]:
        """
        Format an example using the prompt function.
        """
        prompt_text = self.prompt_function(src_entity, tgt_entity)
        answer = self._pos_answer if label else self._neg_answer
        return (prompt_text, answer)


    def _make_positive_example(self) -> tuple[str, str] | None:
        for _ in range(MAX_SAMPLE_RETRIES):
            src_uri, tgt_uri = self._next_train_pair()
            if self._is_in_m_ask(src_uri, tgt_uri):
                continue
            entities = self._resolve_entities(src_uri, tgt_uri)
            if entities is not None:
                return self._format_example(entities[0], entities[1], label=True)
        return None


    def _make_hard_negative(self) -> tuple[str, str] | None:
        if self.sibling_selector is None:
            return self._make_random_negative()

        for _ in range(MAX_SAMPLE_RETRIES):
            try:
                src_uri, tgt_uri = self._next_train_pair()
                tgt_entities = self._resolve_entities(src_uri, tgt_uri)
                if tgt_entities is None:
                    continue

                tgt_entity = tgt_entities[1]
                siblings = self.sibling_selector.select_siblings(tgt_entity, max_count=5) # max_count hardcoded (?) NOTE TODO

                for sib_name, _ in siblings:
                    sib_class = self.OA_target.getClassByName(sib_name)
                    if sib_class is None:
                        continue
                    sib_uri = sib_class.iri
                    if sib_uri == tgt_uri or self._is_in_m_ask(src_uri, sib_uri):
                        continue
                    neg_entities = self._resolve_entities(src_uri, sib_uri)
                    if neg_entities is not None:
                        return self._format_example(
                            neg_entities[0], neg_entities[1], label=False
                        )
            
            except Exception as e:
                logger.debug(f"Hard negative generation failed: {e}")
                continue

        logger.info("Hard negative exhausted; falling back to random")
        return self._make_random_negative()


    def _make_random_negative(self) -> tuple[str, str] | None:
        for _ in range(MAX_SAMPLE_RETRIES):
            src_uri_1, _ = self._next_train_pair()
            _, tgt_uri_2 = self._next_train_pair()
            if self._is_in_m_ask(src_uri_1, tgt_uri_2):
                continue
            entities = self._resolve_entities(src_uri_1, tgt_uri_2)
            if entities is not None:
                return self._format_example(entities[0], entities[1], label=False)
        return None


    def _make_negative_example(self) -> tuple[str, str] | None:
        if self.negative_strategy == "hard":
            return self._make_hard_negative()
        return self._make_random_negative()


    def build_examples(
        self, k: int, bidirectional: bool = False
    ) -> list[tuple[str, str]]:
        """Build k few-shot examples.

        Returns list of (user_prompt, assistant_response) tuples.
        """
        if k <= 0:
            return []

        examples = []

        # the cognitive complexity of this code is too much
        # TODO: rework (make more intuitive)

        if not bidirectional:
            # HAPPY PATH: DIRECTIONAL FEW SHOT
            for i in range(k):
                if i % 2 == 0:
                    ex = self._make_positive_example()
                else:
                    ex = self._make_negative_example()
                if ex is not None:
                    examples.append(ex)
        else:
            # (BIDIRECTIONAL PATH)
            # we provide interleving examples 
            # a single example is technically two examples
            # in the bidirecitonal path ...
            n_unique = math.ceil(k / 2)
            for i in range(n_unique):
                is_positive = (i % 2 == 0)
                if is_positive:
                    ex = self._make_positive_example()
                else:
                    ex = self._make_negative_example()
                if ex is not None:
                    examples.append(ex)
                    if len(examples) < k:
                        # add reversed version
                        examples.append((ex[0], ex[1]))

        return examples[:k]



def build_few_shot_examples(
    train_path: str,
    OA_source,
    OA_target,
    prompt_function,
    m_ask_df,
    k: int,
    bidirectional: bool = False,
    negative_strategy: str = "hard",
    sibling_selector=None,
    seed: int = 42,
    answer_format: str = "true_false",
) -> list[tuple[str, str]]:
    """
    Convenience function to build few-shot examples in one call.

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
        ``'hard'`` or ``'random'``.
    
    sibling_selector : SiblingSelector or None
        Required for hard negatives.
    
    seed : int
        Random seed.
    
    answer_format : str
        'true_false' or 'yes_no' — controls the JSON answer strings
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
