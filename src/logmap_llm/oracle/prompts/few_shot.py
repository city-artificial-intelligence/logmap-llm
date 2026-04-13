"""
logmap_llm.oracle.prompts.few_shot

Few-shot example prompt generation: builds few-shot examples in the form:

    (user_prompt, assistant_response) <-- these are pairs (tuples)

^ pairs are used for prompt injection between the developer prompt and M_ask query

These are formatted according to the same template function as the M_ask query
where positive examples are either sampled from train.tsv (or LogMap anchors); 
negatives are obtained via sibling replacement (hard) or a column-swap (random)
(however, note that the class itself doesnt concern itself with sampling from
LogMap anchors, it simply accepts a 'train_tsv', which, in our case, can either
be a legitimate training set -- eg. from bioml, or a constructed set of anchors)
     -> handled by stage_two (build prompts) logic ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
from __future__ import annotations

import math
import random
import logging

import pandas as pd

from logmap_llm.ontology.access import OntologyAccess

from logmap_llm.ontology.object import (
    ClassEntity,
    ClassNotFoundError,
    PropertyEntity,
    PropertyNotFoundError,
    InstanceNotFoundError,
)

from logmap_llm.constants import (
    PAIRS_SEPARATOR,
    DEFAULT_TOP_K,
    DEFAULT_MAX_SAMPLE_RETRIES,
    VERBOSE,
    VERY_VERBOSE,
)

from logmap_llm.utils.logging import (
    info,
    warn,
    warning,
    critical,
    debug,
)

from logmap_llm.ontology.sibling_retrieval import SiblingSelector  # type: ignore

logger = logging.getLogger(__name__)

# TODO: probably resolve ANSWERS using a helper `_resolve_answer` that reuses some
# exitsing implementation elsewhere, for now this is fine; but ensure it matches/updates
# with changes to constants.py 

ANSWERS = {
    ("true_false", "structured"): ('{"answer": true}',  '{"answer": false}'),
    ("true_false", "plain"):      ('True',              'False'),
    ("yes_no",     "structured"): ('{"answer": "Yes"}', '{"answer": "No"}'),
    ("yes_no",     "plain"):      ('Yes',               'No'),
}

class FewShotExampleBuilder:
    """
    Builds few-shot examples for LLM oracle prompting
    
    train_path : fp to train.tsv (eg. for BioML)
    prompt_function : bound template fn \w signature (src_entity, tgt_entity) -> str
    m_ask_uri_pairs : exclusion set from M_ask (prevents leakage)
    negative_strategy: "hard" (replaces \w sibling) or "random" (rand column swap)
    sibling_selector : required if negative_strategy="hard"
    answer and response (_format): ensures the correct formatting of prompts
    """

    def __init__(
        self,
        train_path: str,
        OA_source: OntologyAccess,
        OA_target: OntologyAccess,
        prompt_function: callable,
        m_ask_uri_pairs: set,
        negative_strategy: str = "hard",
        sibling_selector: SiblingSelector = None,
        seed: int = 42,
        answer_format: str = "true_false",
        response_mode: str = "structured",
    ):
        self.OA_source = OA_source
        self.OA_target = OA_target
        self.prompt_function = prompt_function
        self.m_ask_uri_pairs = m_ask_uri_pairs
        self.negative_strategy = negative_strategy
        self.sibling_selector = sibling_selector
        self.answer_format = answer_format
        self.response_mode = response_mode

        try:
            self._pos_answer, self._neg_answer = ANSWERS[(answer_format, response_mode)]
        except KeyError:
            raise ValueError(f"Unsupported (answer_format, response_mode) combination: ({answer_format!r}, {response_mode!r}).")

        self._rng = random.Random(seed)

        # load training pairs (if available)

        self._train_df = pd.read_csv(
            train_path, sep='\t', header=None, usecols=[0, 1]
        )

        self._train_pairs = list(
            zip(self._train_df.iloc[:, 0], self._train_df.iloc[:, 1])
        )

        self._rng.shuffle(self._train_pairs)
        self._train_idx = 0

        info(f"FewShotExampleBuilder: {len(self._train_pairs)} training pairs.")
        info(f"Strategy={negative_strategy}, format={answer_format}, mode={response_mode}.")

        self._hn_succeeded = 0
        self._hn_exhausted_to_random = 0
        self._hn_unsupported_entity = 0



    def _next_train_pair(self) -> tuple[str, str]:
        """Return the next training pair, cycling through the shuffled list."""
        pair = self._train_pairs[self._train_idx % len(self._train_pairs)]
        self._train_idx += 1
        return pair



    def _is_in_m_ask(self, src_uri: str, tgt_uri: str) -> bool:
        return frozenset({src_uri, tgt_uri}) in self.m_ask_uri_pairs



    def _resolve_entities(self, src_uri: str, tgt_uri: str):
        """Resolve two URIs to entity objects. Returns (src, tgt) or None."""
        try:
            src = ClassEntity(src_uri, self.OA_source)
            tgt = ClassEntity(tgt_uri, self.OA_target)
            return (src, tgt)
        except (ClassNotFoundError, PropertyNotFoundError, InstanceNotFoundError, Exception):
            warn(f"Failed to resolve entities: {src_uri}{PAIRS_SEPARATOR}{tgt_uri}.")
            return None



    def _format_example(self, src_entity, tgt_entity, label: bool) -> tuple[str, str]:
        """Format an example using the prompt function."""
        prompt_text = self.prompt_function(src_entity, tgt_entity)
        answer = self._pos_answer if label else self._neg_answer
        return (prompt_text, answer)



    def _make_positive_example(self) -> tuple[str, str] | None:
        for _count in range(DEFAULT_MAX_SAMPLE_RETRIES):
            src_uri, tgt_uri = self._next_train_pair()
            if self._is_in_m_ask(src_uri, tgt_uri):
                if VERBOSE:
                    debug("Retry triggered while making positive example.")
                continue
            entities = self._resolve_entities(src_uri, tgt_uri)
            if entities is not None:
                return self._format_example(entities[0], entities[1], label=True)
        return None



    def _make_hard_negative(self) -> tuple[str, str] | None:
        if self.sibling_selector is None:
            return self._make_random_negative()

        failure_reasons: list[str] = []

        for _count in range(DEFAULT_MAX_SAMPLE_RETRIES):
            try:
                src_uri, tgt_uri = self._next_train_pair()
                entities = self._resolve_entities(src_uri, tgt_uri)
                if entities is None:
                    failure_reasons.append("entity resolution failed")
                    continue

                tgt_entity = entities[1]

                try:
                    ranked = self.sibling_selector.select_siblings(tgt_entity)
                except NotImplementedError:
                    # tgt is a prop or inst — sibling selection does not yet
                    # implement -- no point retrying \w another pair if the
                    # examples are non-CLS's every retry will cause the same
                    # error (as such, revert to rand negatives)
                    self._hn_unsupported_entity += 1
                    warning("Encountered a non-CLS. Hard negs are CLS-only (for now, see doc-str).")
                    warning("Falling back to random negatives during this call.")
                    return self._make_random_negative()

                if not ranked:
                    failure_reasons.append("empty sibling set")
                    continue

                for sib_class, _score in ranked:
                    sib_uri = sib_class.iri
                    if sib_uri == tgt_uri or self._is_in_m_ask(src_uri, sib_uri):
                        continue

                    neg_entities = self._resolve_entities(src_uri, sib_uri)
                    if neg_entities is not None:
                        self._hn_succeeded += 1
                        return self._format_example(
                            neg_entities[0], neg_entities[1], label=False
                        )

                failure_reasons.append("all ranked candidates blocked")

            except Exception as e:
                failure_reasons.append(f"unexpected: {type(e).__name__}: {e}")
                continue

        # retries exhausted; fall back to random (\w consolidated warnings)
        self._hn_exhausted_to_random += 1

        if VERBOSE:
            from collections import Counter
            reason_counts = Counter(failure_reasons)
            reasons_str = ", ".join(f"{reason}x{n}" for reason, n in reason_counts.most_common())
            debug(f"(_make_hard_negatives) Hard negative exhausted after {DEFAULT_MAX_SAMPLE_RETRIES} retries.")
            debug(f"(_make_hard_negatives) Falling back to random. Reasons: {reasons_str}")

        return self._make_random_negative()



    def _report_hard_negative_audit(self) -> dict:
        """
        Return a snapshot of hard-negative construction outcomes
        intended to be logged once at the end of build_examples
        allows users to see whether the 'hard' strategy worked vs failed
        also enables serialisation for analysis \w few-shot examples
        """
        eligible = self._hn_succeeded + self._hn_exhausted_to_random
        total_calls = eligible + self._hn_unsupported_entity
        return {
            "hard_negatives_total_calls": total_calls,
            "hard_negatives_eligible_calls": eligible,
            "hard_negatives_succeeded": self._hn_succeeded,
            "hard_negatives_exhausted_to_random": self._hn_exhausted_to_random,
            "hard_negatives_unsupported_entity": self._hn_unsupported_entity,
            "hard_negatives_success_rate": (
                self._hn_succeeded / eligible if eligible > 0 else None
            ),
        }



    def _make_random_negative(self) -> tuple[str, str] | None:
        _count = 0 # kinda hacky (only useful when VERBOSE & VERY_VERBOSE)
        for _count in range(DEFAULT_MAX_SAMPLE_RETRIES):
            
            src_uri_1, _tgt = self._next_train_pair()
            _src, tgt_uri_2 = self._next_train_pair()
            
            if self._is_in_m_ask(src_uri_1, tgt_uri_2):
                if VERBOSE and VERY_VERBOSE:
                    debug("(_make_random_negative) Encountered a collision when searching for random negatives.")
                continue
            
            entities = self._resolve_entities(src_uri_1, tgt_uri_2)

            if entities is not None:
                return self._format_example(entities[0], entities[1], label=False)
            
        if VERBOSE and VERY_VERBOSE:
            if _count > 0:
                debug(f"(_make_random_negative) Unable to construct a random negative (after {_count} tries).")
            else:
                debug(f"(_make_random_negative) Unable to construct a random negative, since none exist.")
        
        return None



    def _make_negative_example(self) -> tuple[str, str] | None:
        if self.negative_strategy == "hard":
            return self._make_hard_negative()
        return self._make_random_negative()



    def build_examples(self, k: int, bidirectional: bool = False, contrastive: bool = True) -> list[tuple[str, str]]:
        """
        Build k few-shot examples:
        returns a list of (user_prompt, assistant_response) tuples
        """
        if k <= 0:
            return []

        examples = []

        if not bidirectional and contrastive:
            
            # produce interleving contrastive examples
            
            for i in range(k):
                if i % 2 == 0:
                    ex = self._make_positive_example()
                else:
                    ex = self._make_negative_example()
                if ex is not None:
                    examples.append(ex)
        
                if VERBOSE and VERY_VERBOSE and ex is None:
                    debug("(build_examples) Unable to construct example.")
                    debug("(build_examples)   (NOT bidirectional & CONTRASTIVE).")

        elif not bidirectional and not contrastive: 

            # produce positive examples up to N examples

            for i in range(k):
                ex = self._make_positive_example()
                if ex is not None:
                    examples.append(ex)

                if VERBOSE and VERY_VERBOSE and ex is None:
                    debug("(build_examples) Unable to construct example.")
                    debug("(build_examples)   (NOT bidirectional & NOT contrastive)")

        elif bidirectional and contrastive:

            # produce interleving contrastive examples in both forward and reverse directions

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
                        examples.append((ex[0], ex[1]))

                if VERBOSE and VERY_VERBOSE and ex is None:
                    debug("(build_examples) Unable to construct example.")
                    debug("(build_examples)   (BIDIRECTIONAL & CONTRASTIVE)")

        elif bidirectional and not contrastive:

            # produce positive examples in both directions up to N / 2 examples

            n_unique = math.ceil(k / 2)

            for i in range(n_unique):
                ex = self._make_positive_example()
                if ex is not None:
                    examples.append(ex)
                    if len(examples) < k:
                        examples.append((ex[0], ex[1]))

                if VERBOSE and VERY_VERBOSE and ex is None:
                    debug("(build_examples) Unable to construct example.")
                    debug("(build_examples)   (BIDIRECTIONAL & NOT contrastive)")

        else:
            # some strange combination has been encountered ...
            warning("A serious issue has been encountered while constructing examples (expect unpredictable behaviour).")

        if VERBOSE and VERY_VERBOSE:
            debug(f"(build_examples) The top-k ({len(examples[:k])}) examples have been constructed (out of a total of {len(examples)} examples).")
            debug("(build_examples) PRINTING EXAMPLES:")
            for ex_idx, example in enumerate(examples[:k]):
                debug(f"(build_examples) (example: {ex_idx}) {example[0]} : {example[1]}.")

        # (generally) useful reporting
        if self.negative_strategy == "hard" and self.sibling_selector is not None:
            audit = self._report_hard_negative_audit()
            info(f"(build_examples) HARD-NEGATIVE AUDIT: ")
            info(f"(build_examples)  {audit['hard_negatives_succeeded']} / {audit['hard_negatives_total_calls']} succeeded.")
            info(f"(build_examples)  {audit['hard_negatives_exhausted_to_random']} exhausted retries (to random).")
            info(f"(build_examples)  {audit['hard_negatives_unsupported_entity']} encountered an unsupported entity (see doc-string).")

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
    response_mode: str = "structured",
) -> list[tuple[str, str]]:
    """Convenience function to build few-shot examples in one call"""
    if k <= 0:
        return []
    
    # build direction-agnostic 
    # exclusion set from m_ask
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
        response_mode=response_mode,
    )

    return builder.build_examples(k=k, bidirectional=bidirectional)
