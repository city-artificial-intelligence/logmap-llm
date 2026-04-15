'''
logmap_llm.oracle.prompts.nl_verbaliser — converts a selection of prompts to NLF

Provides English verbalisation for KG instance property-value pairs.

For Example:

Current output:  is "homeworld" "Tatooine" and is "species" "Human"
NLF output:      whose homeworld is "Tatooine" and whose species is "Human"

TODO: integrate into pipeline as a configurable usable module; at present
we just test this manually during KG evaluation.
'''
from __future__ import annotations

import re


def _normalise_predicate_label(label: str) -> str:
    """
    Normalise a predicate label to lowercase space-separated words
    """
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    s = s.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", s).strip().lower()


# TODO: consider moving such tuples to constants.py
#       and allow for larger degree of user configuration

_VERB_PREFIXES = ("has ", "is ", "was ", "can ", "does ", "did ", "are ")

_TEMPORAL_PREDICATES = frozenset({
    "born", "died", "birth date", "death date", "date",
    "founded", "established", "created", "dissolved",
    "start date", "end date", "year",
})

_DATE_PATTERN = re.compile(
    r"^\d{4}(-\d{2}(-\d{2})?)?$|^\d{1,2}/\d{1,2}/\d{2,4}$"
)


def verbalise_property_value(predicate_label: str, value: str) -> str:
    """
    Convert a predicate-value pair into a natural language clause
    """
    norm = _normalise_predicate_label(predicate_label)

    # already verb-phrase-shaped
    for prefix in _VERB_PREFIXES:
        if norm.startswith(prefix):
            return f'which {norm} "{value}"'

    # temporal predicates with date-like values
    if norm in _TEMPORAL_PREDICATES and _DATE_PATTERN.match(value.strip()):
        if norm in ("born", "birth date"):
            return f'who was born on {value}'
        elif norm in ("died", "death date"):
            return f'who died on {value}'
        else:
            return f'whose {norm} is {value}'

    # default: "whose {pred} is {value}"
    return f'whose {norm} is "{value}"'


def format_instance_attribute_clause_nl(
    properties: list[dict],
    max_properties: int = 5,
    conjunction: str = " and ",
) -> str:
    """
    Verbalise a list of instance property dicts into a single NL clause
    """
    clauses = []
    for prop in properties[:max_properties]:
        pred = prop.get("predicate_label", "")
        val = prop.get("value") or prop.get("object_label", "")
        if pred and val:
            clauses.append(verbalise_property_value(pred, val))
    return conjunction.join(clauses)
