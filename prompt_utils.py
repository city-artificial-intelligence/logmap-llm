from __future__ import annotations

from onto_object import OntologyEntryAttr


def get_name_string(name_set: set | list | OntologyEntryAttr) -> str:
    """Get a string representation of the name set."""
    # If the name_set is a set or list, join the elements with a comma
    if isinstance(name_set, (set, list)):
        return ", ".join(sorted(name_set))
    return str(name_set)


def get_single_name(name_set: set | list | str | OntologyEntryAttr) -> str | None:
    """Get a single name from the name set."""
    return next(iter(name_set), None) if isinstance(name_set, (set, list)) else name_set


def select_best_direct_entity_names(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> list[OntologyEntryAttr, OntologyEntryAttr]:
    """If there are multiple direct parents, select one and find child element for it."""
    src_parents = next(iter(src_entity.get_direct_parents()), None)
    tgt_parents = next(iter(tgt_entity.get_direct_parents()), None)
    return [
        get_name_string(x.get_preferred_names()) if x else None
        for x in [src_parents, tgt_parents, src_entity, tgt_entity]
    ]


def format_hierarchy(
    hierarchy_dict: dict[int, set[OntologyEntryAttr]], no_level: bool = False, add_thing: bool = True
) -> list[str] | str:
    formatted = []
    for level, parents in sorted(hierarchy_dict.items()):
        parent_name = get_name_string([get_name_string(i.get_preferred_names()) for i in parents])

        if not add_thing and parent_name == "Thing":
            continue

        if no_level:
            formatted.append(parent_name)
        else:
            formatted.append(f"\tLevel {level}: {parent_name}")

    return formatted if no_level else "\n".join(formatted)


def select_best_direct_entity_names_with_synonyms(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, add_thing: bool = True
) -> list:
    """Select preferred names and synonyms for source and target entities and their direct parents."""

    def get_parent_name(entity: OntologyEntryAttr) -> str | None:
        parent = next(iter(entity.get_direct_parents()), None)
        parent_name = get_name_string(parent.get_preferred_names()) if parent else None
        if parent_name == "Thing" and not add_thing:
            return None
        return parent_name

    def get_clean_synonyms(entity: OntologyEntryAttr) -> list[str]:
        synonyms = list(entity.get_synonyms())
        entity_class_name = str(entity.thing_class.name)
        return [] if len(synonyms) == 1 and synonyms[0] == entity_class_name else synonyms

    src_parent_name = get_parent_name(src_entity)
    tgt_parent_name = get_parent_name(tgt_entity)

    src_entity_name = get_name_string(src_entity.get_preferred_names())
    tgt_entity_name = get_name_string(tgt_entity.get_preferred_names())
    src_synonyms = get_clean_synonyms(src_entity)
    tgt_synonyms = get_clean_synonyms(tgt_entity)

    return [
        src_parent_name,  # string | None
        tgt_parent_name,  # string | None
        src_entity_name,  # string
        tgt_entity_name,  # string
        src_synonyms,  # list of strings
        tgt_synonyms,  # list of strings
    ]


def select_best_sequential_hierarchy_with_synonyms(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, max_level: int
) -> tuple[list[str], list[str], list[list[str]], list[list[str]]]:
    """Select the best synonyms for an entity and its hierarchical parents."""
    src_parents_by_levels = src_entity.get_parents_by_levels(max_level)
    tgt_parents_by_levels = tgt_entity.get_parents_by_levels(max_level)

    def get_synonyms_and_class(parents_by_levels: dict[int, set[OntologyEntryAttr]], idx: int) -> tuple[list[str], str]:
        if len(parents_by_levels) > idx:
            entry = next(iter(parents_by_levels[idx]))
            syns = entry.get_synonyms() if hasattr(entry, "get_synonyms") else []
            cls = str(entry.onto.getClassByURI(entry.annotation["uri"])).split(".")[-1]
            return syns, cls
        return [], ""

    def clean(synonyms: list, cls: str) -> list[str]:
        return [] if len(synonyms) == 1 and next(iter(synonyms)) == cls else synonyms

    src_results = [clean(*get_synonyms_and_class(src_parents_by_levels, i)) for i in range(len(src_parents_by_levels))]
    tgt_results = [clean(*get_synonyms_and_class(tgt_parents_by_levels, i)) for i in range(len(tgt_parents_by_levels))]

    return src_results[0], tgt_results[0], src_results[1:], tgt_results[1:]


def format_sibling_context(
    sibling_labels: list[str] | list[tuple[str, float]],
    parent_name: str,
) -> str:
    """format sibling labels into a natural-language context clause"""
    if not sibling_labels:
        return ""

    # normalise: accept either plain strings or (label, score) tuples
    labels = []
    for item in sibling_labels:
        if isinstance(item, tuple):
            labels.append(item[0])
        else:
            labels.append(item)

    if len(labels) == 1:
        sibling_text = f'"{labels[0]}"'
    else:
        sibling_text = ", ".join(f'"{s}"' for s in labels[:-1])
        sibling_text += f' and "{labels[-1]}"'

    return f'Other "{parent_name}" concepts include {sibling_text}.'


# property prompt formatting helpers

def format_synonyms_parenthetical(synonyms: set[str], preferred_name: str) -> str:
    # remove the preferred name itself from synonyms
    extra_syns = sorted(s for s in synonyms if s != preferred_name)
    if not extra_syns:
        return ""
    quoted = ", ".join(f'"{s}"' for s in extra_syns)
    return f" (also known as {quoted})"


def format_domain_range_clause(
    prop_label: str,
    domain_names: set[str],
    range_names: set[str],
    domain_synonyms: set[str] | None = None,
    range_synonyms: set[str] | None = None,
    include_synonyms: bool = False,
) -> str:
    """format a property with optional domain/range context"""
    
    domain_str = ", ".join(f'"{n}"' for n in sorted(domain_names)) if domain_names else None
    range_str = ", ".join(f'"{n}"' for n in sorted(range_names)) if range_names else None

    parts = [f'"{prop_label}"']

    if include_synonyms and domain_synonyms is None:
        domain_synonyms = set()
    if include_synonyms and range_synonyms is None:
        range_synonyms = set()

    if domain_str and range_str:
        domain_syn_str = ""
        range_syn_str = ""
        if include_synonyms and domain_synonyms:
            # remove names already used as domain labels
            extra = sorted(s for s in domain_synonyms if s not in domain_names)
            if extra:
                domain_syn_str = " (also known as " + ", ".join(f'"{s}"' for s in extra) + ")"
        if include_synonyms and range_synonyms:
            extra = sorted(s for s in range_synonyms if s not in range_names)
            if extra:
                range_syn_str = " (also known as " + ", ".join(f'"{s}"' for s in extra) + ")"

        parts.append(f", connecting elements of type {domain_str}{domain_syn_str}"
                     f" to elements of type {range_str}{range_syn_str}")
    elif domain_str:
        parts.append(f", used by elements of type {domain_str}")
    elif range_str:
        parts.append(f", targeting elements of type {range_str}")

    return "".join(parts)


# Instance prompt formatting helpers (KG track):

def format_instance_type_clause(type_names: list[str]) -> str:
    """Format rdf:type names into a natural-language clause"""
    if not type_names:
        return ""
    if len(type_names) == 1:
        return f'which is a "{type_names[0]}"'
    quoted = [f'a "{t}"' for t in type_names]
    return "which is " + " and ".join(quoted)


def _format_property_value(value: str, datatype: str) -> str:
    """Format a property value with type-appropriate quoting"""
    numeric_types = {"integer", "int", "float", "double", "decimal", "long", "short", 
                     "nonNegativeInteger", "positiveInteger", "negativeInteger"}
    if datatype in numeric_types:
        return value
    return f'"{value}"'


def format_instance_attribute_clause(
    properties: list[dict],
    max_properties: int = 5,
) -> str:
    """
    format instance property-value pairs as natural-language clauses
    handles both data properties (literal values) and object properties (URI values with labels) in a unified way

    Parameters
    ----------
    properties : list[dict]
        each dict should have at minimum 'predicate_label' and either 'value'+'datatype' (data property) 
        or 'object_label' (object property)
    max_properties : int
        maximum number of property clauses to include

    Returns
    -------
    str
        clauses joined by " and ", e.g., 'is "homeworld" "Tatooine" and is "species" "Human"'
        Returns empty string if no properties.
    """
    if not properties:
        return ""

    clauses = []
    for prop in properties[:max_properties]:
        pred_label = prop["predicate_label"]
        if "value" in prop:
            # data property
            formatted_val = _format_property_value(prop["value"], prop.get("datatype", "string"))
            clauses.append(f'"{pred_label}" {formatted_val}')
        elif "object_label" in prop:
            # object property
            clauses.append(f'"{pred_label}" "{prop["object_label"]}"')

    if not clauses:
        return ""

    return " and ".join(f"is {c}" for c in clauses)


def _get_property_value(prop: dict) -> str:
    """Extract a comparable value string from a property dict"""
    if "value" in prop:
        return str(prop["value"]).lower().strip()
    if "object_label" in prop:
        return str(prop["object_label"]).lower().strip()
    return ""


def _find_best_pair(src_entries: list[dict], tgt_entries: list[dict]) -> tuple[dict, dict]:
    """
    for a shared predicate with multiple values per side, find the best pair.

    If any source value matches any target value, return that matching pair
    (strongest identity evidence for the LLM).  If no values match,
    fall back to the first entry from each side.

    Parameters
    ----------
    src_entries : list[dict]
        All property dicts from the source instance for this predicate label.
    tgt_entries : list[dict]
        All property dicts from the target instance for this predicate label.

    Returns
    -------
    tuple of (best_src, best_tgt)
        The property dict pair to include in the prompt.
    """
    # build a value -> entry index for fast matching
    tgt_by_value = {}
    for t in tgt_entries:
        val = _get_property_value(t)
        if val and val not in tgt_by_value:
            tgt_by_value[val] = t

    # look for a value match
    for s in src_entries:
        val = _get_property_value(s)
        if val and val in tgt_by_value:
            return s, tgt_by_value[val]

    # no match found — fall back to first entries
    return src_entries[0], tgt_entries[0]


def select_intersecting_properties(
    src_props: list[dict],
    tgt_props: list[dict],
    max_properties: int = 5,
    intersection_only: bool = False,
    predicate_entropies: dict | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    select properties for prompt inclusion, prioritising shared predicates; approach:
    1. group all property entries by predicate label
    2. find predicate labels shared across both instances
    3. for shared predicates with multiple values, prefer value-matching pairs (where both entities have eq values)
    4. rank shared predicates by entropy (if provided); alternatively: alphabetically
    5. include top-ranked shared properties (up to max_properties)
    6. unless intersection_only, fill remaining slots with non-intersecting properties

    Parameters
    ----------
    src_props : list[dict]
        All properties from source instance (data + object)
    tgt_props : list[dict]
        All properties from target instance (data + object)
    max_properties : int
        maximum properties to return per side
    intersection_only : bool
        If True, return only properties with matching predicate labels across both instances
    predicate_entropies : dict, optional
        Mapping of predicate_uri -> entropy (float).  When provided, shared predicates are ranked by descending entropy 

    Returns
    -------
    tuple of (src_selected, tgt_selected)
    """
    from collections import defaultdict

    # group all entries by normalised label
    src_all_by_label = defaultdict(list)
    for p in src_props:
        label = p["predicate_label"].lower().strip()
        src_all_by_label[label].append(p)

    tgt_all_by_label = defaultdict(list)
    for p in tgt_props:
        label = p["predicate_label"].lower().strip()
        tgt_all_by_label[label].append(p)

    # find intersecting predicate labels
    shared_labels = set(src_all_by_label.keys()) & set(tgt_all_by_label.keys())

    # for each shared label, find the best (src, tgt) pair 
    # prefer value matches when the predicate has multiple values
    shared_best_pairs = {}
    for label in shared_labels:
        best_src, best_tgt = _find_best_pair(
            src_all_by_label[label], tgt_all_by_label[label]
        )
        shared_best_pairs[label] = (best_src, best_tgt)

    # rank shared properties (by entropy desc, or alphabetically)
    if predicate_entropies is not None and shared_labels:
        def _entropy_for_label(label):
            src_entry, tgt_entry = shared_best_pairs[label]
            src_uri = src_entry.get("predicate_uri", "")
            tgt_uri = tgt_entry.get("predicate_uri", "")
            return max(
                predicate_entropies.get(src_uri, 0.0),
                predicate_entropies.get(tgt_uri, 0.0),
            )
        ranked_shared = sorted(shared_labels, key=_entropy_for_label, reverse=True)
    else:
        ranked_shared = sorted(shared_labels)

    src_selected = []
    tgt_selected = []
    selected_src_labels = set()
    selected_tgt_labels = set()

    # add shared properties in ranked order using best pairs
    for label in ranked_shared:
        if len(src_selected) >= max_properties:
            break
        best_src, best_tgt = shared_best_pairs[label]
        src_selected.append(best_src)
        tgt_selected.append(best_tgt)
        selected_src_labels.add(label)
        selected_tgt_labels.add(label)

    # fill remaining slots with non-shared properties (deduplicated by label)
    if not intersection_only:
        if len(src_selected) < max_properties:
            for label in sorted(set(src_all_by_label.keys()) - shared_labels):
                if label in selected_src_labels:
                    continue
                src_selected.append(src_all_by_label[label][0])
                selected_src_labels.add(label)
                if len(src_selected) >= max_properties:
                    break

        if len(tgt_selected) < max_properties:
            for label in sorted(set(tgt_all_by_label.keys()) - shared_labels):
                if label in selected_tgt_labels:
                    continue
                tgt_selected.append(tgt_all_by_label[label][0])
                selected_tgt_labels.add(label)
                if len(tgt_selected) >= max_properties:
                    break

    return src_selected, tgt_selected





import re

# predicate label normalisation
def _normalise_predicate_label(label: str) -> str:
    """
    Normalise a predicate label to lowercase space-separated words. i.e., handles:
      - camelCase  -> "camel case"
      - snake_case -> "snake case"
      - kebab-case -> "kebab case"
      - Already spaced labels -> pass through
    """
    # split camelCase
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", label)
    # replace underscores and hyphens with spaces
    s = s.replace("_", " ").replace("-", " ")
    # collapse multiple spaces and lowercase
    return re.sub(r"\s+", " ", s).strip().lower()


# ------------------------------------------
# Verb-phrase classification and templates
# ------------------------------------------

_VERB_PREFIXES = ("has ", "is ", "was ", "can ", "does ", "did ", "are ")

_TEMPORAL_PREDICATES = {
    "born", "died", "created", "founded", "established",
    "dissolved", "launched", "destroyed", "built", "published",
    "released", "formed", "introduced", "retired", "decommissioned",
}

_PLURAL_PREDICATES = {
    "masters", "apprentices", "children", "parents", "siblings",
    "allies", "enemies", "members", "weapons", "powers",
    "affiliations", "locations", "appearances", "abilities",
    "languages", "titles", "nicknames", "aliases",
}

# TODO: extend lists (or, alternatively use something like a domain-tuned natural language fluency model -- DART?)
# ^^ probably overkill for this.

def _verbalize_single_property(pred_label: str, formatted_value: str) -> str:
    """
    convert a single predicate-value pair into a natural English clause
    """
    norm = _normalise_predicate_label(pred_label)

    if not norm:
        return f"has value {formatted_value}"

    for prefix in _VERB_PREFIXES:
        if norm.startswith(prefix):
            return f"{norm} {formatted_value}"

    if norm in _TEMPORAL_PREDICATES:
        return f"was {norm} {formatted_value}"

    if norm in _PLURAL_PREDICATES:
        return f"whose {norm} include {formatted_value}"

    return f"whose {norm} is {formatted_value}"

# --------------------------------------------------
# Public formatting function (drop-in replacement)
# --------------------------------------------------

def format_instance_attribute_clause_nl(
    properties: list[dict],
    max_properties: int = 5,
) -> str:
    """
    Format instance property-value pairs as natural-language clauses.
    """
    if not properties:
        return ""

    clauses = []
    for prop in properties[:max_properties]:
        pred_label = prop["predicate_label"]

        if "value" in prop:
            formatted_val = _format_property_value(
                prop["value"], prop.get("datatype", "string")
            )
        elif "object_label" in prop:
            formatted_val = f'"{prop["object_label"]}"'
        else:
            continue

        clause = _verbalize_single_property(pred_label, formatted_val)
        clauses.append(clause)

    if not clauses:
        return ""

    # oxford-comma style joining for readability
    if len(clauses) == 1:
        return clauses[0]
    elif len(clauses) == 2:
        return f"{clauses[0]} and {clauses[1]}"
    else:
        return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
