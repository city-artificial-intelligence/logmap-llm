from onto_object import OntologyEntryAttr
from prompt_utils import (
    format_hierarchy,
    format_sibling_context,
    get_single_name,
    select_best_direct_entity_names,
    select_best_direct_entity_names_with_synonyms,
    select_best_sequential_hierarchy_with_synonyms,
)
from prompt_utils import (
    get_single_name,
    format_domain_range_clause,
    format_synonyms_parenthetical,
)
from constants import RESPONSE_INSTRUCTION, DEFAULT_ANSWER_FORMAT, ANSWER_FORMATS
from typing import Callable

# ----------------------------------
# Module-level answer format state
# ----------------------------------

_response_instruction: str = RESPONSE_INSTRUCTION[DEFAULT_ANSWER_FORMAT]


def set_answer_format(fmt: str) -> None:
    """Set the response instruction wording used by all prompt templates.
    param: 
        'true_false' or 'yes_no'
    """
    global _response_instruction
    if fmt not in ANSWER_FORMATS:
        raise ValueError(
            f"Unknown answer_format '{fmt}'. "
            f"Valid options: {sorted(ANSWER_FORMATS)}"
        )
    _response_instruction = RESPONSE_INSTRUCTION[fmt]


def get_response_instruction() -> str:
    """Return the current response instruction string"""
    return _response_instruction


# -----------------------------------
# Module-level ontology domain state
# -----------------------------------

_ontology_domain: str = "biomedical"

TRACK_TO_DOMAIN = {
    "bioml": "biomedical",
    "anatomy": "biomedical",
    "conference": "conference",
    "knowledgegraph": "knowledge graph",
}


def set_ontology_domain(track: str | None) -> None:
    """
    set the ontology domain qualifier used in prompt preambles
    """
    global _ontology_domain
    _ontology_domain = TRACK_TO_DOMAIN.get(track, "biomedical")     # type: ignore


def get_ontology_domain() -> str:
    """Return the current ontology domain qualifier string."""
    return _ontology_domain


# -------------------------------
#        Old Prompts Section
# -------------------------------


def prompt_all_data_dummy(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    **Task Description:**
    Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

    1. **Source Entity:**
    **All Entity names:** {src_entety.get_preferred_names()}
    **Parent Entity Namings:** {src_entety.get_parents_preferred_names()}
    **Child Entity Namings:** {src_entety.get_children_preferred_names()}

    2. **Target Entity:**
    **All Entity names:** {tgt_entety.get_preferred_names()}
    **Parent Entity Namings:** {tgt_entety.get_parents_preferred_names()}
    **Child Entity Namings:** {tgt_entety.get_children_preferred_names()}

    Write "Yes" if the entities refer to the same concepts, and "No" otherwise.
    """.strip()


def prompt_only_names(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}

    {_response_instruction}
    """.strip()


def prompt_with_hierarchy(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}
    Parent Entity Namings: {src_entety.get_parents_preferred_names()}
    Child Entity Namings: {src_entety.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}
    Parent Entity Namings: {tgt_entety.get_parents_preferred_names()}
    Child Entity Namings: {tgt_entety.get_children_preferred_names()}

    {_response_instruction}
    """.strip()


def prompt_only_with_parents(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names and parent relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}
    Parent Entity Namings: {src_entety.get_parents_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}
    Parent Entity Namings: {tgt_entety.get_parents_preferred_names()}

    {_response_instruction}
    """.strip()


def prompt_only_with_children(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entety.get_all_entity_names()}
    Child Entity Namings: {src_entety.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entety.get_all_entity_names()}
    Child Entity Namings: {tgt_entety.get_children_preferred_names()}

    {_response_instruction}
    """.strip()


# -------------------------------
#        New Prompts Section
# -------------------------------

#
# DH: Notes on the prompt function names in this section ...
# 1) The prompt functions have been renamed using a new naming convention
# 2) Acronym 'oupt' stands for 'oracle user prompt template'. We use this
#    as a prefix to shorten the names of the prompt functions.
# 3) Every prompt template includes the source and target entities 
#    involved in a candidate mapping; we do not refer to them in the
#    function names
#

# 
# Structured prompts
# 


#def prompt_direct_entity_ontological()
def oupt_one_level_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Ontological prompt that uses ontology-focused language."""
    src_parent, tgt_parent, src_entity_names, tgt_entity_names = select_best_direct_entity_names(src_entity, tgt_entity)

    prompt_lines = [
        f"Analyze the following entities, each originating from a distinct {_ontology_domain} ontology.",
        "Your task is to assess whether they represent the **same ontological concept**, considering both their semantic meaning and hierarchical position.",
        f'\n1. Source entity: "{src_entity_names}"',
        f"\t- Direct ontological parent: {src_parent}",
        f'\n2. Target entity: "{tgt_entity_names}"',
        f"\t- Direct ontological parent: {tgt_parent}",
        f'\nAre these entities **ontologically equivalent** within their respective ontologies? {_response_instruction}',
    ]

    return "\n".join(prompt_lines)

#def prompt_sequential_hierarchy_ontological()
def oupt_two_levels_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Ontological prompt that uses ontology-focused language, and takes hierarchical relationships into account."""
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2))
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2))

    prompt_lines = [
        f"Analyze the following entities, each originating from a distinct {_ontology_domain} ontology.",
        "Each is represented by its **ontological lineage**, capturing its hierarchical placement from the most general to the most specific level.",
        f"\n1. Source entity ontological lineage:\n{src_hierarchy}",
        f"\n2. Target entity ontological lineage:\n{tgt_hierarchy}",
        f'\nBased on their **ontological positioning, hierarchical relationships, and semantic alignment**, do these entities represent the **same ontological concept**? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)

#
# Natural language friendly prompts
#

#def prompt_direct_entity()
def oupt_one_level_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Regular prompt that uses natural language and is more intuitive."""
    src_parent, tgt_parent, src_entity_names, tgt_entity_names = select_best_direct_entity_names(src_entity, tgt_entity)
    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        (
            f'The first one is "{src_entity_names}"'
            + (f', which belongs to the broader category "{src_parent}"' if src_parent else "")
        ),
        (
            f'The second one is "{tgt_entity_names}"'
            + (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else "")
        ),
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)


#def prompt_sequential_hierarchy()
def oupt_two_levels_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Regular prompt that uses natural language and is more intuitive."""
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        (
            f'The first one is "{src_hierarchy[0]}"'
            + (f', which belongs to the broader category "{src_hierarchy[1]}"' if len(src_hierarchy) > 1 else "")
            + (f', under the even broader category "{src_hierarchy[2]}"' if len(src_hierarchy) > 2 else "")
        ),
        (
            f'The second one is "{tgt_hierarchy[0]}"'
            + (f', which belongs to the broader category "{tgt_hierarchy[1]}"' if len(tgt_hierarchy) > 1 else "")
            + (f', under the even broader category "{tgt_hierarchy[2]}"' if len(tgt_hierarchy) > 2 else "")
        ),
        f'\nDo they mean the same thing? {_response_instruction}',
    ]

    return "\n".join(prompt_lines)

#
# Natural language prompts with synonyms
#

#def prompt_direct_entity_with_synonyms()
def oupt_one_level_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Natural language prompt that includes synonyms for a more intuitive comparison."""
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""

    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)

#def prompt_sequential_hierarchy_with_synonyms()
def oupt_two_levels_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """Generate a natural language prompt asking whether two ontology entities (with synonyms and hierarchy).

    Represent the same concept (True/False).
    """
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)

    src_syns, tgt_syns, src_parents_syns, tgt_parents_syns = select_best_sequential_hierarchy_with_synonyms(
        src_entity, tgt_entity, max_level=2
    )

    def describe_entity(hierarchy: list[str], entity_syns: list[str], parent_syns: list[list[str]]) -> str:
        # Base name and its synonyms
        name_part = f'"{hierarchy[0]}"'
        if entity_syns:
            alt = ", ".join(f'"{s}"' for s in entity_syns)
            name_part += f", also known as {alt}"

        parts = [name_part]

        labels = ["belongs to broader category", "under the even broader category", "under the even broader category"]
        for i, parent_name in enumerate(hierarchy[1:]):
            text = f'{labels[i]} "{parent_name}"'
            if parent_syns[i]:
                alt = ", ".join(f'"{s}"' for s in parent_syns[i])
                text += f" (also known as {alt})"
            parts.append(text)

        return ", ".join(parts)

    src_desc = describe_entity(src_hierarchy, src_syns, src_parents_syns)   # type: ignore
    tgt_desc = describe_entity(tgt_hierarchy, tgt_syns, tgt_parents_syns)   # type: ignore

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        f"The first one is {src_desc}.",
        f"The second one is {tgt_desc}.",
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)


# -------------------------------
#   Subsumption Prompts Section
# -------------------------------

# requires 2 prompts
# TODO: rename, name is misleading, it also uses synonyms!!
def oupt_sub_labels_only(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    subsumption prompt using only concept labels and synonyms
    """
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"?'
        f' {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



# requires 2 prompts
def oupt_sub_parents_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    # Build entity description lines
    src_line = f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".'
    tgt_line = f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".'

    # Build subsumption question with optional parent disambiguation
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        src_line,
        tgt_line,
        f'\nIf something is a "{src_entity_names}"{src_qual}, is it also'
        f' a "{tgt_entity_names}"{tgt_qual}?'
        f' {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



# requires single prompt
def oupt_equiv_with_ancestral_disjointness(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:

    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    src_line = f'The first one is "{src_entity_names}"{src_synonyms_text}.'
    tgt_line = f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}".'

    # obtain ancestral context
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        src_line,
        tgt_line,
        f'\nIf something is a "{src_entity_names}"{src_qual}, is it also a "{tgt_entity_names}"{tgt_qual}?',
        f'\n{_response_instruction}',
    ]
    return "\n".join(prompt_lines)


# requires single prompt
def oupt_deductive_equiv(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    premise-based deductive equivalence prompt
    """
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    # build fact lines as bullet points (premise-like style)
    facts = []
    facts.append(f'- "{src_entity_names}" is a kind of "{src_parent}" (Ontology 1)' if src_parent
                 else f'- "{src_entity_names}" is a concept in Ontology 1')
    if src_synonyms:
        syn_text = " and ".join(f'"{s}"' for s in src_synonyms)
        facts.append(f'- "{src_entity_names}" is also known as {syn_text} (Ontology 1)')

    facts.append(f'- "{tgt_entity_names}" is a kind of "{tgt_parent}" (Ontology 2)' if tgt_parent
                 else f'- "{tgt_entity_names}" is a concept in Ontology 2')
    if tgt_synonyms:
        syn_text = " and ".join(f'"{s}"' for s in tgt_synonyms)
        facts.append(f'- "{tgt_entity_names}" is also known as {syn_text} (Ontology 2)')

    facts_text = "\n".join(facts)

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies. Consider the following facts:",
        facts_text,
        f'\nGiven these facts, can you conclude that "{src_entity_names}" (Ontology 1)'
        f' and "{tgt_entity_names}" (Ontology 2) refer to the same real-world concept?'
        f' {_response_instruction}',
    ]
    return "\n".join(prompt_lines)


def _retrieve_siblings(entity, sibling_selector, max_count=2):
    """
    retrieve sibling labels for an entity.

    Uses SapBERT-ranked selection if a sibling_selector is provided
    else fallback to alphabetical get_siblings()

    Returns a list of (label, score) tuples.
    """
    if sibling_selector is not None:
        return sibling_selector.select_siblings(entity, max_count=max_count)
    else:
        sib_entries = entity.get_siblings(max_count=max_count)
        return [
            (min(s.get_preferred_names()) if s.get_preferred_names()
             else str(s.thing_class.name), 1.0)
            for s in sib_entries
        ]


def oupt_equiv_parents_siblings(
    src_entity: OntologyEntryAttr,
    tgt_entity: OntologyEntryAttr,
    sibling_selector=None,
) -> str:
    """
    equivalence prompt with parent categories and SapBERT-ranked siblings
    """
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, _, _ = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)

    # if src_parent and tgt_parent:
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}", which falls under the category "{src_parent}".{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}", which falls under the category "{tgt_parent}".{tgt_sibling_clause}',
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



def oupt_equiv_parents_synonyms_siblings(
    src_entity: OntologyEntryAttr,
    tgt_entity: OntologyEntryAttr,
    sibling_selector=None,
) -> str:
    """
    equivalence prompt with parent categories, synonyms, and SapBERT-ranked siblings
    """
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)

    # if src_parent and tgt_parent:
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""

    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".{tgt_sibling_clause}',
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



def oupt_synonyms_only(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    natural language prompt that includes synonyms for a more intuitive comparison
    """
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
        select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    )

    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    prompt_lines = [
        "We have two entities from different ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



# -----------------------------------------------
#  property equiv prompt templates (conf track)
# ---------------------------------------------------------------------------------------------
# these templates accept PropertyEntryAttr objects (from onto_object.py) rather than OntologyEntryAttr
# They are registered separately in oracle_prompt_building.py under PROPERTY_TEMPLATES.
# ---------------------------------------------------------------------------------------------


def oupt_prop_labels_only(src_prop, tgt_prop) -> str:
    """
    P1 — property equivalence with labels only (no context) -- baseline
    """
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)

    return (
        f'We have two properties from different ontologies.\n\n'
        f'The first property is "{src_name}".\n\n'
        f'The second property is "{tgt_name}".\n\n'
        f'Do these properties represent the same relationship?\n\n'
        f'{_response_instruction}'
    )


def oupt_prop_domain_range(src_prop, tgt_prop) -> str:
    """
    P2 — property equivalence with domain and range context
    """
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)

    src_desc = format_domain_range_clause(
        src_name, src_prop.get_domain_names(), src_prop.get_range_names()
    )
    tgt_desc = format_domain_range_clause(
        tgt_name, tgt_prop.get_domain_names(), tgt_prop.get_range_names()
    )

    return (
        f'We have two properties from different ontologies.\n\n'
        f'The first property is {src_desc}.\n\n'
        f'The second property is {tgt_desc}.\n\n'
        f'Do these properties represent the same relationship?\n\n'
        f'{_response_instruction}'
    )


def oupt_prop_domain_range_synonyms(src_prop, tgt_prop) -> str:
    """
    P3 — property equivalence with domain, range, and synonyms
    """
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)

    src_syn_text = format_synonyms_parenthetical(src_prop.get_synonyms(), src_name)
    tgt_syn_text = format_synonyms_parenthetical(tgt_prop.get_synonyms(), tgt_name)

    # build the property description lines with synonyms
    src_line = f'The first property is "{src_name}"{src_syn_text}'
    tgt_line = f'The second property is "{tgt_name}"{tgt_syn_text}'

    # append domain/range context with synonyms
    src_domain = src_prop.get_domain_names()
    src_range = src_prop.get_range_names()
    tgt_domain = tgt_prop.get_domain_names()
    tgt_range = tgt_prop.get_range_names()

    if src_domain or src_range:
        src_dr = format_domain_range_clause(
            src_name, src_domain, src_range,
            domain_synonyms=src_prop.get_domain_synonyms(),
            range_synonyms=src_prop.get_range_synonyms(),
            include_synonyms=True,
        )
        # extract the connecting clause (everything after the property name)
        connecting = src_dr[len(f'"{src_name}"'):]
        src_line += connecting

    if tgt_domain or tgt_range:
        tgt_dr = format_domain_range_clause(
            tgt_name, tgt_domain, tgt_range,
            domain_synonyms=tgt_prop.get_domain_synonyms(),
            range_synonyms=tgt_prop.get_range_synonyms(),
            include_synonyms=True,
        )
        connecting = tgt_dr[len(f'"{tgt_name}"'):]
        tgt_line += connecting

    return (
        f'We have two properties from different ontologies.\n\n'
        f'{src_line}.\n\n'
        f'{tgt_line}.\n\n'
        f'Do these properties represent the same relationship?\n\n'
        f'{_response_instruction}'
    )



# -----------------------------------------------------------------
# instance prompt templates (KG track):
# -----------------------------------------------------------------
# these templates accept (InstanceEntryAttr, InstanceEntryAttr) -> str
# forming a progressive ablation:
#   - labels only (no context)
#   - labels + type assertions
#   - labels + types + attributes (data + object properties, intersection-first)
#   - labels + types + full context (all property types, explicit separation)
# -----------------------------------------------------------------
# Module-level formatting function for NLF prompts in KG tracks
# -----------------------------------------------------------------
#   see the parameters that accept _global_fmt_fn
# -----------------------------------------------------------------

from prompt_utils import (
    format_instance_type_clause,
    format_instance_attribute_clause,
    select_intersecting_properties,
    # format_instance_attribute_clause_nl, # <-- this resulted in worse performance
)

# common callback for fmt fn (change in one place, 
# could add a service container / singleton type pattern)
# _GLOBAL_FMT_FN = format_instance_attribute_clause


_global_fmt_fn: Callable = format_instance_attribute_clause


def set_global_fmt_fn(fn: Callable | None) -> None:
    """Set the global formatting function for natural language friendly prompts for KG track"""
    _global_fmt_fn = fn


def get_global_fmt_fn() -> Callable:
    """Return the current fmt fn"""
    return _global_fmt_fn


#################


def oupt_inst_labels_only(src_inst, tgt_inst) -> str:
    """
    instance equivalence with labels only
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is "{src_label}".\n\n'
        f'The second is "{tgt_label}".\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )


def oupt_inst_labels_with_types(src_inst, tgt_inst) -> str:
    """
    instance equivalence with labels and class assertions
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    # Build description lines
    src_desc = f'"{src_label}"'
    if src_type_clause:
        src_desc += f', {src_type_clause}'

    tgt_desc = f'"{tgt_label}"'
    if tgt_type_clause:
        tgt_desc += f', {tgt_type_clause}'

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is {src_desc}.\n\n'
        f'The second is {tgt_desc}.\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )


def oupt_inst_types_and_attributes(src_inst, tgt_inst, max_properties: int = 3, fmt_fn: Callable = _global_fmt_fn) -> str:
    """
    instance equivalence with types and attributes
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    # Select properties with intersection-first heuristic
    src_all_props = src_inst.get_all_properties()
    tgt_all_props = tgt_inst.get_all_properties()
    src_selected, tgt_selected = select_intersecting_properties(
        src_all_props, tgt_all_props, max_properties=max_properties
    )

    src_attr_clause = fmt_fn(src_selected, max_properties)
    tgt_attr_clause = fmt_fn(tgt_selected, max_properties)

    # Build description lines
    src_desc = f'"{src_label}"'
    if src_type_clause:
        src_desc += f', {src_type_clause}'
    if src_attr_clause:
        src_desc += f' and {src_attr_clause}'

    tgt_desc = f'"{tgt_label}"'
    if tgt_type_clause:
        tgt_desc += f', {tgt_type_clause}'
    if tgt_attr_clause:
        tgt_desc += f' and {tgt_attr_clause}'

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is {src_desc}.\n\n'
        f'The second is {tgt_desc}.\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )


def oupt_inst_full_context(src_inst, tgt_inst, max_properties: int = 3, fmt_fn: Callable = _global_fmt_fn) -> str:
    """
    instance equivalence with full context
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    # separate data and object properties, select with intersection-first
    src_data = src_inst.get_data_properties()
    tgt_data = tgt_inst.get_data_properties()
    src_obj = src_inst.get_object_properties()
    tgt_obj = tgt_inst.get_object_properties()

    # select data properties
    src_data_sel, tgt_data_sel = select_intersecting_properties(
        src_data, tgt_data, max_properties=max_properties
    )
    # select object properties
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(
        src_obj, tgt_obj, max_properties=max_properties
    )

    src_data_clause = fmt_fn(src_data_sel, max_properties)
    tgt_data_clause = fmt_fn(tgt_data_sel, max_properties)
    src_obj_clause = fmt_fn(src_obj_sel, max_properties)
    tgt_obj_clause = fmt_fn(tgt_obj_sel, max_properties)

    # build description lines with explicit separation
    def _build_desc(label, type_clause, data_clause, obj_clause):
        desc = f'"{label}"'
        if type_clause:
            desc += f', {type_clause}'
        if data_clause:
            desc += f' and has attributes: {data_clause}'
        if obj_clause:
            desc += f' and has relationships: {obj_clause}'
        return desc

    src_desc = _build_desc(src_label, src_type_clause, src_data_clause, src_obj_clause)
    tgt_desc = _build_desc(tgt_label, tgt_type_clause, tgt_data_clause, tgt_obj_clause)

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is {src_desc}.\n\n'
        f'The second is {tgt_desc}.\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )


# intersection-only variants:
# these are identical above prompts \w attributes & full context except they pass intersection_only=True
# this applies select_intersecting_properties(), excluding non-shared predicates


def oupt_inst_types_and_attributes_intersect(src_inst, tgt_inst, max_properties: int = 3, fmt_fn: Callable = _global_fmt_fn) -> str:
    """
    instance equivalence with types and intersection-only attributes
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    # select properties — intersection only, no random fill
    src_all_props = src_inst.get_all_properties()
    tgt_all_props = tgt_inst.get_all_properties()
    src_selected, tgt_selected = select_intersecting_properties(
        src_all_props, tgt_all_props,
        max_properties=max_properties,
        intersection_only=True,
    )

    src_attr_clause = fmt_fn(src_selected, max_properties)
    tgt_attr_clause = fmt_fn(tgt_selected, max_properties)

    # build description lines
    src_desc = f'"{src_label}"'
    if src_type_clause:
        src_desc += f', {src_type_clause}'
    if src_attr_clause:
        src_desc += f' and {src_attr_clause}'

    tgt_desc = f'"{tgt_label}"'
    if tgt_type_clause:
        tgt_desc += f', {tgt_type_clause}'
    if tgt_attr_clause:
        tgt_desc += f' and {tgt_attr_clause}'

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is {src_desc}.\n\n'
        f'The second is {tgt_desc}.\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )


def oupt_inst_full_context_intersect(src_inst, tgt_inst, max_properties: int = 3, fmt_fn: Callable = _global_fmt_fn) -> str:
    """
    instance equivalence with full context, intersection-only
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    # separate data and object properties, select intersection only
    src_data = src_inst.get_data_properties()
    tgt_data = tgt_inst.get_data_properties()
    src_obj = src_inst.get_object_properties()
    tgt_obj = tgt_inst.get_object_properties()

    # select data properties — intersection only
    src_data_sel, tgt_data_sel = select_intersecting_properties(
        src_data, tgt_data,
        max_properties=max_properties,
        intersection_only=True,
    )
    # select object properties — intersection only
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(
        src_obj, tgt_obj,
        max_properties=max_properties,
        intersection_only=True,
    )

    src_data_clause = fmt_fn(src_data_sel, max_properties)
    tgt_data_clause = fmt_fn(tgt_data_sel, max_properties)
    src_obj_clause = fmt_fn(src_obj_sel, max_properties)
    tgt_obj_clause = fmt_fn(tgt_obj_sel, max_properties)

    # build description lines with explicit separation
    def _build_desc(label, type_clause, data_clause, obj_clause):
        desc = f'"{label}"'
        if type_clause:
            desc += f', {type_clause}'
        if data_clause:
            desc += f' and has attributes: {data_clause}'
        if obj_clause:
            desc += f' and has relationships: {obj_clause}'
        return desc

    src_desc = _build_desc(src_label, src_type_clause, src_data_clause, src_obj_clause)
    tgt_desc = _build_desc(tgt_label, tgt_type_clause, tgt_data_clause, tgt_obj_clause)

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is {src_desc}.\n\n'
        f'The second is {tgt_desc}.\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )



# --> entropy-ranked intersection-only variants <--
# these combine intersection-only selection with entropy-based ranking:
# shared predicates are ranked by descending Shannon entropy of their value distribution across the
# ontology so the most discriminating predicates appear first in the prompt


def _get_merged_entropies(src_inst, tgt_inst) -> dict:
    """
    compute and merge predicate entropies from both ontologies
    """
    src_ent = src_inst.onto.compute_predicate_entropies()
    tgt_ent = tgt_inst.onto.compute_predicate_entropies()
    merged = {}
    merged.update(src_ent)
    for uri, entropy in tgt_ent.items():
        if uri not in merged or entropy > merged[uri]:
            merged[uri] = entropy
    return merged


def oupt_inst_types_and_attributes_entropy(src_inst, tgt_inst, max_properties: int = 3, fmt_fn: Callable = _global_fmt_fn) -> str:
    """
    instance equivalence with entropy-ranked intersection-only attributes.
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    # compute merged entropy ranking
    entropies = _get_merged_entropies(src_inst, tgt_inst)

    # select properties — intersection only, ranked by entropy
    src_all_props = src_inst.get_all_properties()
    tgt_all_props = tgt_inst.get_all_properties()
    src_selected, tgt_selected = select_intersecting_properties(
        src_all_props, tgt_all_props,
        max_properties=max_properties,
        intersection_only=True,
        predicate_entropies=entropies,
    )

    src_attr_clause = fmt_fn(src_selected, max_properties)
    tgt_attr_clause = fmt_fn(tgt_selected, max_properties)

    # build description lines
    src_desc = f'"{src_label}"'
    if src_type_clause:
        src_desc += f', {src_type_clause}'
    if src_attr_clause:
        src_desc += f' and {src_attr_clause}'

    tgt_desc = f'"{tgt_label}"'
    if tgt_type_clause:
        tgt_desc += f', {tgt_type_clause}'
    if tgt_attr_clause:
        tgt_desc += f' and {tgt_attr_clause}'

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is {src_desc}.\n\n'
        f'The second is {tgt_desc}.\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )


def oupt_inst_full_context_entropy(src_inst, tgt_inst, max_properties: int = 3, fmt_fn: Callable = _global_fmt_fn) -> str:
    """
    instance equivalence with full context, entropy-ranked intersection-only
    same as I4-int but ranks shared predicates by descending Shannon entropy
    data and object properties are ranked independently within their respective sections
    """
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    # compute merged entropy ranking
    entropies = _get_merged_entropies(src_inst, tgt_inst)

    # separate data and object properties, select intersection only + entropy
    src_data = src_inst.get_data_properties()
    tgt_data = tgt_inst.get_data_properties()
    src_obj = src_inst.get_object_properties()
    tgt_obj = tgt_inst.get_object_properties()

    # select data properties — intersection only, entropy ranked
    src_data_sel, tgt_data_sel = select_intersecting_properties(
        src_data, tgt_data,
        max_properties=max_properties,
        intersection_only=True,
        predicate_entropies=entropies,
    )
    # select object properties — intersection only, entropy ranked
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(
        src_obj, tgt_obj,
        max_properties=max_properties,
        intersection_only=True,
        predicate_entropies=entropies,
    )

    src_data_clause = fmt_fn(src_data_sel, max_properties)
    tgt_data_clause = fmt_fn(tgt_data_sel, max_properties)
    src_obj_clause = fmt_fn(src_obj_sel, max_properties)
    tgt_obj_clause = fmt_fn(tgt_obj_sel, max_properties)

    # build description lines with explicit separation
    def _build_desc(label, type_clause, data_clause, obj_clause):
        desc = f'"{label}"'
        if type_clause:
            desc += f', {type_clause}'
        if data_clause:
            desc += f' and has attributes: {data_clause}'
        if obj_clause:
            desc += f' and has relationships: {obj_clause}'
        return desc

    src_desc = _build_desc(src_label, src_type_clause, src_data_clause, src_obj_clause)
    tgt_desc = _build_desc(tgt_label, tgt_type_clause, tgt_data_clause, tgt_obj_clause)

    return (
        f'We have two entities from different knowledge graphs.\n\n'
        f'The first is {src_desc}.\n\n'
        f'The second is {tgt_desc}.\n\n'
        f'Do these refer to the same entity?\n\n'
        f'{_response_instruction}'
    )