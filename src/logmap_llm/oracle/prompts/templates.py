'''
TODO: refactor & clean-up this file 
(there's far too many prompts & construction methods are not intuitive)
'''
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

from logmap_llm.constants import (
    EntityType,
    PAIRS_SEPARATOR,
    RESPONSE_INSTRUCTION,
    DEFAULT_ANSWER_FORMAT,
    ANSWER_FORMATS,
)

from logmap_llm.ontology.entities import (
    OntologyEntryAttr,
    ClassNotFoundError,
)

from logmap_llm.oracle.prompts.formatting import (
    format_hierarchy,
    get_single_name,
    select_best_direct_entity_names,
    select_best_direct_entity_names_with_synonyms,
    select_best_sequential_hierarchy_with_synonyms,
    format_sibling_context,
    format_synonyms_parenthetical,
    format_domain_range_clause,
    format_restriction_context,
    format_relational_signature,
    format_property_characteristics,
    format_instance_type_clause,
    format_instance_attribute_clause,
    select_intersecting_properties,
)

import pandas as pd

from tqdm import tqdm

# TODO: this (GLOBAL) pattern should probably be more appropraitely contained

###
# GLOBAL ANSWER FORMAT SETTINGS
###

_response_instruction: str = RESPONSE_INSTRUCTION[DEFAULT_ANSWER_FORMAT]

def set_answer_format(fmt: str) -> None:
    global _response_instruction
    if fmt not in ANSWER_FORMATS:
        raise ValueError(
            f"Unknown answer_format '{fmt}'. Valid options: {sorted(ANSWER_FORMATS)}"
        )
    _response_instruction = RESPONSE_INSTRUCTION[fmt]


def get_response_instruction() -> str:
    return _response_instruction


###
# GLOBAL ONTOLOGICAL DOMAIN SETTINGS
###

_ontology_domain: str = "biomedical"

TRACK_TO_DOMAIN = {
    "bioml": "biomedical",
    "anatomy": "biomedical",
    "conference": "conference",
    "knowledgegraph": "knowledge graph",
}


def set_ontology_domain(track: str | None) -> None:
    global _ontology_domain
    if track and track in TRACK_TO_DOMAIN:
        _ontology_domain = TRACK_TO_DOMAIN[track]


def get_ontology_domain() -> str:
    return _ontology_domain


###
# GLOBAL KG FORMATTING
###

_global_fmt_fn: Callable = format_instance_attribute_clause

def set_global_fmt_fn(fn: Callable | None) -> None:
    global _global_fmt_fn
    if fn is not None:
        _global_fmt_fn = fn


def get_global_fmt_fn() -> Callable:
    return _global_fmt_fn


###
# TEMPLATE REGISTRY
###

@dataclass(frozen=True)
class TemplateSpec:
    """Specification for a registered prompt template."""
    fn: Callable
    entity_type: EntityType = EntityType.CLASS
    bidirectional: bool = False
    requires_siblings: bool = False


class TemplateRegistry:
    """Registry for prompt template functions.

    Templates are registered via the ``@registry.register`` decorator
    or by calling ``registry.register_fn()`` directly.
    """

    def __init__(self):
        self._templates: dict[str, TemplateSpec] = {}

    def register(
        self,
        name: str,
        entity_type: EntityType = EntityType.CLASS,
        bidirectional: bool = False,
        requires_siblings: bool = False,
    ) -> Callable:
        """Decorator for registering template functions."""
        def decorator(fn: Callable) -> Callable:
            self._templates[name] = TemplateSpec(
                fn=fn,
                entity_type=entity_type,
                bidirectional=bidirectional,
                requires_siblings=requires_siblings,
            )
            return fn
        return decorator

    def register_fn(
        self,
        name: str,
        fn: Callable,
        entity_type: EntityType = EntityType.CLASS,
        bidirectional: bool = False,
        requires_siblings: bool = False,
    ) -> None:
        """Register a template function directly (non-decorator)."""
        self._templates[name] = TemplateSpec(
            fn=fn,
            entity_type=entity_type,
            bidirectional=bidirectional,
            requires_siblings=requires_siblings,
        )

    def get(self, name: str) -> TemplateSpec:
        """Look up a template by name."""
        if name not in self._templates:
            raise KeyError(
                f"Template '{name}' not found. "
                f"Available: {list(self._templates.keys())}"
            )
        return self._templates[name]

    def is_bidirectional(self, name: str) -> bool:
        return self.get(name).bidirectional

    def requires_siblings(self, name: str) -> bool:
        return self.get(name).requires_siblings

    def get_by_entity_type(self, et: EntityType) -> dict[str, TemplateSpec]:
        return {k: v for k, v in self._templates.items() if v.entity_type == et}

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)

    def keys(self) -> list[str]:
        return list(self._templates.keys())


# Global registry instance
registry = TemplateRegistry()


###
# HELPERS
###


def _retrieve_siblings(entity, sibling_selector=None, max_count=2):
    if sibling_selector is not None:
        return sibling_selector.select_siblings(entity, max_count=max_count)
    else:
        sib_entries = entity.get_siblings(max_count=max_count)
        return [
            (min(s.get_preferred_names()) if s.get_preferred_names()
             else str(s.thing_class.name), 1.0)
            for s in sib_entries
        ]

def _get_merged_entropies(src_inst, tgt_inst) -> dict:
    src_ent = src_inst.onto.compute_predicate_entropies()
    tgt_ent = tgt_inst.onto.compute_predicate_entropies()
    merged = {}
    merged.update(src_ent)
    for uri, entropy in tgt_ent.items():
        if uri not in merged or entropy > merged[uri]:
            merged[uri] = entropy
    return merged





# -------------------------------
#        Old Prompts Section
# -------------------------------

# def prompt_all_data_dummy(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
#     return f"""
#     **Task Description:**
#     Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

#     1. **Source Entity:**
#     **All Entity names:** {src_entety.get_preffered_names()}
#     **Parent Entity Namings:** {src_entety.get_parents_preferred_names()}
#     **Child Entity Namings:** {src_entety.get_children_preferred_names()}

#     2. **Target Entity:**
#     **All Entity names:** {tgt_entety.get_preffered_names()}
#     **Parent Entity Namings:** {tgt_entety.get_parents_preferred_names()}
#     **Child Entity Namings:** {tgt_entety.get_children_preferred_names()}

#     Write "Yes" if the entities refer to the same concepts, and "No" otherwise.
#     """.strip()


# def prompt_only_names(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
#     return f"""
#     Given two entities from different ontologies with their names, determine if these concepts are the same:

#     1. Source Entity:
#     All Entity names: {src_entety.get_all_entity_names()}

#     2. Target Entity:
#     All Entity names: {tgt_entety.get_all_entity_names()}

#     Response with True or False
#     """.strip()


# def prompt_with_hierarchy(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
#     return f"""
#     Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

#     1. Source Entity:
#     All Entity names: {src_entety.get_all_entity_names()}
#     Parent Entity Namings: {src_entety.get_parents_preferred_names()}
#     Child Entity Namings: {src_entety.get_children_preferred_names()}

#     2. Target Entity:
#     All Entity names: {tgt_entety.get_all_entity_names()}
#     Parent Entity Namings: {tgt_entety.get_parents_preferred_names()}
#     Child Entity Namings: {tgt_entety.get_children_preferred_names()}

#     Response with True or False
#     """.strip()


# def prompt_only_with_parents(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
#     return f"""
#     Given two entities from different ontologies with their names and parent relationships, determine if these concepts are the same:

#     1. Source Entity:
#     All Entity names: {src_entety.get_all_entity_names()}
#     Parent Entity Namings: {src_entety.get_parents_preferred_names()}

#     2. Target Entity:
#     All Entity names: {tgt_entety.get_all_entity_names()}
#     Parent Entity Namings: {tgt_entety.get_parents_preferred_names()}

#     Response with True or False
#     """.strip()


# def prompt_only_with_children(src_entety: OntologyEntryAttr, tgt_entety: OntologyEntryAttr) -> str:
#     return f"""
#     Given two entities from different ontologies with their names and child relationships, determine if these concepts are the same:

#     1. Source Entity:
#     All Entity names: {src_entety.get_all_entity_names()}
#     Child Entity Namings: {src_entety.get_children_preferred_names()}

#     2. Target Entity:
#     All Entity names: {tgt_entety.get_all_entity_names()}
#     Child Entity Namings: {tgt_entety.get_children_preferred_names()}

#     Response with True or False
#     """.strip()


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


@registry.register("one_level_of_parents_structured")
def oupt_one_level_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names) = select_best_direct_entity_names(src_entity, tgt_entity)
    
    prompt_lines = [
        "Analyze the following entities, each originating from a distinct biomedical ontology.",
        "Your task is to assess whether they represent the **same ontological concept**, "
        "considering both their semantic meaning and hierarchical position.",
        f'\n1. Source entity: "{src_entity_names}"',
        f"\t- Direct ontological parent: {src_parent}",
        f'\n2. Target entity: "{tgt_entity_names}"',
        f"\t- Direct ontological parent: {tgt_parent}",
        '\nAre these entities **ontologically equivalent** within their respective '
        'ontologies? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents_structured")
def oupt_two_levels_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2))
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2))
    
    prompt_lines = [
        "Analyze the following entities, each originating from a distinct biomedical ontology.",
        "Each is represented by its **ontological lineage**, capturing its hierarchical "
        "placement from the most general to the most specific level.",
        f"\n1. Source entity ontological lineage:\n{src_hierarchy}",
        f"\n2. Target entity ontological lineage:\n{tgt_hierarchy}",
        '\nBased on their **ontological positioning, hierarchical relationships, and '
        'semantic alignment**, do these entities represent the **same ontological '
        'concept**? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)



#
# Natural language friendly prompts
#



@registry.register("one_level_of_parents")
def oupt_one_level_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names) = select_best_direct_entity_names(src_entity, tgt_entity)
    
    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        (f'The first one is "{src_entity_names}"' + (f', which belongs to the broader category "{src_parent}"' if src_parent else "")),
        (f'The second one is "{tgt_entity_names}"' + (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else "")),
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents")
def oupt_two_levels_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)

    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
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
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)



#
# Natural language prompts with synonyms
#



@registry.register("one_level_of_parents_and_synonyms")
def oupt_one_level_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms) = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in sorted(src_synonyms))) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in sorted(tgt_synonyms))) if tgt_synonyms else ""
    
    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents_and_synonyms")
def oupt_two_levels_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)
    
    (src_syns, tgt_syns, src_parents_syns, tgt_parents_syns) = select_best_sequential_hierarchy_with_synonyms(src_entity, tgt_entity, max_level=2)

    def describe_entity(hierarchy, entity_syns, parent_syns):
        name_part = f'"{hierarchy[0]}"'
        if entity_syns:
            alt = ", ".join(f'"{s}"' for s in sorted(entity_syns))
            name_part += f", also known as {alt}"
        parts = [name_part]
        labels = ["belongs to broader category", "under the even broader category", "under the even broader category"]
        for i, parent_name in enumerate(hierarchy[1:]):
            text = f'{labels[i]} "{parent_name}"'
            if parent_syns[i]:
                alt = ", ".join(f'"{s}"' for s in sorted(parent_syns[i]))
                text += f" (also known as {alt})"
            parts.append(text)
        return ", ".join(parts)

    src_desc = describe_entity(src_hierarchy, src_syns, src_parents_syns)
    tgt_desc = describe_entity(tgt_hierarchy, tgt_syns, tgt_parents_syns)
    
    prompt_lines = [
        f"We have two entities from different {_ontology_domain} ontologies.",
        f"The first one is {src_desc}.",
        f"The second one is {tgt_desc}.",
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



###
# SUBSUMPTION-BASED PROMPTS:
# TODO: refine prompts to only those that are required
###



@registry.register("sub_labels_only", bidirectional=True)
def oupt_sub_labels_only(src_entity, tgt_entity):
    
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"? {_response_instruction}',
    ])



@registry.register("sub_parents_synonyms", bidirectional=True)
def oupt_sub_parents_synonyms(src_entity, tgt_entity):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"? {_response_instruction}',
    ])



@registry.register("sub_syns_conj_parent", bidirectional=True)
def oupt_sub_syns_conj_parent(src_entity, tgt_entity):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"{tgt_qual}?',
        f' {_response_instruction}',
    ])



###
# ADDITIONAL PROMPTS
# TODO: refine prompts to only those that are required
###



@registry.register("single_subs")
def oupt_single_subs(src_entity, tgt_entity):
    
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}".',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}".',
        f'\nAre both statements true? {_response_instruction}',
    ])



@registry.register("single_subs_with_parents")
def oupt_single_subs_with_parents(src_entity, tgt_entity):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}".',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}".',
        f'\nAre both statements true? {_response_instruction}',
    ])



@registry.register("single_subs_with_conj_parent")
def oupt_single_subs_with_conj_parent(src_entity, tgt_entity):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}"{tgt_qual}.',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}"{src_qual}.',
        f'\nAre both statements true? {_response_instruction}',
    ])



@registry.register("equiv_test_for_ancestral_disjointness", bidirectional=True)
def oupt_equiv_test_for_ancestral_disjointness(src_entity, tgt_entity):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nCan something be a "{src_entity_names}"{src_qual} at the same time as being a "{tgt_entity_names}"{tgt_qual}?',
        _response_instruction,
    ])



@registry.register("equiv_with_ancestral_disjointness", bidirectional=True)
def oupt_equiv_with_ancestral_disjointness(src_entity, tgt_entity):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nIf something is a "{src_entity_names}"{src_qual}, is it also a "{tgt_entity_names}"{tgt_qual}?',
        f'\n{_response_instruction}',
    ])



@registry.register("deductive_equiv")
def oupt_deductive_equiv(src_entity, tgt_entity):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    facts = [] # _premises_

    # onto 1:
    facts.append(f'- "{src_entity_names}" is a kind of "{src_parent}" (Ontology 1)' if src_parent else f'- "{src_entity_names}" is a concept in Ontology 1')
    if src_synonyms:
        facts.append(f'- "{src_entity_names}" is also known as ' + " and ".join(f'"{s}"' for s in src_synonyms) + ' (Ontology 1)')
    
    # onto 2:
    facts.append(f'- "{tgt_entity_names}" is a kind of "{tgt_parent}" (Ontology 2)' if tgt_parent else f'- "{tgt_entity_names}" is a concept in Ontology 2')
    if tgt_synonyms:
        facts.append(f'- "{tgt_entity_names}" is also known as ' + " and ".join(f'"{s}"' for s in tgt_synonyms) + ' (Ontology 2)')
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies. Consider the following facts:",
        "\n".join(facts),
        f'\nGiven these facts, can you conclude that "{src_entity_names}" (Ontology 1) and "{tgt_entity_names}" (Ontology 2) refer to the same real-world concept? {_response_instruction}',
    ])



@registry.register("falsification_equiv", requires_siblings=True)
def oupt_falsification_equiv(src_entity, tgt_entity, sibling_selector=None):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".{tgt_sibling_clause}',
        f'\nFirst consider: could something classified as a "{src_parent}" in one ontology plausibly also be classified as a "{tgt_parent}" in another? If the categories suggest entirely different domains, the concepts are unlikely to match. If the categories are compatible or overlapping, the concepts may well match.',
        f'\nDo "{src_entity_names}" (a "{src_parent}") and "{tgt_entity_names}" (a "{tgt_parent}") refer to the same concept? {_response_instruction}',
    ])



@registry.register("synonyms_only")
def oupt_synonyms_only(src_entity, tgt_entity):
    
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        "We have two entities from different ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nDo they mean the same thing? {_response_instruction}',
    ])



###
# SIBLING-BASED PROMPTS
###



@registry.register("equiv_parents_siblings", requires_siblings=True)
def oupt_equiv_parents_siblings(src_entity, tgt_entity, sibling_selector=None):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, _, _ = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}", which falls under the category "{src_parent}".{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}", which falls under the category "{tgt_parent}".{tgt_sibling_clause}',
        f'\nDo they mean the same thing? {_response_instruction}',
    ])



@registry.register("equiv_parents_synonyms_siblings", requires_siblings=True)
def oupt_equiv_parents_synonyms_siblings(src_entity, tgt_entity, sibling_selector=None):
    
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".{tgt_sibling_clause}',
        f'\nDo they mean the same thing? {_response_instruction}',
    ])



###
# PROPERTY-BASED TEMPLATES
###



@registry.register("prop_labels_only", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_labels_only(src_prop, tgt_prop):
    
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)
    
    return f'We have two properties from different ontologies.\n\nThe first property is "{src_name}".\n\nThe second property is "{tgt_name}".\n\nDo these properties represent the same relationship?\n\n{_response_instruction}'



@registry.register("prop_domain_range", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_domain_range(src_prop, tgt_prop):
    
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)
    
    src_desc = format_domain_range_clause(src_name, src_prop.get_domain_names(), src_prop.get_range_names())
    tgt_desc = format_domain_range_clause(tgt_name, tgt_prop.get_domain_names(), tgt_prop.get_range_names())
    
    return f'We have two properties from different ontologies.\n\nThe first property is {src_desc}.\n\nThe second property is {tgt_desc}.\n\nDo these properties represent the same relationship?\n\n{_response_instruction}'



@registry.register("prop_domain_range_synonyms", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_domain_range_synonyms(src_prop, tgt_prop):
    
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)
    
    src_syn_text = format_synonyms_parenthetical(src_prop.get_synonyms(), src_name)
    tgt_syn_text = format_synonyms_parenthetical(tgt_prop.get_synonyms(), tgt_name)
    
    src_line = f'The first property is "{src_name}"{src_syn_text}'
    tgt_line = f'The second property is "{tgt_name}"{tgt_syn_text}'
    
    if src_prop.get_domain_names() or src_prop.get_range_names():
        src_dr = format_domain_range_clause(src_name, src_prop.get_domain_names(), src_prop.get_range_names(), domain_synonyms=src_prop.get_domain_synonyms(), range_synonyms=src_prop.get_range_synonyms(), include_synonyms=True)
        src_line += src_dr[len(f'"{src_name}"'):]
    
    if tgt_prop.get_domain_names() or tgt_prop.get_range_names():
        tgt_dr = format_domain_range_clause(tgt_name, tgt_prop.get_domain_names(), tgt_prop.get_range_names(), domain_synonyms=tgt_prop.get_domain_synonyms(), range_synonyms=tgt_prop.get_range_synonyms(), include_synonyms=True)
        tgt_line += tgt_dr[len(f'"{tgt_name}"'):]
    
    return f'We have two properties from different ontologies.\n\n{src_line}.\n\n{tgt_line}.\n\nDo these properties represent the same relationship?\n\n{_response_instruction}'



@registry.register("prop_with_inverse_and_chars", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_with_inverse_and_chars(src_prop, tgt_prop):
    
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)
    
    src_desc = format_domain_range_clause(src_name, src_prop.get_domain_names(), src_prop.get_range_names())
    tgt_desc = format_domain_range_clause(tgt_name, tgt_prop.get_domain_names(), tgt_prop.get_range_names())
    
    src_chars = src_prop.get_characteristics()
    src_inverse = src_prop.get_inverse_name()
    
    tgt_chars = tgt_prop.get_characteristics()
    tgt_inverse = tgt_prop.get_inverse_name()
    
    src_char_text = " " + format_property_characteristics(src_chars, src_inverse) if (src_chars or src_inverse) else ""
    tgt_char_text = " " + format_property_characteristics(tgt_chars, tgt_inverse) if (tgt_chars or tgt_inverse) else ""
    
    return f'We have two properties from different ontologies.\n\nThe first property is {src_desc}.{src_char_text}\n\nThe second property is {tgt_desc}.{tgt_char_text}\n\nDo these properties represent the same relationship?\n\nRespond with "True" or "False".'



###
# CONF ENRICHED PROMPTS
###


@registry.register("class_with_restrictions")
def oupt_class_with_restrictions(src_entity, tgt_entity):
    
    _, _, src_name, tgt_name, src_syns, tgt_syns = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in src_syns)) if src_syns else ""
    tgt_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_syns)) if tgt_syns else ""
    
    src_restrictions = src_entity.get_restrictions()
    tgt_restrictions = tgt_entity.get_restrictions()
    
    src_restr_text = " " + format_restriction_context(src_restrictions) if src_restrictions else ""
    tgt_restr_text = " " + format_restriction_context(tgt_restrictions) if tgt_restrictions else ""
    
    return f'We have two entities from different ontologies.\nThe first one is "{src_name}"{src_syn_text}.{src_restr_text}\nThe second one is "{tgt_name}"{tgt_syn_text}.{tgt_restr_text}\n\nDo they mean the same thing? Respond with "True" or "False".'



@registry.register("class_role_signature")
def oupt_class_role_signature(src_entity, tgt_entity):
    
    _, _, src_name, tgt_name, src_syns, tgt_syns = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in src_syns)) if src_syns else ""
    tgt_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_syns)) if tgt_syns else ""
    
    src_sig = src_entity.get_relational_signature()
    tgt_sig = tgt_entity.get_relational_signature()
    
    src_sig_text = " " + format_relational_signature(src_sig) if (src_sig.get('as_domain') or src_sig.get('as_range')) else ""
    tgt_sig_text = " " + format_relational_signature(tgt_sig) if (tgt_sig.get('as_domain') or tgt_sig.get('as_range')) else ""
    
    return f'We have two entities from different ontologies.\nThe first one is "{src_name}"{src_syn_text}.{src_sig_text}\nThe second one is "{tgt_name}"{tgt_syn_text}.{tgt_sig_text}\n\nDo they mean the same thing? Respond with "True" or "False".'



###
# INSTANCE-BASED PROMPTS
###


@registry.register("inst_labels_only", entity_type=EntityType.INSTANCE)
def oupt_inst_labels_only(src_inst, tgt_inst):
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is "{src_label}".\n\nThe second is "{tgt_label}".\n\nDo these refer to the same entity?\n\n{_response_instruction}'



@registry.register("inst_labels_with_types", entity_type=EntityType.INSTANCE)
def oupt_inst_labels_with_types(src_inst, tgt_inst):
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "")
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



@registry.register("inst_types_and_attributes", entity_type=EntityType.INSTANCE)
def oupt_inst_types_and_attributes(src_inst, tgt_inst, max_properties=3, fmt_fn=None):
    
    if fmt_fn is None: fmt_fn = _global_fmt_fn
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_selected, tgt_selected = select_intersecting_properties(src_inst.get_all_properties(), tgt_inst.get_all_properties(), max_properties=max_properties)
    
    src_attr = fmt_fn(src_selected, max_properties)
    tgt_attr = fmt_fn(tgt_selected, max_properties)
    
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "") + (f' and {src_attr}' if src_attr else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "") + (f' and {tgt_attr}' if tgt_attr else "")
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



@registry.register("inst_full_context", entity_type=EntityType.INSTANCE)
def oupt_inst_full_context(src_inst, tgt_inst, max_properties=3, fmt_fn=None):
    
    if fmt_fn is None: fmt_fn = _global_fmt_fn
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_data_sel, tgt_data_sel = select_intersecting_properties(src_inst.get_data_properties(), tgt_inst.get_data_properties(), max_properties=max_properties)
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(src_inst.get_object_properties(), tgt_inst.get_object_properties(), max_properties=max_properties)
    
    def _build(label, tc, dc, oc):
        d = f'"{label}"' + (f', {tc}' if tc else "") + (f' and has attributes: {dc}' if dc else "") + (f' and has relationships: {oc}' if oc else "")
        return d
    
    src_desc = _build(src_label, src_type_clause, fmt_fn(src_data_sel, max_properties), fmt_fn(src_obj_sel, max_properties))
    tgt_desc = _build(tgt_label, tgt_type_clause, fmt_fn(tgt_data_sel, max_properties), fmt_fn(tgt_obj_sel, max_properties))
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



@registry.register("inst_types_and_attributes_intersect", entity_type=EntityType.INSTANCE)
def oupt_inst_types_and_attributes_intersect(src_inst, tgt_inst, max_properties=3, fmt_fn=None):
    
    if fmt_fn is None: fmt_fn = _global_fmt_fn
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_selected, tgt_selected = select_intersecting_properties(src_inst.get_all_properties(), tgt_inst.get_all_properties(), max_properties=max_properties, intersection_only=True)
    
    src_attr = fmt_fn(src_selected, max_properties)
    tgt_attr = fmt_fn(tgt_selected, max_properties)
    
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "") + (f' and {src_attr}' if src_attr else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "") + (f' and {tgt_attr}' if tgt_attr else "")
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



@registry.register("inst_full_context_intersect", entity_type=EntityType.INSTANCE)
def oupt_inst_full_context_intersect(src_inst, tgt_inst, max_properties=3, fmt_fn=None):
    
    if fmt_fn is None: fmt_fn = _global_fmt_fn
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_data_sel, tgt_data_sel = select_intersecting_properties(src_inst.get_data_properties(), tgt_inst.get_data_properties(), max_properties=max_properties, intersection_only=True)
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(src_inst.get_object_properties(), tgt_inst.get_object_properties(), max_properties=max_properties, intersection_only=True)
    
    def _build(label, tc, dc, oc):
        return f'"{label}"' + (f', {tc}' if tc else "") + (f' and has attributes: {dc}' if dc else "") + (f' and has relationships: {oc}' if oc else "")
    
    src_desc = _build(src_label, src_type_clause, fmt_fn(src_data_sel, max_properties), fmt_fn(src_obj_sel, max_properties))
    tgt_desc = _build(tgt_label, tgt_type_clause, fmt_fn(tgt_data_sel, max_properties), fmt_fn(tgt_obj_sel, max_properties))
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



@registry.register("inst_types_and_attributes_entropy", entity_type=EntityType.INSTANCE)
def oupt_inst_types_and_attributes_entropy(src_inst, tgt_inst, max_properties=3, fmt_fn=None):
    
    if fmt_fn is None: fmt_fn = _global_fmt_fn
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    entropies = _get_merged_entropies(src_inst, tgt_inst)
    src_selected, tgt_selected = select_intersecting_properties(src_inst.get_all_properties(), tgt_inst.get_all_properties(), max_properties=max_properties, intersection_only=True, predicate_entropies=entropies)
    
    src_attr = fmt_fn(src_selected, max_properties)
    tgt_attr = fmt_fn(tgt_selected, max_properties)
    
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "") + (f' and {src_attr}' if src_attr else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "") + (f' and {tgt_attr}' if tgt_attr else "")
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



@registry.register("inst_full_context_entropy", entity_type=EntityType.INSTANCE)
def oupt_inst_full_context_entropy(src_inst, tgt_inst, max_properties=3, fmt_fn=None):
    
    if fmt_fn is None: fmt_fn = _global_fmt_fn
    
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    entropies = _get_merged_entropies(src_inst, tgt_inst)
    src_data_sel, tgt_data_sel = select_intersecting_properties(src_inst.get_data_properties(), tgt_inst.get_data_properties(), max_properties=max_properties, intersection_only=True, predicate_entropies=entropies)
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(src_inst.get_object_properties(), tgt_inst.get_object_properties(), max_properties=max_properties, intersection_only=True, predicate_entropies=entropies)
    
    def _build(label, tc, dc, oc):
        return f'"{label}"' + (f', {tc}' if tc else "") + (f' and has attributes: {dc}' if dc else "") + (f' and has relationships: {oc}' if oc else "")
    
    src_desc = _build(src_label, src_type_clause, fmt_fn(src_data_sel, max_properties), fmt_fn(src_obj_sel, max_properties))
    tgt_desc = _build(tgt_label, tgt_type_clause, fmt_fn(tgt_data_sel, max_properties), fmt_fn(tgt_obj_sel, max_properties))
    
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



########
# ORACLE_PROMPT_BUILDING.py
########


'''
This module contains functionality supporting the building of
LLM Oracle 'user' prompts.

Note: LLM Oracle 'developer' prompts are managed elsewhere.
'''

def get_oracle_user_prompt_template_function(oupt_name: str) -> Callable:
    """Look up a template function by name from the registry."""
    return registry.get(oupt_name).fn


def build_oracle_user_prompts(
    oupt_name: str,
    onto_src_filepath: str,
    onto_tgt_filepath: str,
    m_ask_df: pd.DataFrame,
    OA_source=None,
    OA_target=None,
    sibling_selector=None,
    property_prompt_function=None,
    instance_prompt_function=None,
    property_prompt_name=None,
    instance_prompt_name=None
) -> dict:
    """
    Build oracle user prompts for all mappings in M_ask

    Parameters
    ----------
    oupt_name : str
        Name of the oracle user prompt template.
    
    onto_src_filepath : str
        Absolute path to the source ontology file.
    
    onto_tgt_filepath : str
        Absolute path to the target ontology file.
    
    m_ask_df : pd.DataFrame
        The M_ask mappings from LogMap.
    
    OA_source, OA_target : OntologyAccess, optional
        Pre-loaded ontology access objects.  If None, loaded here.
    
    sibling_selector : SiblingSelector, optional
        For sibling-aware templates.
    
    property_prompt_function : callable, optional
        Template function for property entities.
    
    instance_prompt_function : callable, optional
        Template function for instance entities.

    property_prompt_name : str, optional
        Template name for property entities.

    instance_prompt_name : str, optional
        Template name for instance entities.

    Returns
    -------
    dict
        Mapping of 'src_uri|tgt_uri' -> prompt string.
    """
    from logmap_llm.ontology.access import OntologyAccess
    from logmap_llm.ontology.entities import resolve_entity

    # Load ontologies if not provided
    if OA_source is None or OA_target is None:
        for onto_path in tqdm(iterable=[onto_src_filepath], desc='Preparing source ontology'):
            OA_source = OntologyAccess(onto_path, annotate_on_init=True)
        print()
        for onto_path in tqdm(iterable=[onto_tgt_filepath], desc='Preparing target ontology'):
            OA_target = OntologyAccess(onto_path, annotate_on_init=True)

    prompt_function = get_oracle_user_prompt_template_function(oupt_name)

    if sibling_selector is not None and registry.requires_siblings(oupt_name):
        prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

    if property_prompt_name and property_prompt_function is None:
        try: 
            property_prompt_function = get_oracle_user_prompt_template_function(property_prompt_name)
        except KeyError: 
            tqdm.write(f"  WARNING: Property template '{property_prompt_name}' not found")

    if instance_prompt_name and instance_prompt_function is None:
        try: 
            instance_prompt_function = get_oracle_user_prompt_template_function(instance_prompt_name)
        except KeyError: 
            tqdm.write(f"  WARNING: Instance template '{instance_prompt_name}' not found")

    print()
    print(f"Prompt template function obtained: {oupt_name}")
    if property_prompt_function: 
        print(f"Property prompt template: {property_prompt_name}")
    if instance_prompt_function: 
        print(f"Instance prompt template: {instance_prompt_name}")
    print()

    m_ask_oracle_user_prompts = {}
    skipped_mappings = []
    n_class, n_prop, n_inst = 0, 0, 0

    for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0], desc="Building the prompts"):
        row_series = row[1]
        src_uri, tgt_uri = row_series.iloc[0], row_series.iloc[1]
        try:
            src_entity, src_type = resolve_entity(src_uri, OA_source)
            tgt_entity, tgt_type = resolve_entity(tgt_uri, OA_target)
            
            if src_type == "instance" or tgt_type == "instance":
                
                if src_type != tgt_type:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": f"Mixed types: {src_type}/{tgt_type}"})
                    continue
                
                if instance_prompt_function is None:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": "No instance template"})
                    continue
                
                oracle_user_prompt = instance_prompt_function(src_entity, tgt_entity); n_inst += 1
            
            elif src_type == "property" or tgt_type == "property":
                
                if src_type != tgt_type:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": f"Mixed types: {src_type}/{tgt_type}"})
                    continue
                
                if property_prompt_function is None:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": "No property template"})
                    continue
                
                oracle_user_prompt = property_prompt_function(src_entity, tgt_entity); n_prop += 1
            
            else:
                oracle_user_prompt = prompt_function(src_entity, tgt_entity); n_class += 1

            m_ask_oracle_user_prompts[src_uri + PAIRS_SEPARATOR + tgt_uri] = oracle_user_prompt
        
        except (ClassNotFoundError, Exception) as e:
            skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": str(e)})
            tqdm.write(f"  WARNING: Skipping mapping - {e}")

    if skipped_mappings:
        print(f"[WARNING] Skipped {len(skipped_mappings)} mappings.")
    
    print(f"Prompts built: {n_class} class, {n_prop} property, {n_inst} instance ({len(m_ask_oracle_user_prompts)} total)")
    
    return m_ask_oracle_user_prompts



def build_oracle_user_prompts_bidirectional(
    oupt_name: str,
    onto_src_filepath: str,
    onto_tgt_filepath: str,
    m_ask_df,
    OA_source=None,
    OA_target=None,
    sibling_selector=None
) -> dict:
    from logmap_llm.ontology.access import OntologyAccess

    # Load ontologies if not provided
    if OA_source is None or OA_target is None:
        for onto_path in tqdm(iterable=[onto_src_filepath], desc='Preparing source ontology'):
            OA_source = OntologyAccess(onto_path, annotate_on_init=True)
        print()
        for onto_path in tqdm(iterable=[onto_tgt_filepath], desc='Preparing target ontology'):
            OA_target = OntologyAccess(onto_path, annotate_on_init=True)

    prompt_function = get_oracle_user_prompt_template_function(oupt_name)

    if sibling_selector is not None and registry.requires_siblings(oupt_name):
        prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

    print()
    print(f"Prompt template function obtained: {oupt_name} (bidirectional mode)")
    print()

    m_ask_oracle_user_prompts = {}
    skipped_mappings = []
    n_non_equiv_skipped = 0
    n_equiv_candidates = 0
    
    has_relation_col = m_ask_df.shape[1] > 2

    for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0], desc="Building bidirectional prompts"):
        
        row_series = row[1]
        src_uri, tgt_uri = row_series.iloc[0], row_series.iloc[1]
        
        if has_relation_col:
            relation = row_series.iloc[2]
            if relation != '=':
                n_non_equiv_skipped += 1
                continue
        
        n_equiv_candidates += 1
        
        try:
            src_e = OntologyEntryAttr(src_uri, OA_source)
            tgt_e = OntologyEntryAttr(tgt_uri, OA_target)
            base_key = src_uri + PAIRS_SEPARATOR + tgt_uri
            m_ask_oracle_user_prompts[base_key] = prompt_function(src_e, tgt_e)
            m_ask_oracle_user_prompts[base_key + PAIRS_SEPARATOR + "REVERSE"] = prompt_function(tgt_e, src_e)
        
        except (ClassNotFoundError, Exception) as e:
            skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": str(e)})
            tqdm.write(f"  WARNING: Skipping mapping - {e}")

    if n_non_equiv_skipped > 0:
        print(f"  Skipped {n_non_equiv_skipped} non-equivalence candidates.")

    if skipped_mappings:
        print(f"[WARNING] Skipped {len(skipped_mappings)} mappings due to errors.")

    print(f"  Built {n_equiv_candidates * 2} prompts ({n_equiv_candidates} forward + {n_equiv_candidates} reverse)")

    return m_ask_oracle_user_prompts, n_equiv_candidates, n_non_equiv_skipped

