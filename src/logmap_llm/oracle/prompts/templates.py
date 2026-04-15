"""
logmap_llm.oracle.prompts.templates
Provides the prompt template functions and registry + build_oracle_user_prompts() 
and build_oracle_user_prompts_bidirectional() (orchestration functions)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

from logmap_llm.ontology.object import OntologyEntryAttr, ClassNotFoundError, resolve_entity
from logmap_llm.ontology.sibling_retrieval import SiblingSelector, _get_label
from logmap_llm.ontology.access import OntologyAccess
from logmap_llm.ontology.cache import OntologyCache
from logmap_llm.utils.logging import step, success
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
from logmap_llm.constants import (
    EntityType,
    PAIRS_SEPARATOR,
    RESPONSE_INSTRUCTION,
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_RESPONSE_MODE,
    ANSWER_FORMATS,
    RESPONSE_MODES,
    VERBOSE,
)


from tqdm import tqdm

###
# MODULE-LEVEL ANSWER FORMAT STATE
# --------------------------------
# NOTE: see truth table in constants.py for combinations of:
#   'true_false' | 'yes_no' & 'structured' | 'plain'
# and how these dictate the response instruction that is injected
# into the LLM prompts for consultation.
###

# _answer_format : default: 'true_false'
_answer_format: str = DEFAULT_ANSWER_FORMAT

# _response_mode : default: 'structured' 
_response_mode: str = DEFAULT_RESPONSE_MODE

# _response_instruction : default: 
#   "Respond with a JSON object: {"answer": true} or {"answer": false}."
_response_instruction: str = RESPONSE_INSTRUCTION[
    (DEFAULT_ANSWER_FORMAT, DEFAULT_RESPONSE_MODE)
] 

def set_response_config(answer_format: str, response_mode: str) -> None:
    """
    Set the answer format and response mode for prompt rendering.
    Both must be set together because the response instruction text
    depends on the (answer_format, response_mode) pair.
    """
    global _answer_format, _response_mode, _response_instruction
    
    if answer_format not in ANSWER_FORMATS:
        raise ValueError(f"Unknown answer_format '{answer_format}'. Valid options: {sorted(ANSWER_FORMATS)}")
    
    if response_mode not in RESPONSE_MODES:
        raise ValueError(f"Unknown response_mode '{response_mode}'. Valid options: {sorted(RESPONSE_MODES)}")
    
    _answer_format = answer_format
    _response_mode = response_mode
    
    _response_instruction = RESPONSE_INSTRUCTION[
        (answer_format, response_mode)
    ]


def get_answer_format() -> str:
    return _answer_format


def get_response_mode() -> str:
    return _response_mode


def get_response_instruction() -> str:
    return _response_instruction


###
# MODULE-LEVEL DOMAIN STATE
###

_ontology_domain: str | None = None

def set_ontology_domain(domain: str | None) -> None:
    global _ontology_domain
    _ontology_domain = domain


def get_ontology_domain() -> str | None:
    return _ontology_domain


def get_domain_preamble() -> str:
    if _ontology_domain:
        return (f"We have two entities from different {get_ontology_domain()} ontologies.")
    return "We have two entities from different ontologies."


def force_ontology_domain_string() -> str:
    """
    Some legacy prompts contain a substring: 'two biomedical ontologies'.
    In some of these cases, the prompt differs from the expected domain preamble.
    For these cases, we apply a patch that 'forces' a domain string, even if one
    has not been set (in which case it will simply be empty). The purpose of this
    function is to ensure the prompts use the neccesary string spacing, etc.
    """
    if _ontology_domain:
        return str(f" {get_ontology_domain()} ")
    # else:
    return " "


###
# MODULE-LEVEL FORMATTING FOR INSTANCE PROMPTS
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
    fn: Callable
    entity_type: EntityType = EntityType.CLASS
    bidirectional: bool = False
    requires_siblings: bool = False


class TemplateRegistry:
    '''
    The TemplateRegistry defines a container for which template functions (i.e., prompts) can be bound and resolved.
    We provide a decorator so that any defined prompts can be convienently bound to the TemplateRegistry container.
    As of March 2026, we also support features such as 'equivalence by mutual subsumption' where:

        A \sqsubseteq B \land B \sqsubseteq A \iff A \equiv B

    As well as class, property and instance-based matching; and the use of contrastive examples during few-shot prompting.
    When registering a prompt template, you can specify whether you would like these features to be used via the register
    function (or through the decorator), by specifying:

        1. Which entity category your prompt belongs to, i.e.,
           EntityType = EntityType.CLASS | EntityType.OBJECTPROPERTY | EntityType.DATAPROPERTY | EntityType.INSTANCE
           TODO (April 10th 2026): Implement distinct behaviours for processing EntityType.OBJECTPROPERTY and EntityType.DATAPROPERTY
           in case users would like to treat them differently.
           (defaults to EntityType = EntityType.CLASS)

        2. Whether you would like the prompt to be evaluated under the 'equivalence by mutual subsumption' behaviour.
           IMPORTANT: This will double the prompt count for any experimental runs, since each prompt has to be sent twice
           with the src and tgt entities reversed. You can enable this by specifying: `bidirectional = True`.
           (defaults to `bidirectional = False`)

        3. If a prompt requires siblings (i.e., `requires_siblings=True`), it enables the following optional behaviours:
            * fetch siblings, order them alphanumerically by label/preferred term, then select the top-k siblings.
            * embed siblings (via SapBERT or Sentence-Transformers Models [configurable]), fetch the most semantically similar
              by cosine similarity and take the top-k siblings.
            * TODO: fetch siblings by LM-based ontology embedding, allowing for apporximate reasoning in the learnt representation
              space. Hierarchy Transformers allows for subsumption-based reasoning, whereas Ontology Tranformers also allow
              for approximate inferrence \w roles. (NOT YET IMPLEMENTED, April 10th 2026).
          (defaults to `require_siblings = False).
    
    Please refer to project-specific documentation for further guidance on use of these features (TODO).
    '''
    def __init__(self):
        self._templates: dict[str, TemplateSpec] = {}

    # decorator (helper):
    def register(self, name: str, entity_type: EntityType = EntityType.CLASS, bidirectional: bool = False, requires_siblings: bool = False) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._templates[name] = TemplateSpec(
                fn=fn,
                entity_type=entity_type,
                bidirectional=bidirectional,
                requires_siblings=requires_siblings
            )
            return fn
        return decorator

    # bind:
    def register_fn(self, name: str, fn: Callable, entity_type: EntityType = EntityType.CLASS, bidirectional: bool = False, requires_siblings: bool = False) -> None:
        self._templates[name] = TemplateSpec(
            fn=fn, 
            entity_type=entity_type,
            bidirectional=bidirectional,
            requires_siblings=requires_siblings
        )

    # resolve:
    def get(self, name: str) -> TemplateSpec:
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self._templates.keys())}")
        return self._templates[name]

    ###
    # callable helpers:
    ###

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


registry = TemplateRegistry()


###
# HELPERS (FUNCTIONS)
###

def _retrieve_siblings(entity, sibling_selector: SiblingSelector=None, max_count=2):
    if sibling_selector is not None:
        # map sibling_selector.select_siblings(...) -> preferred labels 
        # ... (required by prompt template consumers; also list comprehension is more pythonic!)
        ranked = sibling_selector.select_siblings(entity, max_count=max_count)
        return [
            _get_label(sibling) for sibling, _score in ranked
        ]
    # else:
    sib_entries = entity.get_siblings(max_count=max_count)
    return [
        (
            min(sibling.get_preferred_names()) if sibling.get_preferred_names() 
            else str(sibling.thing_class.name), 1.0
        )
        for sibling in sib_entries
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


###
# LEGACY TEMPLATES
# NOTE: these are not actively used and are currently untested.
# DATE: April 10th 2026.
###

@registry.register("all_data_dummy")
def prompt_all_data_dummy(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    return f"""
    **Task Description:**
    Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

    1. **Source Entity:**
    **All Entity names:** {src_entity.get_preffered_names()}
    **Parent Entity Namings:** {src_entity.get_parents_preferred_names()}
    **Child Entity Namings:** {src_entity.get_children_preferred_names()}

    2. **Target Entity:**
    **All Entity names:** {tgt_entity.get_preffered_names()}
    **Parent Entity Namings:** {tgt_entity.get_parents_preferred_names()}
    **Child Entity Namings:** {tgt_entity.get_children_preferred_names()}

    Write "Yes" if the entities refer to the same concepts, and "No" otherwise.
    """.strip()


@registry.register("only_names")
def prompt_only_names(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entity.get_all_entity_names()}

    2. Target Entity:
    All Entity names: {tgt_entity.get_all_entity_names()}

    Response with True or False
    """.strip()


@registry.register("with_hierarchy")
def prompt_with_hierarchy(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entity.get_all_entity_names()}
    Parent Entity Namings: {src_entity.get_parents_preferred_names()}
    Child Entity Namings: {src_entity.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entity.get_all_entity_names()}
    Parent Entity Namings: {tgt_entity.get_parents_preferred_names()}
    Child Entity Namings: {tgt_entity.get_children_preferred_names()}

    Response with True or False
    """.strip()


@registry.register("only_with_parents")
def prompt_only_with_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names and parent relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entity.get_all_entity_names()}
    Parent Entity Namings: {src_entity.get_parents_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entity.get_all_entity_names()}
    Parent Entity Namings: {tgt_entity.get_parents_preferred_names()}

    Response with True or False
    """.strip()


@registry.register("only_with_children")
def prompt_only_with_children(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    return f"""
    Given two entities from different ontologies with their names and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {src_entity.get_all_entity_names()}
    Child Entity Namings: {src_entity.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {tgt_entity.get_all_entity_names()}
    Child Entity Namings: {tgt_entity.get_children_preferred_names()}

    Response with True or False
    """.strip()



###
# ORIGINAL CLASS EQUIVALENCE TEMPLATES
###



@registry.register("one_level_of_parents_structured")
def oupt_one_level_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    Previously known as `prompt_direct_entity_ontological`.
    Ontological prompt that uses ontology-focused language.
    """
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names) = select_best_direct_entity_names(src_entity, tgt_entity)
    
    prompt_lines = [
        f"Analyze the following entities, each originating from a distinct{force_ontology_domain_string()}ontology.",
        "Your task is to assess whether they represent the **same ontological concept**, considering both their semantic meaning and hierarchical position.",
        f'\n1. Source entity: "{src_entity_names}"',
        f"\t- Direct ontological parent: {src_parent}",
        f'\n2. Target entity: "{tgt_entity_names}"',
        f"\t- Direct ontological parent: {tgt_parent}",
        f'\nAre these entities **ontologically equivalent** within their respective ontologies? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents_structured")
def oupt_two_levels_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    Previously known as `prompt_sequential_hierarchy_ontological`.
    Ontological prompt that uses ontology-focused language, and takes hierarchical relationships into account.
    """
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2))
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2))
    
    prompt_lines = [
        f"Analyze the following entities, each originating from a distinct{force_ontology_domain_string()}ontology.",
        "Each is represented by its **ontological lineage**, capturing its hierarchical placement from the most general to the most specific level.",
        f"\n1. Source entity ontological lineage:\n{src_hierarchy}",
        f"\n2. Target entity ontological lineage:\n{tgt_hierarchy}",
        f'\nBased on their **ontological positioning, hierarchical relationships, and semantic alignment**, do these entities represent the **same ontological concept**? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



@registry.register("one_level_of_parents")
def oupt_one_level_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    Previously known as `prompt_direct_entity`.
    """
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names) = select_best_direct_entity_names(src_entity, tgt_entity)
    
    prompt_lines = [
        get_domain_preamble(),
        (f'The first one is "{src_entity_names}"' + (f', which belongs to the broader category "{src_parent}"' if src_parent else "")),
        (f'The second one is "{tgt_entity_names}"' + (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else "")),
        (f'\nDo they mean the same thing? {_response_instruction}'),
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents")
def oupt_two_levels_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    Previously known as `prompt_sequential_hierarchy`.
    """
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)
    
    prompt_lines = [
        get_domain_preamble(),
        (f'The first one is "{src_hierarchy[0]}"' + (f', which belongs to the broader category "{src_hierarchy[1]}"' if len(src_hierarchy) > 1 else "") + (f', under the even broader category "{src_hierarchy[2]}"' if len(src_hierarchy) > 2 else "")),
        (f'The second one is "{tgt_hierarchy[0]}"' + (f', which belongs to the broader category "{tgt_hierarchy[1]}"' if len(tgt_hierarchy) > 1 else "") + (f', under the even broader category "{tgt_hierarchy[2]}"' if len(tgt_hierarchy) > 2 else "")),
        (f'\nDo they mean the same thing? {_response_instruction}'),
    ]
    return "\n".join(prompt_lines)



@registry.register("one_level_of_parents_and_synonyms")
def oupt_one_level_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    Previously known as `prompt_direct_entity_with_synonyms`.
    Natural language prompt that includes synonyms for a more intuitive comparison.
    """
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms) = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in sorted(src_synonyms))) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in sorted(tgt_synonyms))) if tgt_synonyms else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    prompt_lines = [
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents_and_synonyms")
def oupt_two_levels_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
    """
    Previously known as `prompt_sequential_hierarchy_with_synonyms`.
    Generate a natural language prompt asking whether two ontology entities (with synonyms and hierarchy). Represent the same concept (True/False).
    """
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
        get_domain_preamble(),
        f"The first one is {src_desc}.",
        f"The second one is {tgt_desc}.",
        f'\nDo they mean the same thing? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



###
# NEW CLASS-BASED PROMPTS
# (only the addition of a `synonyms_only` prompt, which aligns to the 'new prompts' in style).
###



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
# SUBSUMPTION TEMPLATES
# (ie. bidirectional=True)
###

### TODO: update the name for this; its not only subs_labels.
### it also contains synonyms (labels + synonyms).
@registry.register("sub_labels_only", bidirectional=True)
def oupt_sub_labels_only(src_entity, tgt_entity):
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"? ',
        f'\n{_response_instruction}',
    ])



@registry.register("sub_parents_synonyms", bidirectional=True)
def oupt_sub_parents_synonyms(src_entity, tgt_entity):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"? ',
        f'\n{_response_instruction}',
    ])



### TODO: SCHEDULED FOR REMOVAL
@registry.register("sub_syns_conj_parent", bidirectional=True)
def oupt_sub_syns_conj_parent(src_entity, tgt_entity):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"{tgt_qual}? ',
        f'\n{_response_instruction}',
    ])



### TODO: SCHEDULED FOR REMOVAL
@registry.register("single_subs")
def oupt_single_subs(src_entity, tgt_entity):
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}".',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}".',
        f'\nAre both statements true? {_response_instruction}',
    ])



### TODO: SCHEDULED FOR REMOVAL
@registry.register("single_subs_with_parents")
def oupt_single_subs_with_parents(src_entity, tgt_entity):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}".',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}".',
        f'\nAre both statements true? {_response_instruction}',
    ])



### TODO: SCHEDULED FOR REMOVAL
@registry.register("single_subs_with_conj_parent")
def oupt_single_subs_with_conj_parent(src_entity, tgt_entity):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""
    
    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}"{tgt_qual}.',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}"{src_qual}.',
        f'\nAre both statements true? {_response_instruction}',
    ])



### TODO: SCHEDULED FOR REMOVAL
@registry.register("equiv_test_for_ancestral_disjointness", bidirectional=True)
def oupt_equiv_test_for_ancestral_disjointness(src_entity, tgt_entity):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nCan something be a "{src_entity_names}"{src_qual} at the same time as being a "{tgt_entity_names}"{tgt_qual}?',
        f'\n{_response_instruction}',
    ])



@registry.register("equiv_with_ancestral_disjointness", bidirectional=True)
def oupt_equiv_with_ancestral_disjointness(src_entity, tgt_entity):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nIf something is a "{src_entity_names}"{src_qual}, is it also a "{tgt_entity_names}"{tgt_qual}?',
        f'\n{_response_instruction}',
    ])



@registry.register("deductive_equiv")
def oupt_deductive_equiv(src_entity, tgt_entity):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    facts = []
    facts.append(f'- "{src_entity_names}" is a kind of "{src_parent}" (Ontology 1)' if src_parent else f'- "{src_entity_names}" is a concept in Ontology 1')
    
    if src_synonyms:
        facts.append(f'- "{src_entity_names}" is also known as ' + " and ".join(f'"{s}"' for s in src_synonyms) + ' (Ontology 1)')
    
    facts.append(f'- "{tgt_entity_names}" is a kind of "{tgt_parent}" (Ontology 2)' if tgt_parent else f'- "{tgt_entity_names}" is a concept in Ontology 2')
    
    if tgt_synonyms:
        facts.append(f'- "{tgt_entity_names}" is also known as ' + " and ".join(f'"{s}"' for s in tgt_synonyms) + ' (Ontology 2)')
    
    return "\n".join([
        f"We have two entities from different {_ontology_domain} ontologies. Consider the following facts:",
        "\n".join(facts),
        f'\nGiven these facts, can you conclude that "{src_entity_names}" (Ontology 1) and "{tgt_entity_names}" (Ontology 2) refer to the same real-world concept? {_response_instruction}',
    ])



### TODO: SCHEDULED FOR REMOVAL
@registry.register("falsification_equiv", requires_siblings=True)
def oupt_falsification_equiv(src_entity, tgt_entity, sibling_selector=None):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""
    
    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""
    
    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.{tgt_sibling_clause}',
        f'\nFirst consider: could something classified as a "{src_parent}" in one ontology plausibly also be classified as a "{tgt_parent}" in another? If the categories suggest entirely different domains, the concepts are unlikely to match. If the categories are compatible or overlapping, the concepts may well match.',
        f'\nDo "{src_entity_names}" (a "{src_parent}") and "{tgt_entity_names}" (a "{tgt_parent}") refer to the same concept? {_response_instruction}',
    ])



###
# SIBLING-AWARE PROMPT TEMPLATES
###



@registry.register("equiv_parents_siblings", requires_siblings=True)
def oupt_equiv_parents_siblings(src_entity, tgt_entity, sibling_selector=None):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, _, _ = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""
    
    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_parent_clause}.{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_parent_clause}.{tgt_sibling_clause}',
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

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""
    
    return "\n".join([
        get_domain_preamble(),
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.{tgt_sibling_clause}',
        f'\nDo they mean the same thing? {_response_instruction}',
    ])



###
# PROPERTY TEMPLATES (OBJECTPROPERTY) -- EntityType.OBJECTPROPERTY
#
# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 
###


@registry.register("prop_labels_only", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_labels_only(src_prop, tgt_prop):
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)
    prompt_lines = [
        f'We have two properties from different ontologies.',
        f'The first property is "{src_name}".',
        f'The second property is "{tgt_name}".',
        f'Do these properties represent the same relationship? {_response_instruction}',
    ]
    return "\n".join(prompt_lines)



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

@registry.register("prop_domain_range", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_domain_range(src_prop, tgt_prop):
    src_name = get_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)
    src_desc = format_domain_range_clause(src_name, src_prop.get_domain_names(), src_prop.get_range_names())
    tgt_desc = format_domain_range_clause(tgt_name, tgt_prop.get_domain_names(), tgt_prop.get_range_names())
    return f'We have two properties from different ontologies.\n\nThe first property is {src_desc}.\n\nThe second property is {tgt_desc}.\n\nDo these properties represent the same relationship?\n\n{_response_instruction}'



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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
# CONFERENCE-INSPIRED ENRICHED TEMPLATES
###



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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
# INSTANCE TEMPLATES -- EntityType.INSTANCE
###



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

@registry.register("inst_labels_only", entity_type=EntityType.INSTANCE)
def oupt_inst_labels_only(src_inst, tgt_inst):
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    return f'We have two entities from different knowledge graphs.\n\nThe first is "{src_label}".\n\nThe second is "{tgt_label}".\n\nDo these refer to the same entity?\n\n{_response_instruction}'



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

@registry.register("inst_labels_with_types", entity_type=EntityType.INSTANCE)
def oupt_inst_labels_with_types(src_inst, tgt_inst):
    src_label = get_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "")
    return f'We have two entities from different knowledge graphs.\n\nThe first is {src_desc}.\n\nThe second is {tgt_desc}.\n\nDo these refer to the same entity?\n\n{_response_instruction}'



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!
## TODO (2): USE --> get_domain_preamble(), <-- 

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



# TODO: !! UPDATE THE READABILITY OF THESE TEMPLATES !!

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



###
# REGISTRY FUNCTION
###

def get_oracle_user_prompt_template_function(oupt_name: str) -> Callable:
    return registry.get(oupt_name).fn



###
# PROMPT-BUILDING ORCHESTRATION
###

# TODO: rework mess
def build_oracle_user_prompts(oupt_name, onto_src_filepath, onto_tgt_filepath, m_ask_df,
                               OA_source=None, OA_target=None, sibling_selector=None,
                               property_prompt_function=None, instance_prompt_function=None,
                               property_prompt_name=None, instance_prompt_name=None):

    if OA_source is None or OA_target is None:
        for p in tqdm(iterable=[onto_src_filepath], desc='Preparing source ontology'):
            OA_source = OntologyAccess(p, annotate_on_init=True)
        print()
        for p in tqdm(iterable=[onto_tgt_filepath], desc='Preparing target ontology'):
            OA_target = OntologyAccess(p, annotate_on_init=True)

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

    print(f"\nPrompt template function obtained: {oupt_name}")
    
    if property_prompt_function: 
        print(f"Property prompt template: {property_prompt_name}")
    
    if instance_prompt_function:
        print(f"Instance prompt template: {instance_prompt_name}")

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
                oracle_user_prompt = prompt_function(src_entity, tgt_entity)
                n_class += 1
            m_ask_oracle_user_prompts[src_uri + PAIRS_SEPARATOR + tgt_uri] = oracle_user_prompt
        except (ClassNotFoundError, Exception) as e:
            skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": str(e)})
            tqdm.write(f"  WARNING: Skipping mapping - {e}")

    if skipped_mappings: 
        print(f"[WARNING] Skipped {len(skipped_mappings)} mappings.")

    print(f"Prompts built: {n_class} cls, {n_prop} prop, {n_inst} inst ({len(m_ask_oracle_user_prompts)} total)")
    
    return m_ask_oracle_user_prompts


# TODO: rework mess
def build_oracle_user_prompts_bidirectional(oupt_name, onto_src_filepath, onto_tgt_filepath,
                                             m_ask_df, OA_source=None, OA_target=None,
                                             sibling_selector=None):

    if OA_source is None or OA_target is None:
        for p in tqdm(iterable=[onto_src_filepath], desc='Preparing source ontology'):
            OA_source = OntologyAccess(p, annotate_on_init=True)
        print()
        for p in tqdm(iterable=[onto_tgt_filepath], desc='Preparing target ontology'):
            OA_target = OntologyAccess(p, annotate_on_init=True)

    prompt_function = get_oracle_user_prompt_template_function(oupt_name)

    if sibling_selector is not None and registry.requires_siblings(oupt_name):
        prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

    print(f"Prompt template function obtained: {oupt_name} (bidirectional mode)")

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



###
# MISC FUNCS
###

# TODO: migrate to preferable location (isn't this an older legacy fn?)
def load_ontologies(onto_src_filepath, onto_tgt_filepath, cache_dir=None):
    '''
    Load and return OntologyAccess objects for source and target ontologies.
    NOTE: This appears to be in the wrong place; migrate to more appropriate module.
          Also, this is a legacy function, this is from an early local version of the codebase.
          However, step_two.py currently relies on this... (TODO: fix)
    '''
    # wrap the raw cache_dir path into an OntologyCache instance so
    # that OntologyAccess can call cache.get_cached_world(urionto)
    # the same instance is reused for both ontologies; cache files
    # are keyed by hashed source filepath inside cache.py
    cache = OntologyCache(cache_dir=cache_dir) if cache_dir is not None else None

    def _print_padded_callback(fn: Callable, **kwargs):
        print()
        fn(**kwargs)
        print()

    def _print_load_message(onto_str: str):
        step("-" * 50)
        step(f"{'-' * 15}LOADING ONTOLOGY {onto_str}{'-' * 15}")
        step("-" * 50)

    def _print_done_message():
        success("-" * 50)
        success(f"{'-' * 23}DONE{'-' * 23}")
        success("-" * 50)

    _print_padded_callback(_print_load_message, onto_str="ONE")

    OA_source_onto = OntologyAccess(onto_src_filepath, annotate_on_init=True, cache=cache)
    
    _print_padded_callback(_print_done_message)
    _print_padded_callback(_print_load_message, onto_str="TWO")

    OA_target_onto = OntologyAccess(onto_tgt_filepath, annotate_on_init=True, cache=cache)
    
    _print_padded_callback(_print_done_message)

    return OA_source_onto, OA_target_onto