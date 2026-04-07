from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from logmap_llm.constants import EntityType, PAIRS_SEPARATOR
from logmap_llm.ontology.entities import OntologyEntryAttr, ClassNotFoundError
from logmap_llm.oracle.prompts.formatting import (
    format_hierarchy,
    select_best_direct_entity_names,
    select_best_direct_entity_names_with_synonyms,
    select_best_sequential_hierarchy_with_synonyms,
)

from tqdm import tqdm

# from onto_object import OntologyEntryAttr

# from prompt_utils import (
#     format_hierarchy,
#     select_best_direct_entity_names,
#     select_best_direct_entity_names_with_synonyms,
#     select_best_sequential_hierarchy_with_synonyms,
# )


# TEMPLATE REGISTRY

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


#def prompt_direct_entity_ontological()
# def oupt_one_level_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
#     """Ontological prompt that uses ontology-focused language."""
#     src_parent, tgt_parent, src_entity_names, tgt_entity_names = select_best_direct_entity_names(src_entity, tgt_entity)

#     prompt_lines = [
#         "Analyze the following entities, each originating from a distinct biomedical ontology.",
#         "Your task is to assess whether they represent the **same ontological concept**, considering both their semantic meaning and hierarchical position.",
#         f'\n1. Source entity: "{src_entity_names}"',
#         f"\t- Direct ontological parent: {src_parent}",
#         f'\n2. Target entity: "{tgt_entity_names}"',
#         f"\t- Direct ontological parent: {tgt_parent}",
#         '\nAre these entities **ontologically equivalent** within their respective ontologies? Respond with "True" or "False".',
#     ]

#     return "\n".join(prompt_lines)

#def prompt_sequential_hierarchy_ontological()
# def oupt_two_levels_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
#     """Ontological prompt that uses ontology-focused language, and takes hierarchical relationships into account."""
#     src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2))
#     tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2))

#     prompt_lines = [
#         "Analyze the following entities, each originating from a distinct biomedical ontology.",
#         "Each is represented by its **ontological lineage**, capturing its hierarchical placement from the most general to the most specific level.",
#         f"\n1. Source entity ontological lineage:\n{src_hierarchy}",
#         f"\n2. Target entity ontological lineage:\n{tgt_hierarchy}",
#         '\nBased on their **ontological positioning, hierarchical relationships, and semantic alignment**, do these entities represent the **same ontological concept**? Respond with "True" or "False".',
#     ]
#     return "\n".join(prompt_lines)

#
# Natural language friendly prompts
#

#def prompt_direct_entity()
# def oupt_one_level_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
#     """Regular prompt that uses natural language and is more intuitive."""
#     src_parent, tgt_parent, src_entity_names, tgt_entity_names = select_best_direct_entity_names(src_entity, tgt_entity)
#     prompt_lines = [
#         "We have two entities from different biomedical ontologies.",
#         (
#             f'The first one is "{src_entity_names}"'
#             + (f', which belongs to the broader category "{src_parent}"' if src_parent else "")
#         ),
#         (
#             f'The second one is "{tgt_entity_names}"'
#             + (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else "")
#         ),
#         '\nDo they mean the same thing? Respond with "True" or "False".',
#     ]
#     return "\n".join(prompt_lines)


#def prompt_sequential_hierarchy()
# def oupt_two_levels_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
#     """Regular prompt that uses natural language and is more intuitive."""
#     src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
#     tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)

#     prompt_lines = [
#         "We have two entities from different biomedical ontologies.",
#         (
#             f'The first one is "{src_hierarchy[0]}"'
#             + (f', which belongs to the broader category "{src_hierarchy[1]}"' if len(src_hierarchy) > 1 else "")
#             + (f', under the even broader category "{src_hierarchy[2]}"' if len(src_hierarchy) > 2 else "")
#         ),
#         (
#             f'The second one is "{tgt_hierarchy[0]}"'
#             + (f', which belongs to the broader category "{tgt_hierarchy[1]}"' if len(tgt_hierarchy) > 1 else "")
#             + (f', under the even broader category "{tgt_hierarchy[2]}"' if len(tgt_hierarchy) > 2 else "")
#         ),
#         '\nDo they mean the same thing? Respond with "True" or "False".',
#     ]

#     return "\n".join(prompt_lines)

#
# Natural language prompts with synonyms
#

#def prompt_direct_entity_with_synonyms()
# def oupt_one_level_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
#     """Natural language prompt that includes synonyms for a more intuitive comparison."""
#     src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = (
#         select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
#     )

#     src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""

#     tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
#     prompt_lines = [
#         "We have two entities from different biomedical ontologies.",
#         f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
#         f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
#         '\nDo they mean the same thing? Respond with "True" or "False".',
#     ]
#     return "\n".join(prompt_lines)

#def prompt_sequential_hierarchy_with_synonyms()
# def oupt_two_levels_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr) -> str:
#     """Generate a natural language prompt asking whether two ontology entities (with synonyms and hierarchy).

#     Represent the same concept (True/False).
#     """
#     src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
#     tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)

#     src_syns, tgt_syns, src_parents_syns, tgt_parents_syns = select_best_sequential_hierarchy_with_synonyms(
#         src_entity, tgt_entity, max_level=2
#     )

#     def describe_entity(hierarchy: list[str], entity_syns: list[str], parent_syns: list[list[str]]) -> str:
#         # Base name and its synonyms
#         name_part = f'"{hierarchy[0]}"'
#         if entity_syns:
#             alt = ", ".join(f'"{s}"' for s in entity_syns)
#             name_part += f", also known as {alt}"

#         parts = [name_part]

#         labels = ["belongs to broader category", "under the even broader category", "under the even broader category"]
#         for i, parent_name in enumerate(hierarchy[1:]):
#             text = f'{labels[i]} "{parent_name}"'
#             if parent_syns[i]:
#                 alt = ", ".join(f'"{s}"' for s in parent_syns[i])
#                 text += f" (also known as {alt})"
#             parts.append(text)

#         return ", ".join(parts)

#     src_desc = describe_entity(src_hierarchy, src_syns, src_parents_syns)
#     tgt_desc = describe_entity(tgt_hierarchy, tgt_syns, tgt_parents_syns)

#     prompt_lines = [
#         "We have two entities from different biomedical ontologies.",
#         f"The first one is {src_desc}.",
#         f"The second one is {tgt_desc}.",
#         '\nDo they mean the same thing? Respond with "True" or "False".',
#     ]
#     return "\n".join(prompt_lines)

@registry.register("one_level_of_parents_structured")
def oupt_one_level_of_parents_structured(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> str:
    """Ontological prompt using ontology-focused language."""
    (src_parent, tgt_parent,
     src_entity_names, tgt_entity_names) = select_best_direct_entity_names(
        src_entity, tgt_entity
    )
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
def oupt_two_levels_of_parents_structured(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> str:
    """Ontological prompt with hierarchical lineage."""
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


@registry.register("one_level_of_parents")
def oupt_one_level_of_parents(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> str:
    """Natural language prompt with one level of parents."""
    (src_parent, tgt_parent,
     src_entity_names, tgt_entity_names) = select_best_direct_entity_names(
        src_entity, tgt_entity
    )
    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        (
            f'The first one is "{src_entity_names}"'
            + (f', which belongs to the broader category "{src_parent}"'
               if src_parent else "")
        ),
        (
            f'The second one is "{tgt_entity_names}"'
            + (f', which belongs to the broader category "{tgt_parent}"'
               if tgt_parent else "")
        ),
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)


@registry.register("two_levels_of_parents")
def oupt_two_levels_of_parents(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> str:
    """Natural language prompt with two levels of parents."""
    src_hierarchy = format_hierarchy(
        src_entity.get_parents_by_levels(max_level=2), True
    )
    tgt_hierarchy = format_hierarchy(
        tgt_entity.get_parents_by_levels(max_level=2), True
    )
    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        (
            f'The first one is "{src_hierarchy[0]}"'
            + (f', which belongs to the broader category "{src_hierarchy[1]}"'
               if len(src_hierarchy) > 1 else "")
            + (f', under the even broader category "{src_hierarchy[2]}"'
               if len(src_hierarchy) > 2 else "")
        ),
        (
            f'The second one is "{tgt_hierarchy[0]}"'
            + (f', which belongs to the broader category "{tgt_hierarchy[1]}"'
               if len(tgt_hierarchy) > 1 else "")
            + (f', under the even broader category "{tgt_hierarchy[2]}"'
               if len(tgt_hierarchy) > 2 else "")
        ),
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)


@registry.register("one_level_of_parents_and_synonyms")
def oupt_one_level_of_parents_and_synonyms(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> str:
    """Natural language prompt with synonyms and one level of parents."""
    (src_parent, tgt_parent,
     src_entity_names, tgt_entity_names,
     src_synonyms, tgt_synonyms) = select_best_direct_entity_names_with_synonyms(
        src_entity, tgt_entity
    )

    src_synonyms_text = (
        ", also known as " + ", ".join(f'"{s}"' for s in sorted(src_synonyms))
    ) if src_synonyms else ""

    tgt_synonyms_text = (
        ", also known as " + ", ".join(f'"{s}"' for s in sorted(tgt_synonyms))
    ) if tgt_synonyms else ""

    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, '
        f'which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, '
        f'which falls under the category "{tgt_parent}".',
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)


@registry.register("two_levels_of_parents_and_synonyms")
def oupt_two_levels_of_parents_and_synonyms(
    src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr
) -> str:
    """Natural language prompt with synonyms and two levels of parents."""
    src_hierarchy = format_hierarchy(
        src_entity.get_parents_by_levels(max_level=2), True
    )
    tgt_hierarchy = format_hierarchy(
        tgt_entity.get_parents_by_levels(max_level=2), True
    )

    (src_syns, tgt_syns,
     src_parents_syns, tgt_parents_syns) = select_best_sequential_hierarchy_with_synonyms(
        src_entity, tgt_entity, max_level=2
    )

    def describe_entity(
        hierarchy: list[str],
        entity_syns: list[str],
        parent_syns: list[list[str]],
    ) -> str:
        name_part = f'"{hierarchy[0]}"'
        if entity_syns:
            alt = ", ".join(f'"{s}"' for s in sorted(entity_syns))
            name_part += f", also known as {alt}"

        parts = [name_part]

        labels = [
            "belongs to broader category",
            "under the even broader category",
            "under the even broader category",
        ]
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
        "We have two entities from different biomedical ontologies.",
        f"The first one is {src_desc}.",
        f"The second one is {tgt_desc}.",
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)

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
    m_ask_df,
    OA_source=None,
    OA_target=None,
    sibling_selector=None,
    property_prompt_function=None,
    instance_prompt_function=None,
) -> dict:
    """Build oracle user prompts for all mappings in M_ask.

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
        Template function for property entities (Branch 6).
    instance_prompt_function : callable, optional
        Template function for instance entities (Branch 6).

    Returns
    -------
    dict
        Mapping of 'src_uri|tgt_uri' -> prompt string.
    """
    from logmap_llm.ontology.access import OntologyAccess

    # Load ontologies if not provided
    if OA_source is None or OA_target is None:
        for onto_path in tqdm(iterable=[onto_src_filepath],
                              desc='Preparing source ontology'):
            OA_source = OntologyAccess(onto_path, annotate_on_init=True)
        print()
        for onto_path in tqdm(iterable=[onto_tgt_filepath],
                              desc='Preparing target ontology'):
            OA_target = OntologyAccess(onto_path, annotate_on_init=True)

    # Get the template function
    prompt_function = get_oracle_user_prompt_template_function(oupt_name)
    print()
    print(f"Prompt template function obtained: {oupt_name}")
    print()

    m_ask_oracle_user_prompts = {}
    skipped_mappings = []

    for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0],
                    desc="Building the prompts"):
        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]

        try:
            src_entity_onto_attrs = OntologyEntryAttr(
                src_entity_uri, OA_source
            )
            tgt_entity_onto_attrs = OntologyEntryAttr(
                tgt_entity_uri, OA_target
            )

            oracle_user_prompt = prompt_function(
                src_entity_onto_attrs, tgt_entity_onto_attrs
            )

            key = src_entity_uri + PAIRS_SEPARATOR + tgt_entity_uri
            m_ask_oracle_user_prompts[key] = oracle_user_prompt

        except ClassNotFoundError as e:
            skipped_mappings.append({
                "src": src_entity_uri,
                "tgt": tgt_entity_uri,
                "reason": str(e),
            })
            tqdm.write(f"  WARNING: Skipping mapping — {e}")

    if skipped_mappings:
        print(f"[WARNING] Skipped {len(skipped_mappings)} mappings.")
    else:
        print(f"All {m_ask_df.shape[0]} mappings resolved successfully.")

    return m_ask_oracle_user_prompts



# from onto_access import OntologyAccess
# from onto_object import OntologyEntryAttr
# from tqdm import tqdm
# from constants import PAIRS_SEPARATOR

# from oracle_user_prompt_templates import (
#     oupt_one_level_of_parents_structured,
#     oupt_two_levels_of_parents_structured,
#     oupt_one_level_of_parents,
#     oupt_two_levels_of_parents,
#     oupt_one_level_of_parents_and_synonyms,
#     oupt_two_levels_of_parents_and_synonyms
# )

# oupt_templates_2_oupt_functions = {
#     prompt_function.__name__.replace("oupt_", ""): prompt_function
#     for prompt_function in [
#         oupt_one_level_of_parents_structured,
#         oupt_two_levels_of_parents_structured,
#         oupt_one_level_of_parents,
#         oupt_two_levels_of_parents,
#         oupt_one_level_of_parents_and_synonyms,
#         oupt_two_levels_of_parents_and_synonyms
#     ]
# }


# def get_oracle_user_prompt_template_function(oupt_name):

#     return oupt_templates_2_oupt_functions[oupt_name]



# def build_oracle_user_prompts(oupt_name, onto_src_filepath, 
#                               onto_tgt_filepath, m_ask_df):
#     '''
#     Build oracle user prompts using a particular template.
    
#     Parameters
#     ----------
#     oupt_name : string
#         the name of an oracle user prompt template
#     onto_src_filepath : string
#         an absolute path to an ontology file
#     onto_tgt_filepath : string
#         an absolute path to an ontology file
#     m_ask_df : pandas DataFrame
#         a Python representation of m_ask from a LogMap alignment
#         - the representation is an in-memory replica of LogMap's output
#           file: 'logmap_mappings_to_ask_oracle_user_llm.txt'
    
#     Returns
#     -------
#     m_ask_oracle_user_prompts : dictionary
#         the oracle user prompts built for each mapping in m_ask_df
#         key: string 
#             a mapping identifier ('src_entity_uri|tgt_entity_uri')
#         value: string 
#             a formatted string representing a prepared oracle user prompt
#     '''
    
#     try:
#         # instantiate an OntologyAccess object for the source ontology
#         for onto_path in tqdm(iterable=[onto_src_filepath],
#                             desc='Preparing source ontology'):
#             OA_source_onto = OntologyAccess(onto_path, annotate_on_init=True)
        
#         print()

#         # instantiate an OntologyAccess object for the target ontology
#         for onto_path in tqdm(iterable=[onto_tgt_filepath],
#                             desc='Preparing target ontology'):
#             OA_target_onto = OntologyAccess(onto_path, annotate_on_init=True)
#     except Exception as e:
#         raise e
    
#     # get the function for the specified oracle 'user' prompt template
#     prompt_function = get_oracle_user_prompt_template_function(oupt_name)
#     print()
#     print(f"Prompt template function obtained: {oupt_name}")
#     print()
    
#     # initialise a container for the prompts
#     m_ask_oracle_user_prompts = {}

#     # iterate over the mappings in m_ask
#     # (each 'row' is an (index, Series) tuple)
#     for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0], 
#                     desc="Building the prompts"):
        
#         # get the URIs of the two entities involved in the current mapping
#         row_series = row[1]
#         src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]

#         # get attributes of the ontological neighbourhood of the source 
#         # and target entities, subsets of which are likely fillers for the
#         # prompt template being used to build the user prompts
#         src_entity_onto_attrs = OntologyEntryAttr(src_entity_uri, OA_source_onto)
#         tgt_entity_onto_attrs = OntologyEntryAttr(tgt_entity_uri, OA_target_onto)

#         # build the oracle user prompt for the current mapping
#         oracle_user_prompt = prompt_function(src_entity_onto_attrs, 
#                                              tgt_entity_onto_attrs)
        
#         # store the oracle user prompt for the current mapping
#         key = src_entity_uri + PAIRS_SEPARATOR + tgt_entity_uri 
#         m_ask_oracle_user_prompts[key] = oracle_user_prompt
    
#     return m_ask_oracle_user_prompts

