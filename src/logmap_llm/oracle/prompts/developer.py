"""
logmap_llm.oracle.prompts.developer — Developer (system) prompt registry.

Contains all developer prompt constants and the registry that maps
config-friendly names to prompt text.

IMPORT CONSTRAINT: This module must NEVER import from logmap_llm.pipeline.*
to prevent circular imports.
"""
from __future__ import annotations


###
# DEVELOPER (SYSTEM) PROMPTS
###

DEV_PROMPT_GENERIC = (
    "You are an expert in ontology matching. "
    "You will be asked to determine if two entities from different ontologies "
    "are semantically equivalent. "
)

DEV_PROMPT_CLASS_EQUIVALENCE = (
    "You are assisting in an OWL ontology alignment exercise. "
    "You will be presented with a pair of class entities, one from each of two "
    "different ontologies. Additional ontological context such as parent "
    "categories and synonyms may be provided. You will be asked to decide "
    "whether the two classes are semantically equivalent. "
)

DEV_PROMPT_BIOMEDICAL = (
    "You are a biomedical ontology expert. "
    "You will be presented with two biomedical concepts from different ontologies. "
    "Additional ontological context such as hierarchical placement and synonyms "
    "may be provided. You will be asked to decide whether the two concepts are "
    "semantically equivalent. "
)

DEV_PROMPT_BIOMEDICAL_EQUIV_SYNONYMS = (
    "You are a biomedical ontology expert. "
    "You will be presented with two biomedical concepts from different ontologies, "
    "along with their synonyms and hierarchical context. "
    "You will be asked to decide whether the two concepts are semantically equivalent. "
)

DEV_PROMPT_CLASS_SUBSUMPTION = (
    "You are an expert in ontology matching. "
    "You will be asked to determine if one class is a subclass of another. "
)

DEV_PROMPT_CLASS_SUBSUMPTION_MOD = (
    "You are an expert in ontology matching and subsumption inference. "
    "You will be presented with two classes from different ontologies, along "
    "with hierarchical context. You will be asked to decide if one class is "
    "a subclass of the other. "
)

DEV_PROMPT_BIOMEDICAL_SUBSUMPTION = (
    "You are a biomedical ontology expert. "
    "You will be asked to determine if one biomedical concept is a subclass of "
    "another. "
)

DEV_PROMPT_CONFERENCE_CLASS = (
    "You are assisting in an OWL ontology alignment exercise. "
    "You will be presented with a pair of class entities from different "
    "ontologies describing the domain of academic conferences. "
    "Additional ontological context such as parent categories and synonyms "
    "may be provided. You will be asked to decide whether the two classes "
    "are semantically equivalent. "
)

DEV_PROMPT_PROPERTY_EQUIVALENCE = (
    "You are assisting in an ontology alignment exercise. "
    "You will be presented with a pair of properties (relationships) from "
    "different ontologies. Additional context such as domain and range "
    "classes may be provided. You will be asked to decide whether the two "
    "properties represent the same relationship. "
)

DEV_PROMPT_INSTANCE_EQUIVALENCE = (
    "You are assisting in an entity resolution exercise across knowledge graphs. "
    "You will be presented with a pair of entities, one from each of two different "
    "knowledge graphs. Additional context such as entity types, attributes, and "
    "relationships may be provided. You will be asked to decide whether the two "
    "entities refer to the same real-world entity. "
)

###
# DEVELOPER (SYSTEM) PROMPT INSTRUCTIONS
###

TRUE_FALSE_PLAIN = (
    'Give only a one-word binary response: "True" or "False".'
)

TRUE_FALSE_STRUCTURED = (
    'Respond with a JSON object containing a single boolean field "answer": {"answer": true} or {"answer": false}.'
)

YES_NO_PLAIN = (
    'Give only a one-word binary response: "Yes" or "No".'
)

YES_NO_STRUCTURED = (
    'Respond with a JSON object containing a single string field "answer" with value "Yes" or "No": {"answer": "Yes"} or {"answer": "No"}.'
)

###
# DEVELOPER (SYSTEM) PROMPT REGISTRY
###

DEVELOPER_PROMPT_REGISTRY = {
    "generic": DEV_PROMPT_GENERIC,
    "class_equivalence": DEV_PROMPT_CLASS_EQUIVALENCE,
    "biomedical": DEV_PROMPT_BIOMEDICAL,
    "biomedical_equiv_synonyms": DEV_PROMPT_BIOMEDICAL_EQUIV_SYNONYMS,
    "class_subsumption": DEV_PROMPT_CLASS_SUBSUMPTION,
    "class_subsumption_mod": DEV_PROMPT_CLASS_SUBSUMPTION_MOD,
    "biomedical_subsumption": DEV_PROMPT_BIOMEDICAL_SUBSUMPTION,
    "conference_class": DEV_PROMPT_CONFERENCE_CLASS,
    "property_equivalence": DEV_PROMPT_PROPERTY_EQUIVALENCE,
    "instance_equivalence": DEV_PROMPT_INSTANCE_EQUIVALENCE,
    "none": None,
}

###
# DEVELOPER (SYSTEM) PROMPT INSTRUCTION REGISTRY
###

DEVELOPER_PROMPT_INSTRUCTION_REGISTRY = {
    ("true_false",  "plain"     ): TRUE_FALSE_PLAIN,
    ("true_false",  "structured"): TRUE_FALSE_STRUCTURED,
    ("yes_no",      "plain"     ): YES_NO_PLAIN,
    ("yes_no",      "structured"): YES_NO_STRUCTURED,
}

def get_developer_prompt(name: str, answer_format: str = "true_false", response_mode: str = "structured") -> str | None:
    """Resolve a developer prompt by its registry name, optionally resolve its answer_format and response mode."""
    if name not in DEVELOPER_PROMPT_REGISTRY:
        raise ValueError(f"Developer prompt '{name}' not found in registry.")
    if (answer_format, response_mode) not in DEVELOPER_PROMPT_INSTRUCTION_REGISTRY.keys():
        raise ValueError(f"Developer instruction ('{answer_format}', '{response_mode}') not found in registry.")
    return DEVELOPER_PROMPT_REGISTRY[name] + DEVELOPER_PROMPT_INSTRUCTION_REGISTRY[(answer_format, response_mode)]

