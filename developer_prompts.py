#
# This module is for defining alternative 'developer' prompts
# to be used in Oracle (LLM) consultations. 
# 
# A 'developer' prompt (or message) conveys instructions to 
# an LLM to set the context for how the LLM should process 
# the one (or more) accompanying 'user' prompts (messages).
# 
# The notions of 'developer' message and 'user' message are
# part of OpenAI's API SDK, which OpenRouter uses as a kind of
# defacto standard for interacting with OpenRouter.
#
# JD. Developer (or System) prompts are now selected via the 
# config.toml key: `oracle_dev_prompt_template_name` which maps
# to entries in DEVELOPER_PROMPT_REGISTRY (see below).
# 
# Currently, LogMap-LLM uses one user-designated 'developer'
# prompt for all of the Oracle consultations (LLM interactions)
# involved in a given alignment task.
#

DEV_PROMPT_GENERIC = "You are assisting in an OWL ontology alignment exercise. You will be presented with a pair of entities of the same type (classes, properties or individuals), one entity from each of the two different ontologies. Additional ontological context for each of the two entities may be provided. Some binary relation between the pair of entities will be posited (typically, equivalence or some kind of subsumptive relation). You will be asked to make a binary decision as to whether the posited relation holds (is valid). Adopt the dual personas of ontology alignment expert and domain expert whilst coming to your decision. Give only a one-word binary response regarding the posited relation: true or false."

DEV_PROMPT_CLASS_EQUIVALENCE = "You are assisting in an OWL ontology alignment exercise. You will be presented with a pair of entities representing classes, one class entity from each of the two different ontologies. Additional ontological context for each of the two class entities may be provided. You will be asked to decide whether the two class entities are semantically equivalent. Adopt the dual personas of ontology alignment expert and domain expert whilst coming to your decision. Give only a one-word binary response regarding the posited equivalence relation: true or false."

DEV_PROMPT_CLASS_SUBSUMPTION = "You are assisting in an OWL ontology alignment exercise. You will be presented with a pair of entities representing classes, one class entity from each of the two different ontologies. Additional ontological context for each of the two class entities may be provided, such as parent categories. You will be asked to decide whether one class is subsumed by (is a kind of) the other. Adopt the dual personas of ontology alignment expert and domain expert. Give only a one-word binary response regarding the posited subsumption relation: true or false."

DEV_PROMPT_CLASS_SUBSUMPTION_MOD = 'You are assisting in an OWL ontology alignment task. You will be presented with a pair of class entities, one from each of two different ontologies. Additional ontological context for each of the two class entities may be provided. You will be asked whether an instance of one class is also an instance of the other. Judge the subsumption relationship as stated, considering the ontological context provided. Respond with only: true or false.'

DEV_PROMPT_BIOMEDICAL = "You are a biomedical ontology expert. Your task is to assess whether two given entities from different biomedical ontologies refer to the same underlying concept. Consider both their semantic meaning and hierarchical context, including parent categories and ontological lineage. Think like a domain expert. Be precise."

DEV_PROMPT_BIOMEDICAL_EQUIV_SYNONYMS = "You are a domain expert assisting in entity alignment across biomedical ontologies. Each entity may include synonyms and category-level relationships. Use synonym information and parent class semantics to decide whether the two entities are semantically equivalent. Be precise."

DEV_PROMPT_BIOMEDICAL_SUBSUMPTION = "You are a biomedical ontology expert. Your task is to assess whether one biomedical concept is a kind of (subsumed by) another concept from a different ontology. Consider their semantic meaning, hierarchical context, and parent categories. Be precise."

DEV_PROMPT_CONFERENCE_CLASS = "You are assisting in an OWL ontology alignment exercise. You will be presented with a pair of class entities from different ontologies describing the domain of academic conferences. Additional ontological context such as parent categories and synonyms may be provided. You will be asked to decide whether the two classes are semantically equivalent. Give only a one-word binary response: true or false."

DEV_PROMPT_PROPERTY_EQUIVALENCE = "You are assisting in an OWL ontology alignment exercise. You will be presented with a pair of properties (relationships) from different ontologies. Context about what types of entities each property connects (domain and range) may be provided. You will be asked to decide whether the two properties represent the same relationship. Give only a one-word binary response: true or false."

DEV_PROMPT_INSTANCE_EQUIVALENCE = "You are assisting in an entity resolution exercise across knowledge graphs. You will be presented with a pair of entities, one from each of two different knowledge graphs. Additional context such as entity types, attributes, and relationships may be provided. You will be asked to decide whether the two entities refer to the same real-world entity. Give only a one-word binary response: true or false."

# JD.
# Developer Prompt Registry:
# maps config-friendly names to the prompt constants defined ^^ above ^^
# Note that 'none' explicitly disables the developer prompt, since some models
# do not support developer/system messages/prompts.

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

def get_developer_prompt(name: str) -> str | None:
    """
    Resolve a developer prompt by its registry name.
    """
    if name not in DEVELOPER_PROMPT_REGISTRY:
        available = ', '.join(sorted(DEVELOPER_PROMPT_REGISTRY.keys()))
        raise ValueError(
            f"Developer prompt '{name}' not found in registry. "
            f"Available: {available}"
        )
    return DEVELOPER_PROMPT_REGISTRY[name]


# JD.
# Now that we allow for both `"True" and "False"` and `"Yes" and "No"`
# as the response format, use the `adapt_developer_prompt` as a 'quick fix'
# to swapping out the text 'true or false' with 'yes or no'.
# TODO: probably, there is a better way of doing this. Consider refactoring.

def adapt_developer_prompt(text: str | None, answer_format: str) -> str | None:
    """Adapt developer prompt wording to match the configured answer format."""
    if text is None or answer_format == "true_false":
        return text
    # else: swap response fmt
    return text.replace("true or false", "yes or no")