'''
This module contains functionality supporting the building of
LLM Oracle 'user' prompts.

Note: LLM Oracle 'developer' prompts are managed elsewhere.
'''

#%%

from __future__ import annotations
from typing import Callable
from onto_access import OntologyAccess
from onto_object import OntologyEntryAttr, ClassNotFoundError, PropertyEntryAttr, PropertyNotFoundError, InstanceEntryAttr, InstanceNotFoundError, resolve_entity
from tqdm import tqdm
from constants import PAIRS_SEPARATOR

from log_utils import error, warn, warning, info, step, success

from functools import partial

from oracle_user_prompt_templates import (
    oupt_one_level_of_parents_structured,
    oupt_two_levels_of_parents_structured,
    oupt_one_level_of_parents,
    oupt_two_levels_of_parents,
    oupt_one_level_of_parents_and_synonyms,
    oupt_two_levels_of_parents_and_synonyms,
    oupt_sub_labels_only,
    oupt_equiv_parents_siblings,
    oupt_equiv_parents_synonyms_siblings,
    oupt_sub_parents_synonyms,
    oupt_equiv_with_ancestral_disjointness,
    oupt_deductive_equiv,
    oupt_synonyms_only,
    oupt_prop_labels_only,
    oupt_prop_domain_range,
    oupt_prop_domain_range_synonyms,
    oupt_inst_labels_only,
    oupt_inst_labels_with_types,
    oupt_inst_types_and_attributes,
    oupt_inst_full_context,
    oupt_inst_types_and_attributes_intersect,
    oupt_inst_full_context_intersect,
    oupt_inst_types_and_attributes_entropy,
    oupt_inst_full_context_entropy,
)

oupt_templates_2_oupt_functions = {
    prompt_function.__name__.replace("oupt_", ""): prompt_function
    for prompt_function in [
        oupt_one_level_of_parents_structured,
        oupt_two_levels_of_parents_structured,
        oupt_one_level_of_parents,
        oupt_two_levels_of_parents,
        oupt_one_level_of_parents_and_synonyms,
        oupt_two_levels_of_parents_and_synonyms,
        oupt_sub_labels_only,
        oupt_equiv_parents_siblings,
        oupt_equiv_parents_synonyms_siblings,
        oupt_sub_parents_synonyms,
        oupt_equiv_with_ancestral_disjointness,
        oupt_deductive_equiv,
        oupt_synonyms_only,
    ]
}

# template names that require bidirectional prompt generation
# when one of these is selected, the pipeline should use build_oracle_user_prompts_bidirectional() 
# instead of build_oracle_user_prompts()

BIDIRECTIONAL_TEMPLATES = {
    "sub_labels_only",
    "sub_parents_synonyms",
    "equiv_with_ancestral_disjointness",
}

# template names that require a SiblingSelector instance - when one of these is selected, the pipeline 
# initialises a SiblingSelector and binds it to the template via functools.partial

SIBLING_TEMPLATES = {
    "equiv_parents_siblings",
    "equiv_parents_synonyms_siblings",
}

# property prompt template registry — for Conference track property mappings
# these templates accept PropertyEntryAttr rather than OntologyEntryAttr

property_templates_2_functions = {
    "prop_labels_only": oupt_prop_labels_only,
    "prop_domain_range": oupt_prop_domain_range,
    "prop_domain_range_synonyms": oupt_prop_domain_range_synonyms,
}

PROPERTY_TEMPLATES = set(property_templates_2_functions.keys())

# instance prompt template registry — for KG track individual mappings
# these templates accept InstanceEntryAttr rather than OntologyEntryAttr

instance_templates_2_functions = {
    "inst_labels_only": oupt_inst_labels_only,
    "inst_labels_with_types": oupt_inst_labels_with_types,
    "inst_types_and_attributes": oupt_inst_types_and_attributes,
    "inst_full_context": oupt_inst_full_context,
    "inst_types_and_attributes_intersect": oupt_inst_types_and_attributes_intersect,
    "inst_full_context_intersect": oupt_inst_full_context_intersect,
    "inst_types_and_attributes_entropy": oupt_inst_types_and_attributes_entropy,
    "inst_full_context_entropy": oupt_inst_full_context_entropy,
}

INSTANCE_TEMPLATES = set(instance_templates_2_functions.keys())

#%%

def get_oracle_user_prompt_template_function(oupt_name):
    """Docstring for get_oracle_user_prompt_template_function"""
    return oupt_templates_2_oupt_functions[oupt_name]


def is_bidirectional_template(oupt_name):
    return oupt_name in BIDIRECTIONAL_TEMPLATES


def is_sibling_template(oupt_name):
    return oupt_name in SIBLING_TEMPLATES


def get_property_prompt_template_function(name):
    return property_templates_2_functions.get(name)


def is_property_template(name):
    return name in PROPERTY_TEMPLATES


def get_instance_prompt_template_function(name):
    return instance_templates_2_functions.get(name)


def is_instance_template(name):
    return name in INSTANCE_TEMPLATES


#%%

def load_ontologies(onto_src_filepath, onto_tgt_filepath, cache_dir=None):
    '''
    Load and return OntologyAccess objects for source and target ontologies.

    Separated from prompt building to allow reuse across multiple
    prompt-building passes without reloading ontologies.

    Parameters
    ----------
    onto_src_filepath : string
        an absolute path to the source ontology file
    onto_tgt_filepath : string
        an absolute path to the target ontology file
    cache_dir : string or None, optional
        Path to a directory for owlready2 quadstore caching.  When
        provided, parsed ontologies are cached to SQLite files on disk
        so that subsequent runs skip the expensive OWL/RDF parse.
        When None (default), ontologies are parsed into ephemeral
        in-memory quadstores (original behaviour).

    Returns
    -------
    OA_source_onto : OntologyAccess
        the source ontology
    OA_target_onto : OntologyAccess
        the target ontology
    '''
    
    def _print_padded_callback(fn: Callable, **kwargs):
        print()
        fn(**kwargs)
        print()

    def _print_load_message(onto_str: str):
        step(f"-" * 50)
        step(f"{'-' * 15}LOADING ONTOLOGY {onto_str}{'-' * 15}")
        step(f"-" * 50)

    def _print_done_message():
        success("-" * 50)
        success(f"{'-' * 23}DONE{'-' * 23}")
        success("-" * 50)

    # TODO: I thought I removed this weird callback functionality
    # for printing... (TODO: fix this)

    _print_padded_callback(_print_load_message, onto_str="ONE")

    OA_source_onto = OntologyAccess(onto_src_filepath, annotate_on_init=True, cache_dir=cache_dir)
    
    _print_padded_callback(_print_done_message)
    _print_padded_callback(_print_load_message, onto_str="TWO")

    OA_target_onto = OntologyAccess(onto_tgt_filepath, annotate_on_init=True, cache_dir=cache_dir)
    
    _print_padded_callback(_print_done_message)

    return OA_source_onto, OA_target_onto


#%%

def build_oracle_user_prompts(oupt_name, onto_src_filepath, 
                              onto_tgt_filepath, m_ask_df,
                              OA_source=None, OA_target=None,
                              sibling_selector=None,
                              property_prompt_function=None,
                              instance_prompt_function=None):
    '''
    Build oracle user prompts using a particular template.

    Parameters
    ----------
    oupt_name : string
        the name of an oracle user prompt template
    onto_src_filepath : string
        an absolute path to an ontology file
    onto_tgt_filepath : string
        an absolute path to an ontology file
    m_ask_df : pandas DataFrame
        a Python representation of m_ask from a LogMap alignment
        - the representation is an in-memory replica of LogMap's output
          file: 'logmap_mappings_to_ask_oracle_user_llm.txt'
    OA_source : OntologyAccess, optional
        pre-loaded source ontology (avoids reloading for multi-pass use)
    OA_target : OntologyAccess, optional
        pre-loaded target ontology (avoids reloading for multi-pass use)
    sibling_selector : SiblingSelector, optional
        pre-initialised SiblingSelector for templates that require sibling
        context (see SIBLING_TEMPLATES).  If None and the template requires
        siblings, the template's fallback (alphabetical selection) is used.
    property_prompt_function : callable, optional
        Template function for property mappings (Conference track).
        Accepts (PropertyEntryAttr, PropertyEntryAttr) -> str.
        If None, property mappings in M_ask are skipped with a warning.
    instance_prompt_function : callable, optional
        Template function for instance mappings (KG track).
        Accepts (InstanceEntryAttr, InstanceEntryAttr) -> str.
        If None, instance mappings in M_ask are skipped with a warning.
    
    Returns
    -------
    m_ask_oracle_user_prompts : dictionary
        the oracle user prompts built for each mapping in m_ask_df
        key: string 
            a mapping identifier ('src_entity_uri|tgt_entity_uri')
        value: string 
            a formatted string representing a prepared oracle user prompt
    '''
    
    # Load ontologies if not pre-loaded
    if OA_source is None or OA_target is None:
        OA_source_onto, OA_target_onto = load_ontologies(
            onto_src_filepath, onto_tgt_filepath
        )
    else:
        OA_source_onto = OA_source
        OA_target_onto = OA_target
    
    # get the function for the specified oracle 'user' prompt template
    prompt_function = get_oracle_user_prompt_template_function(oupt_name)

    # bind sibling_selector to template if it accepts one
    if sibling_selector is not None and is_sibling_template(oupt_name):
        prompt_function = partial(prompt_function, sibling_selector=sibling_selector)
        print(f"  SiblingSelector bound to template (cache: {sibling_selector.cache_size} embeddings)")

    print()
    print(f"Prompt template function obtained: {oupt_name}")
    if property_prompt_function is not None:
        print(f"Property prompt function: {property_prompt_function.__name__}")
    if instance_prompt_function is not None:
        print(f"Instance prompt function: {instance_prompt_function.__name__}")
    print()
    
    # initialise a container for the prompts
    m_ask_oracle_user_prompts = {}

    # track skipped mappings for summary reporting
    skipped_mappings = []

    # counters for entity type reporting
    n_class_prompts = 0
    n_property_prompts = 0
    n_instance_prompts = 0

    # iterate over the mappings in m_ask
    # (each 'row' is an (index, Series) tuple)
    for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0], 
                    desc="Building the prompts"):
        
        # get the URIs of the two entities involved in the current mapping
        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]

        try:
            # Resolve entities — dispatches to class, property, or instance
            src_entity, src_type = resolve_entity(src_entity_uri, OA_source_onto)
            tgt_entity, tgt_type = resolve_entity(tgt_entity_uri, OA_target_onto)

            if src_type == "instance" or tgt_type == "instance":
                # Instance mapping — both sides should be instances
                if src_type != tgt_type:
                    skipped_mappings.append({
                        "src": src_entity_uri,
                        "tgt": tgt_entity_uri,
                        "reason": f"Mixed entity types: src={src_type}, tgt={tgt_type}",
                    })
                    tqdm.write(f"  WARNING: Skipping mixed-type mapping "
                               f"(src={src_type}, tgt={tgt_type})")
                    continue

                if instance_prompt_function is None:
                    skipped_mappings.append({
                        "src": src_entity_uri,
                        "tgt": tgt_entity_uri,
                        "reason": "Instance mapping but no instance template configured",
                    })
                    tqdm.write(f"  WARNING: Skipping instance mapping — "
                               f"no instance template configured")
                    continue

                oracle_user_prompt = instance_prompt_function(src_entity, tgt_entity)
                n_instance_prompts += 1

            elif src_type == "property" or tgt_type == "property":
                # Property mapping — both sides should be properties
                if src_type != tgt_type:
                    skipped_mappings.append({
                        "src": src_entity_uri,
                        "tgt": tgt_entity_uri,
                        "reason": f"Mixed entity types: src={src_type}, tgt={tgt_type}",
                    })
                    tqdm.write(f"  WARNING: Skipping mixed-type mapping "
                               f"(src={src_type}, tgt={tgt_type})")
                    continue

                if property_prompt_function is None:
                    skipped_mappings.append({
                        "src": src_entity_uri,
                        "tgt": tgt_entity_uri,
                        "reason": "Property mapping but no property template configured",
                    })
                    tqdm.write(f"  WARNING: Skipping property mapping — "
                               f"no property template configured")
                    continue

                oracle_user_prompt = property_prompt_function(src_entity, tgt_entity)
                n_property_prompts += 1
            else:
                # Class mapping — use the standard class template
                oracle_user_prompt = prompt_function(src_entity, tgt_entity)
                n_class_prompts += 1
            
            # store the oracle user prompt for the current mapping
            key = src_entity_uri + PAIRS_SEPARATOR + tgt_entity_uri 
            m_ask_oracle_user_prompts[key] = oracle_user_prompt

        except (ClassNotFoundError, PropertyNotFoundError, InstanceNotFoundError) as e:
            # Entity URI exists in LogMap's M_ask but cannot be resolved.
            skipped_mappings.append({
                "src": src_entity_uri,
                "tgt": tgt_entity_uri,
                "reason": str(e),
            })
            tqdm.write(f"  WARNING: Skipping mapping - {e}")

    # Print summary of skipped mappings
    if skipped_mappings:
        print()
        print(f"WARNING: {len(skipped_mappings)} of {m_ask_df.shape[0]} "
              f"mappings skipped due to unresolvable entity URIs.")
        print(f"  Skipped URIs:")
        # Collect unique unresolvable URIs for a compact summary
        unresolvable_uris = set()
        for s in skipped_mappings:
            # Check which side failed --- the URI that isn't in the onto
            if OA_source_onto.getEntityByURI(s["src"]) is None:
                unresolvable_uris.add(s["src"])
            if OA_target_onto.getEntityByURI(s["tgt"]) is None:
                unresolvable_uris.add(s["tgt"])
        for uri in sorted(unresolvable_uris):
            print(f"    {uri}")
        print()
    else:
        print()
        print(f"All {m_ask_df.shape[0]} mappings resolved successfully.")
        print()
    
    # Report entity type breakdown if mixed
    if n_property_prompts > 0 or n_instance_prompts > 0:
        print(f"  Prompts built: {n_class_prompts} class, "
              f"{n_property_prompts} property, {n_instance_prompts} instance")

    return m_ask_oracle_user_prompts


#%%

def build_oracle_user_prompts_bidirectional(oupt_name, onto_src_filepath,
                                             onto_tgt_filepath, m_ask_df,
                                             OA_source=None, OA_target=None,
                                             sibling_selector=None):
    '''
    Build bidirectional subsumption prompts for m_ask candidates

    For each candidate pair (C, D), produces two prompts:
      - Forward:  template(C, D)  — "Is C a kind of D?"
      - Reverse:  template(D, C)  — "Is D a kind of C?"

    only processes m_ask candidates with relation '=' (equivalence)
    candidates with '<' or '>' relations are skipped with a log message, as these are repair byproducts
    TODO: consider modifying this in future ^^^

    Parameters
    ----------
    oupt_name : string
        the name of an oracle user prompt template (should be a
        bidirectional/subsumption template from BIDIRECTIONAL_TEMPLATES)
    onto_src_filepath : string
        an absolute path to the source ontology file
    onto_tgt_filepath : string
        an absolute path to the target ontology file
    m_ask_df : pandas DataFrame
        a Python representation of m_ask from a LogMap alignment
    OA_source : OntologyAccess, optional
        pre-loaded source ontology (avoids reloading for multi-pass use)
    OA_target : OntologyAccess, optional
        pre-loaded target ontology (avoids reloading for multi-pass use)
    sibling_selector : SiblingSelector, optional
        pre-initialised SiblingSelector for templates that require sibling
        context.  Currently no bidirectional templates use siblings, but
        this parameter is accepted for forward-compatibility.

    Returns
    -------
    m_ask_oracle_user_prompts : dictionary
        the oracle user prompts built for each mapping in m_ask_df
        key: string
            direction-annotated mapping identifier:
            'src_entity_uri|tgt_entity_uri|forward' or
            'src_entity_uri|tgt_entity_uri|reverse'
        value: string
            a formatted string representing a prepared oracle user prompt
    n_equiv_candidates : int
        number of '=' candidates that were processed
    n_non_equiv_skipped : int
        number of '<' / '>' candidates that were skipped
    '''

    # Load ontologies if not pre-loaded
    if OA_source is None or OA_target is None:
        OA_source_onto, OA_target_onto = load_ontologies(
            onto_src_filepath, onto_tgt_filepath
        )
    else:
        OA_source_onto = OA_source
        OA_target_onto = OA_target

    # get the function for the specified oracle 'user' prompt template
    prompt_function = get_oracle_user_prompt_template_function(oupt_name)

    # bind sibling_selector to template if it accepts one
    if sibling_selector is not None and is_sibling_template(oupt_name):
        prompt_function = partial(prompt_function, sibling_selector=sibling_selector)
        print(f"  SiblingSelector bound to template (cache: {sibling_selector.cache_size} embeddings)")

    print()
    print(f"Prompt template function obtained: {oupt_name} (bidirectional mode)")
    print()

    # initialise containers
    m_ask_oracle_user_prompts = {}
    skipped_mappings = []
    n_non_equiv_skipped = 0
    n_equiv_candidates = 0

    # Check if the relation column exists in m_ask_df
    # The relation column is at index 2 per bridging.py column order:
    #   [source_entity_uri, target_entity_uri, relation, confidence, entityType]
    has_relation_col = m_ask_df.shape[1] > 2

    for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0],
                    desc="Building bidirectional prompts"):

        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]

        # Filter: only process equivalence ('=') candidates
        if has_relation_col:
            relation = row_series.iloc[2]
            if relation != '=':
                n_non_equiv_skipped += 1
                continue

        n_equiv_candidates += 1

        try:
            src_entity_onto_attrs = OntologyEntryAttr(src_entity_uri, OA_source_onto)
            tgt_entity_onto_attrs = OntologyEntryAttr(tgt_entity_uri, OA_target_onto)

            # forward: Is src a kind of tgt?
            forward_prompt = prompt_function(src_entity_onto_attrs, tgt_entity_onto_attrs)

            # reverse: Is tgt a kind of src?
            # Note: we swap the arguments so the template asks about tgt being subsumed by src
            reverse_prompt = prompt_function(tgt_entity_onto_attrs, src_entity_onto_attrs)

            # store with direction-annotated keys
            base_key = src_entity_uri + PAIRS_SEPARATOR + tgt_entity_uri
            m_ask_oracle_user_prompts[base_key + PAIRS_SEPARATOR + "forward"] = forward_prompt
            m_ask_oracle_user_prompts[base_key + PAIRS_SEPARATOR + "reverse"] = reverse_prompt

        except (ClassNotFoundError, PropertyNotFoundError) as e:
            skipped_mappings.append({
                "src": src_entity_uri,
                "tgt": tgt_entity_uri,
                "reason": str(e),
            })
            tqdm.write(f"  WARNING: Skipping mapping - {e}")

    # Summary reporting
    print()
    print(f"Bidirectional prompt building summary:")
    print(f"  Equivalence ('=') candidates processed: {n_equiv_candidates}")
    if n_non_equiv_skipped > 0:
        print(f"  Non-equivalence ('<'/'>' repair) candidates skipped: {n_non_equiv_skipped}")
    n_prompt_pairs = len(m_ask_oracle_user_prompts) // 2
    print(f"  Prompt pairs built: {n_prompt_pairs} (forward + reverse)")
    print(f"  Total individual prompts: {len(m_ask_oracle_user_prompts)}")

    if skipped_mappings:
        print()
        print(f"WARNING: {len(skipped_mappings)} of {n_equiv_candidates} "
              f"equivalence candidates skipped due to unresolvable entity URIs.")
        unresolvable_uris = set()
        for s in skipped_mappings:
            if OA_source_onto.getEntityByURI(s["src"]) is None:
                unresolvable_uris.add(s["src"])
            if OA_target_onto.getEntityByURI(s["tgt"]) is None:
                unresolvable_uris.add(s["tgt"])
        print(f"  Skipped URIs:")
        for uri in sorted(unresolvable_uris):
            print(f"    {uri}")
    else:
        print(f"  All equivalence candidates resolved successfully.")

    print()

    return m_ask_oracle_user_prompts, n_equiv_candidates, n_non_equiv_skipped



#%%