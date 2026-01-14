'''
This module contains functionality supporting the building of
LLM Oracle 'user' prompts.

Note: LLM Oracle 'developer' prompts are managed elsewhere.
'''

#%%

from onto_access import OntologyAccess
from onto_object import OntologyEntryAttr
from tqdm import tqdm
from constants import PAIRS_SEPARATOR

from oracle_user_prompt_templates import (
    oupt_one_level_of_parents_structured,
    oupt_two_levels_of_parents_structured,
    oupt_one_level_of_parents,
    oupt_two_levels_of_parents,
    oupt_one_level_of_parents_and_synonyms,
    oupt_two_levels_of_parents_and_synonyms
)

oupt_templates_2_oupt_functions = {
    prompt_function.__name__.replace("oupt_", ""): prompt_function
    for prompt_function in [
        oupt_one_level_of_parents_structured,
        oupt_two_levels_of_parents_structured,
        oupt_one_level_of_parents,
        oupt_two_levels_of_parents,
        oupt_one_level_of_parents_and_synonyms,
        oupt_two_levels_of_parents_and_synonyms
    ]
}

#%%

def get_oracle_user_prompt_template_function(oupt_name):

    return oupt_templates_2_oupt_functions[oupt_name]


#%%

def build_oracle_user_prompts(oupt_name, onto_src_filepath, 
                              onto_tgt_filepath, m_ask_df):
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
    
    Returns
    -------
    m_ask_oracle_user_prompts : dictionary
        the oracle user prompts built for each mapping in m_ask_df
        key: string 
            a mapping identifier ('src_entity_uri|tgt_entity_uri')
        value: string 
            a formatted string representing a prepared oracle user prompt
    '''
    
    try:
        # instantiate an OntologyAccess object for the source ontology
        for onto_path in tqdm(iterable=[onto_src_filepath],
                            desc='Preparing source ontology'):
            OA_source_onto = OntologyAccess(onto_path, annotate_on_init=True)
        
        print()

        # instantiate an OntologyAccess object for the target ontology
        for onto_path in tqdm(iterable=[onto_tgt_filepath],
                            desc='Preparing target ontology'):
            OA_target_onto = OntologyAccess(onto_path, annotate_on_init=True)
    except Exception as e:
        raise e
    
    # get the function for the specified oracle 'user' prompt template
    prompt_function = get_oracle_user_prompt_template_function(oupt_name)
    print()
    print(f"Prompt template function obtained: {oupt_name}")
    print()
    
    # initialise a container for the prompts
    m_ask_oracle_user_prompts = {}

    # iterate over the mappings in m_ask
    # (each 'row' is an (index, Series) tuple)
    for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0], 
                    desc="Building the prompts"):
        
        # get the URIs of the two entities involved in the current mapping
        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]

        # get attributes of the ontological neighbourhood of the source 
        # and target entities, subsets of which are likely fillers for the
        # prompt template being used to build the user prompts
        src_entity_onto_attrs = OntologyEntryAttr(src_entity_uri, OA_source_onto)
        tgt_entity_onto_attrs = OntologyEntryAttr(tgt_entity_uri, OA_target_onto)

        # build the oracle user prompt for the current mapping
        oracle_user_prompt = prompt_function(src_entity_onto_attrs, 
                                             tgt_entity_onto_attrs)
        
        # store the oracle user prompt for the current mapping
        key = src_entity_uri + PAIRS_SEPARATOR + tgt_entity_uri 
        m_ask_oracle_user_prompts[key] = oracle_user_prompt
    
    return m_ask_oracle_user_prompts


#%%







