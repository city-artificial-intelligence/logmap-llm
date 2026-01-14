'''
This module contains functionality that supports consulting 
(interacting with) LLM Oracles.
'''
from __future__ import annotations

import functools
from typing import Any, Callable
from oracle_consultation_managers import OracleConsultationManager_OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from constants import BinaryOutputFormat, BinaryOutputFormatWithReasoning
from tqdm import tqdm
import time
import numpy as np
import developer_prompts as dp
from openai import BadRequestError


def retry(max_retries: int = 1) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Retry the function up to `max_retries` times if an exception occurs."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
            attempts = 0
            while attempts <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts > max_retries:
                        raise e
            return None

        return wrapper

    return decorator


def get_llm_mapping_prediction(response):
    '''
    Get the prediction of an LLM Oracle regarding a candidate
    mapping from within the set of attributes gathered 
    regarding the LLM's overall response.

    Parameters
    ----------
    response : LLMCallOuput

    Returns
    -------
    answer : bool
        An LLM's prediction as to whether a candidate mapping represents
        a True or a False mapping.
    '''
    if isinstance(response.parsed, (BinaryOutputFormat, BinaryOutputFormatWithReasoning)):
        return response.parsed.answer
    raise NotImplementedError()


def calculate_logprobs_confidence(log_probs: list) -> float:
    for pred_tocken_info in log_probs:
        if pred_tocken_info["token"].strip() not in ["true", "false"]:
            continue
        top_logprobs = pred_tocken_info["top_logprobs"]
        positive_logprob = max([e["logprob"] for e in top_logprobs if e["token"].strip() == "true"], default=np.nan)
        negative_logprob = max([e["logprob"] for e in top_logprobs if e["token"].strip() == "false"], default=np.nan)
        break
    else:
        positive_logprob = 0.0
        negative_logprob = 0.0
    return np.nanmax([np.exp(positive_logprob), np.exp(negative_logprob)])


@retry(max_retries=1)
def consult_oracle_for_mapping(entity_pair, prompt, llm_oracle):
    '''
    Consult an LLM Oracle for one mapping.

    If an exception occurs, the function is retried once.

    Parameters
    ----------
    entity_pair : str
        'src_entity_uri|tgt_entity_uri'
    prompt : str
        A formatted user prompt for a particular mapping
    llm_oracle : OracleConsultationManager
        An OracleConsultationManager of some kind

    Returns
    -------
    mapping_prediction : list
        An LLM prediction for a mapping, formatted as
        [src_entity_uri, tgt_entity_uri, prediction, confidence]
    token_usage : tuple 
        A 2-tuple of (input tokens, output tokens) for the interaction
        with the LLM
    '''

    sleep_time = 0.0

    src_entity_uri, tgt_entity_uri = entity_pair.split('|')

    try:
        if sleep_time > 0:
            time.sleep(sleep_time)

        # consult an LLM Oracle regarding a candidate mapping
        response = llm_oracle.consult_oracle(prompt)
        
        # retrieve the LLM's prediction regarding the candidate mapping
        prediction = get_llm_mapping_prediction(response)

        # gather token stats from tracking overall token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        token_usage = (input_tokens, output_tokens)

        confidence = calculate_logprobs_confidence(response.logprobs)

        mapping_prediction = [src_entity_uri, tgt_entity_uri, prediction, confidence]
        
        return mapping_prediction, token_usage

    except BadRequestError as bre:
        print('ERROR during Oracle consultation for mapping')
        print(entity_pair)
        # try to extract the error msg from within the OpenAI BadRequestError
        parts1 = bre.message.split("'message':")
        parts2 = parts1[1].split(",")
        error_msg = parts2[0].lstrip()
        if len(error_msg) > 0:
            print(f'BadRequestError: {error_msg}')
        else:
            print(f'BadRequestError: {bre}')
        print()
        mapping_prediction = [src_entity_uri, tgt_entity_uri, "error", str(np.nan)]
        token_usage = (np.nan, np.nan)
        return mapping_prediction, token_usage
    
    except Exception as e:
        print('ERROR during Oracle consultation for mapping')
        print(entity_pair)
        print(f'Exception: {e}')
        mapping_prediction = [src_entity_uri, tgt_entity_uri, "error", str(np.nan)]
        token_usage = (np.nan, np.nan)
        return mapping_prediction, token_usage


def consult_oracle_for_mappings_to_ask(m_ask_oracle_user_prompts, 
                                       api_key,
                                       model_name,
                                       max_workers,
                                       m_ask_df):
    '''
    Consult an LLM Oracle for each candidate mapping in a set of
    candidate mappings.

    Parameters
    ----------
    m_ask_oracle_user_prompts : dictionary
        The LLM user prompts prepared for each mapping to ask an Oracle.
        key : str 
            - An entity pair 'src_entity_uri|tgt_entity_uri'
        value : str
            - An LLM user prompt for a particular mapping as a formatted
            string.
    api_key : str
        An LLM API key
    model_name : str
        An LLM model name
    max_workers : int
        The maximum number of Oracle consultations to run in parallel,
        via thread pooling.
    m_ask_df : pandas DataFrame
        The LogMap mappings_to_ask an Oracle

    Returns
    -------
    m_ask_df_ext : pandas DataFrame
        A copy of input m_ask_df extended with new columns for LLM
        predictions and related attributes
    '''

    # Set the style to be used for interacting with the LLM Oracle
    # for the current alignment task.  (Choice of API, choice of how
    # to use that API, approach to structured outputs, etc..)
    # For now, there is only one choice!
    # TODO: externalise this choice in the config.toml file; but in the
    # expert config.toml file, not the basic config file
    interaction_style_name = 'openai_chat_completions_parse_structured_output'

    # TODO: consider externalising all of these LLM Oracle config parameters,
    # but in a 2nd config.toml file --- a config file for detailed control
    # of LLM Oracle config, for experts; so we'll have 2 config files, one 
    # for basic config, one for expert config

    # TODO: we need more flexibility around configuring LLM interactions
    # a) not all LLM models support 'developer' (aka 'system') prompts; so
    #    perhaps the expert config.toml file needs to allow users to switch-off 
    #    use of 'developer' prompts
    # b) some OpenAI models, like o1-preview and o1-mini models only
    #    support temperature=1; ie if you do specify temperature, it must be 1.

    # assemble keyword arguments for configuring the LLM interaction
    # - reasoning_effort: none, minimal, low, medium, high, and xhigh
    kwargs = {
        "temperature": 0,
        "top_p": 1,
        "reasoning_effort": 'minimal',
        "max_completion_tokens": 1000,
        "response_format": BinaryOutputFormat # BinaryOutputFormatWithReasoning
    }

    # instantiate a fresh Oracle consultation manager to conduct the
    # Oracle consultations for the set of candidate mappings in m_ask
    llm_oracle = OracleConsultationManager_OpenAI(api_key, model_name, 
                                                  interaction_style_name, **kwargs)

    # TODO: externalise the choice of developer message in the config.toml file
    llm_oracle.add_developer_message(dp.DEV_PROMPT_CLASS_EQUIVALENCE)

    # container for LLM mapping predictions: 
    # [source, target, prediction, confidence]
    mapping_predictions = []

    # container for per-mapping (per consultation) token usage: 
    # (input_tokens, output_tokens)
    tokens_usage = []

    failure_count = 0
    # TODO: externalise failure tolerance and/or think of smarter policy
    failure_tolerance = 3
    abort_consultations = False

    # each Oracle consultation is synchronous; use a threadpool
    # to parallelise some number of synchronous consultations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for entity_pair, user_prompt in m_ask_oracle_user_prompts.items():
            futures.append(executor.submit(consult_oracle_for_mapping, 
                                           entity_pair, user_prompt, llm_oracle)
            )

        for future in tqdm(as_completed(futures), total=len(futures), 
                           desc="Oracle consultations"):
            # future.result() returns the outputs of the function being executed
            mapping_prediction, token_usage = future.result()

            # check for errors; if too many, abort consultation exercise
            if mapping_prediction[2] == "error":
                failure_count += 1
                if failure_count >= failure_tolerance:
                    print()
                    print('ABORTING Oracle consultations prematurely!')
                    print('Tolerance for Oracle consultation failures exceeded')
                    print('Pending consultations will be cancelled')
                    print('Running consultations will run to completion')
                    print()
                    abort_consultations = True
                    executor.shutdown(wait=True, cancel_futures=True)
                    break
                    # TODO: consider policies for aborting based on proportions
                    # of failures rather than a count of failures;
                    # if all consultations are failing, we want to abort fast;
                    # but if some consultations are succeeding, then we want to 
                    # abort more judiciously, and perhaps not at all; 
                    # IE conceive a smarter policy for managing the aborting of
                    # the consultations; the policy we have now is a blunt 
                    # instrument, perhaps too blunt
            
            # save the results
            mapping_predictions.append(mapping_prediction)
            tokens_usage.append(token_usage)

    
    if abort_consultations:
        return None

    #
    # record the LLM prediction information with their associated
    # mappings_to_ask in an extended dataframe
    #
    
    # associate the correct index with each entity pair
    # mapping_prediction: [src_entity_uri, tgt_entity_uri, prediction, confidence]
    entity_pair_to_idx = {(mp[0], mp[1]): idx for idx, mp in enumerate(mapping_predictions)}

    # containers for new dataframe columns
    ordered_llm_mapping_predictions = []
    ordered_llm_prediction_confidences = []
    ordered_token_usage_input = []
    ordered_token_usage_output = []

    # iterate over the mappings_to_ask 
    for row in m_ask_df.iterrows():
        # get the URIs of the two entities involved in the current mapping
        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]
        
        # get the correct LLM prediction for the current mapping_to_ask
        idx = entity_pair_to_idx[(src_entity_uri, tgt_entity_uri)]
        mp = mapping_predictions[idx]   # [source, target, prediction, confidence]
        
        # store the binary prediction (True/False) and prediction confidence 
        ordered_llm_mapping_predictions.append(mp[2])
        ordered_llm_prediction_confidences.append(mp[3])
        
        # order the token usage correspondingly
        ordered_token_usage_input.append(tokens_usage[idx][0])
        ordered_token_usage_output.append(tokens_usage[idx][1])

    # initialise an extended dataframe with a copy of the original
    m_ask_df_ext = m_ask_df.copy()

    # extend the mappings_to_ask an Oracle with columns for the 
    # corresponding Oracle (LLM) predictions and confidences
    m_ask_df_ext['Oracle_prediction'] = ordered_llm_mapping_predictions
    m_ask_df_ext['Oracle_confidence'] = ordered_llm_prediction_confidences

    # extend the mappings_to_ask an Oracle with columns for the
    # token usage associated with the LLM interactions
    m_ask_df_ext['Oracle_input_tokens'] = ordered_token_usage_input
    m_ask_df_ext['Oracle_output_tokens'] = ordered_token_usage_output

    return m_ask_df_ext

