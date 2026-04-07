'''
This module contains functionality that supports consulting 
(interacting with) LLM Oracles.
'''
from __future__ import annotations

import functools
from typing import Any, Callable
from logmap_llm.oracle.manager import OracleConsultationManager_OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from logmap_llm.constants import (
    BinaryOutputFormat,
    BinaryOutputFormatWithReasoning,
    YesNoOutputFormat,
    PAIRS_SEPARATOR,
    POSITIVE_TOKENS,
    NEGATIVE_TOKENS,
    RESPONSE_FORMAT_FOR_ANSWER,
)
from tqdm import tqdm
import time
import numpy as np
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
    if isinstance(response.parsed, YesNoOutputFormat):
        answer_str = response.parsed.answer.strip().lower()
        if answer_str in POSITIVE_TOKENS:
            return True
        elif answer_str in NEGATIVE_TOKENS:
            return False
        else:
            raise ValueError(f"YesNoOutputFormat answer not recognised: '{response.parsed.answer}'")
    raise NotImplementedError()


def calculate_logprobs_confidence(log_probs: list) -> float:
    if not log_probs:
        return float('nan')

    for token_info in log_probs:
        token_text = token_info["token"].strip().lower()
        if token_text not in POSITIVE_TOKENS and token_text not in NEGATIVE_TOKENS:
            continue

        top_logprobs = token_info.get("top_logprobs", [])

        positive_logprob = float('-inf')
        negative_logprob = float('-inf')

        for entry in top_logprobs:
            entry_token = entry["token"].strip().lower()
            if entry_token in POSITIVE_TOKENS:
                positive_logprob = max(positive_logprob, entry["logprob"])
            elif entry_token in NEGATIVE_TOKENS:
                negative_logprob = max(negative_logprob, entry["logprob"])

        probs = []
        if positive_logprob > float('-inf'):
            probs.append(np.exp(positive_logprob))
        if negative_logprob > float('-inf'):
            probs.append(np.exp(negative_logprob))

        return max(probs) if probs else float('nan')

    # no relevant token found in any position
    return float('nan')


def _check_failure_abort(
    prediction_status,
    consecutive_failures,
    cumulative_failures,
    effective_failure_tolerance,
    consecutive_limit=5,
):
    """
    check whether oracle consultations should be aborted
    """
    if prediction_status == "error":
        consecutive_failures += 1
        cumulative_failures += 1
        should_abort = (
            consecutive_failures >= consecutive_limit
            or cumulative_failures >= effective_failure_tolerance
        )
        return should_abort, consecutive_failures, cumulative_failures
    else:
        # reset consecutive on success
        return False, 0, cumulative_failures


def _setup_oracle_consultation(
    m_ask_oracle_user_prompts,
    api_key,
    model_name,
    max_workers,
    base_url=None,
    enable_thinking=None,
    interaction_style=None,
    supports_chat_template_kwargs=None,
    developer_prompt_text=None,
    max_completion_tokens=None,
    failure_tolerance=None,
    temperature=None,
    top_p=None,
    reasoning_effort=None,
    mode="standard",
    few_shot_examples=None,
    answer_format="true_false",
):
    """
    shared setup logic for oracle consultation functions
    """

    # Set the style to be used for interacting with the LLM Oracle
    # for the current alignment task.  (Choice of API, choice of how
    # to use that API, approach to structured outputs, etc..)

    # resolve interaction style
    if interaction_style is None:
        interaction_style = 'auto'
    
    interaction_style_name = interaction_style

    # apply defaults for configurable parameters (TODO: allow modifable upper bound default)
    effective_max_tokens = max_completion_tokens if max_completion_tokens is not None else 1000

    # failure tolerance:
    # TODO: extract failure tolerance to its own private method implementation
    n_total = len(m_ask_oracle_user_prompts)
    if failure_tolerance is not None:
        effective_failure_tolerance = max(failure_tolerance, int(n_total * 0.05))
    else:
        effective_failure_tolerance = max(5, int(n_total * 0.05))

    # llm sampling & reasoning parameters
    effective_temperature = temperature if temperature is not None else 0
    effective_top_p = top_p if top_p is not None else 1
    effective_reasoning_effort = reasoning_effort if reasoning_effort is not None else 'minimal'

    ###
    # construct kwargs for the consultation manager
    ###

    response_format = RESPONSE_FORMAT_FOR_ANSWER.get(answer_format, BinaryOutputFormat)

    # most LLM orcale parameters have been externalised
    # TODO: use two config files for detailed control of LLM Oracle config
    # one for exprts, and one for basic users  (one for basic config, one for expert config)

    # TODO: we need more flexibility around configuring LLM interactions
    #
    # a) not all LLM models support 'developer' (aka 'system') prompts; so
    #    perhaps the expert config.toml file needs to allow users to switch-off 
    #    use of 'developer' prompts
    #
    # b) some OpenAI models, like o1-preview and o1-mini models only
    #    support temperature=1; ie if you do specify temperature, it must be 1.

    # assemble keyword arguments for configuring the LLM interaction

    kwargs = {
        "temperature": effective_temperature,
        "top_p": effective_top_p,
        "reasoning_effort": effective_reasoning_effort,
        "max_completion_tokens": effective_max_tokens,
        "response_format": response_format,
    }

    if base_url:
        kwargs["base_url"] = base_url

    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking

    if supports_chat_template_kwargs is not None:
        kwargs["supports_chat_template_kwargs"] = supports_chat_template_kwargs

    # instantiate a fresh Oracle consultation manager to conduct the
    # Oracle consultations for the set of candidate mappings in m_ask
    llm_oracle = OracleConsultationManager_OpenAI(
        api_key, model_name, interaction_style_name, **kwargs
    )

    # externalises the choice of developer message in the config.toml file
    if developer_prompt_text is not None:
        llm_oracle.add_developer_message(developer_prompt_text)

    # TODO: few-shot RAG-based implementation
    if few_shot_examples:
       llm_oracle.add_few_shot_examples(few_shot_examples)

    oracle_params = {
        "model_name": model_name,
        "base_url": base_url,
        "interaction_style": interaction_style_name,
        "temperature": kwargs.get("temperature"),
        "top_p": kwargs.get("top_p"),
        "reasoning_effort": kwargs.get("reasoning_effort"),
        "max_completion_tokens": kwargs.get("max_completion_tokens"),
        "enable_thinking": kwargs.get("enable_thinking"),
        "supports_chat_template_kwargs": llm_oracle.supports_chat_template_kwargs,
        "response_format": kwargs.get("response_format", "N/A"),
        "developer_prompt_set": developer_prompt_text is not None,
        "max_workers": max_workers,
        "failure_tolerance": effective_failure_tolerance,
        "few_shot_examples": len(few_shot_examples) if few_shot_examples else 0,
        "mode": mode,
    }

    return llm_oracle, oracle_params, effective_failure_tolerance

@retry(max_retries=1)
def consult_oracle_for_mapping(entity_pair, prompt, llm_oracle, developer_override=None):
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

    src_entity_uri, tgt_entity_uri = entity_pair.split(PAIRS_SEPARATOR) # '|'

    try:
        if sleep_time > 0:
            time.sleep(sleep_time)

        # consult an LLM Oracle regarding a candidate mapping
        response = llm_oracle.consult_oracle(prompt, developer_override)
        
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
        try:
            parts1 = bre.message.split("'message':")
            error_msg = parts1[1].split(",")[0].strip()
            print(f'BadRequestError: {error_msg}')
        except (IndexError, AttributeError):
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


def consult_oracle_for_mappings_to_ask(
    m_ask_oracle_user_prompts,
    api_key,
    model_name,
    max_workers,
    m_ask_df,
    base_url=None,
    enable_thinking=None,
    interaction_style=None,
    supports_chat_template_kwargs=None,
    developer_prompt_text=None,
    developer_prompt_map=None,
    max_completion_tokens=None,
    failure_tolerance=None,
    temperature=None,
    top_p=None,
    reasoning_effort=None,
    few_shot_examples=None,
    answer_format="true_false",
):                               
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

    llm_oracle, oracle_params, effective_failure_tolerance = _setup_oracle_consultation(
        m_ask_oracle_user_prompts=m_ask_oracle_user_prompts,
        api_key=api_key,
        model_name=model_name,
        max_workers=max_workers,
        base_url=base_url,
        enable_thinking=enable_thinking,
        interaction_style=interaction_style,
        supports_chat_template_kwargs=supports_chat_template_kwargs,
        developer_prompt_text=developer_prompt_text,
        max_completion_tokens=max_completion_tokens,
        failure_tolerance=failure_tolerance,
        temperature=temperature,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        mode="standard",
        few_shot_examples=few_shot_examples,
        answer_format=answer_format,
    )

    # container for LLM mapping predictions: 
    # [source, target, prediction, confidence]
    mapping_predictions = []

    # container for per-mapping (per consultation) token usage: 
    # (input_tokens, output_tokens)
    tokens_usage = []

    # TODO: externalise failure tolerance and/or think of smarter policy
    # TODO: consider implementing the failure tolerance as is discussed, 
    # as a policy of some kind that can be easily switched out, at present,
    # the policy is hardcoded

    consecutive_failures = 0
    cumulative_failures = 0
    abort_consultations = False
    
    # freeze messages before multithreaded access
    llm_oracle.freeze_messages()

    # build entity_pair -> entityType lookup for developer prompt selection
    pair_entity_types = {}
    if developer_prompt_map:
        for _, row in m_ask_df.iterrows():
            pair_key = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
            etype = str(row.iloc[4]).strip() if len(row) > 4 else "CLS"
            pair_entity_types[pair_key] = etype
    
    # each Oracle consultation is synchronous; use a threadpool
    # to parallelise some number of synchronous consultations
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for entity_pair, user_prompt in m_ask_oracle_user_prompts.items():

            # look up per-mapping developer prompt override
            dev_override = None
            if developer_prompt_map:
                etype = pair_entity_types.get(entity_pair, "CLS")
                dev_override = developer_prompt_map.get(etype)

            futures.append(
                executor.submit(
                    consult_oracle_for_mapping,
                    entity_pair, user_prompt, llm_oracle,
                    developer_override=dev_override,
                )
            )

        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc="Oracle consultations",
        ):
            # future.result() returns the outputs of the function being executed
            mapping_prediction, token_usage = future.result()

            # dual-threshold failure counting
            should_abort, consecutive_failures, cumulative_failures = _check_failure_abort(
                mapping_prediction[2],
                consecutive_failures,
                cumulative_failures,
                effective_failure_tolerance,
            )

            # check for errors; if too many, abort consultation exercise
            if should_abort:
                print()
                print('ABORTING Oracle consultations prematurely!')
                if consecutive_failures >= 5:
                    print(f'  {consecutive_failures} consecutive failures — likely a systematic issue')
                print(f'  {cumulative_failures} cumulative failures (threshold: {effective_failure_tolerance})')
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

                # TODO: ^^^ mostly implemented, though, could perhaps be a
                # little more intuitive as to how it works, plus remove some
                # hardcoded values (?)
            
            # save the results
            mapping_predictions.append(mapping_prediction)
            tokens_usage.append(token_usage)


    if abort_consultations:
        return None, oracle_params

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

    skipped_count = 0

    # iterate over the mappings_to_ask 
    for row in m_ask_df.iterrows():
        # get the URIs of the two entities involved in the current mapping
        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]
        
        # get the correct LLM prediction for the current mapping_to_ask
        idx = entity_pair_to_idx.get((src_entity_uri, tgt_entity_uri))
        if idx is not None:
            mp = mapping_predictions[idx]
            ordered_llm_mapping_predictions.append(mp[2])
            ordered_llm_prediction_confidences.append(mp[3])
            ordered_token_usage_input.append(tokens_usage[idx][0])
            ordered_token_usage_output.append(tokens_usage[idx][1])
        else:
            skipped_count += 1
            ordered_llm_mapping_predictions.append("skipped")
            ordered_llm_prediction_confidences.append(str(np.nan))
            ordered_token_usage_input.append(np.nan)
            ordered_token_usage_output.append(np.nan)
        
        # mp = mapping_predictions[idx]   # [source, target, prediction, confidence]
        
        # store the binary prediction (True/False) and prediction confidence 
        # ordered_llm_mapping_predictions.append(mp[2])
        # ordered_llm_prediction_confidences.append(mp[3])
        
        # order the token usage correspondingly
        # ordered_token_usage_input.append(tokens_usage[idx][0])
        # ordered_token_usage_output.append(tokens_usage[idx][1])

    if skipped_count > 0:
        print(f"[WARNING] {skipped_count} mappings were skipped.")

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

    return m_ask_df_ext, oracle_params



#####
# DIRECTIONAL / BIDIRECTIONAL IMPLEMENTATION
# ------------------------------------------
# Largely mirrors the above logic, which is still
# quite similar/faithful to the original implementation
#####

def _consult_oracle_for_directed_mapping(full_key, prompt, llm_oracle, developer_override=None):
    
    try:
        response = llm_oracle.consult_oracle(prompt, developer_override)
        prediction = get_llm_mapping_prediction(response)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        token_usage = (input_tokens, output_tokens)
        confidence = calculate_logprobs_confidence(response.logprobs)

        return full_key, prediction, confidence, token_usage

    except BadRequestError as bre:
        print(f'ERROR during Oracle consultation for mapping: {full_key}')
        try:
            parts1 = bre.message.split("'message':")
            error_msg = parts1[1].split(",")[0].strip()
            print(f'BadRequestError: {error_msg}')
        except (IndexError, AttributeError):
            print(f'BadRequestError: {bre}')
        print()
        return full_key, "error", str(np.nan), (np.nan, np.nan)

    except Exception as e:
        print(f'ERROR during Oracle consultation for mapping: {full_key}')
        print(f'Exception: {e}')
        return full_key, "error", str(np.nan), (np.nan, np.nan)


def consult_oracle_bidirectional(
    m_ask_oracle_user_prompts,
    api_key,
    model_name,
    max_workers,
    m_ask_df,
    base_url=None,
    enable_thinking=None,
    interaction_style=None,
    supports_chat_template_kwargs=None,
    developer_prompt_text=None,
    developer_prompt_map=None,
    max_completion_tokens=None,
    failure_tolerance=None,
    temperature=None,
    top_p=None,
    reasoning_effort=None,
    few_shot_examples=None,
    answer_format="true_false",
):
    """
    Consult an LLM Oracle with bidirectional subsumption prompts
    ------------------------------------------------------------
    Sends forward and reverse subsumption queries, 
    then aggregates into per-candidate equivalence predictions.
    A candidate is predicted as equivalent (True) if and only if 
    both forward and reverse subsumption hold (otherwise False).
    """

    llm_oracle, oracle_params, effective_failure_tolerance = _setup_oracle_consultation(
        m_ask_oracle_user_prompts=m_ask_oracle_user_prompts,
        api_key=api_key,
        model_name=model_name,
        max_workers=max_workers,
        base_url=base_url,
        enable_thinking=enable_thinking,
        interaction_style=interaction_style,
        supports_chat_template_kwargs=supports_chat_template_kwargs,
        developer_prompt_text=developer_prompt_text,
        max_completion_tokens=max_completion_tokens,
        failure_tolerance=failure_tolerance,
        temperature=temperature,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        mode="bidirectional",
        few_shot_examples=few_shot_examples,
        answer_format=answer_format,
    )

    # TODO: eliminate duplicate code between methods
    #       (this process should be simpler)

    results = {}
    consecutive_failures = 0
    cumulative_failures = 0
    abort_consultations = False

    llm_oracle.freeze_messages()

    pair_entity_types = {}
    if developer_prompt_map:
        for _, row in m_ask_df.iterrows():
            base_key = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
            etype = str(row.iloc[4]).strip() if len(row) > 4 else "CLS"
            pair_entity_types[base_key] = etype

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for full_key, user_prompt in m_ask_oracle_user_prompts.items():
            dev_override = None
            if developer_prompt_map:
                parts = full_key.split(PAIRS_SEPARATOR)
                base_key = PAIRS_SEPARATOR.join(parts[:2])
                etype = pair_entity_types.get(base_key, "CLS")
                dev_override = developer_prompt_map.get(etype)

            futures.append(executor.submit(
                _consult_oracle_for_directed_mapping,
                full_key, user_prompt, llm_oracle,
                developer_override=dev_override,
            ))

        for future in tqdm(as_completed(futures), total=len(futures), desc="Oracle consultations (bidirectional)"):
            
            full_key, prediction, confidence, token_usage = future.result()

            should_abort, consecutive_failures, cumulative_failures = _check_failure_abort(
                prediction,
                consecutive_failures,
                cumulative_failures,
                effective_failure_tolerance,
            )

            if should_abort:
                print()
                print('ABORTING Oracle consultations prematurely!')
                if consecutive_failures >= 5:
                    print(f'  {consecutive_failures} consecutive failures — likely a systematic issue')
                print(f'  {cumulative_failures} cumulative failures (threshold: {effective_failure_tolerance})')
                print()
                abort_consultations = True
                executor.shutdown(wait=True, cancel_futures=True)
                break

            results[full_key] = {
                "prediction": prediction,
                "confidence": confidence,
                "token_usage": token_usage,
            }

    if abort_consultations:
        return None, oracle_params

    ###
    # Aggregate forward/reverse into per-candidate results
    ###

    pair_results = {}

    for full_key, result in results.items():
        parts = full_key.split(PAIRS_SEPARATOR)
        base_key = PAIRS_SEPARATOR.join(parts[:2])
        # normalise direction to lowercase
        # prompt keys use "REVERSE" (uppercase) suffix 
        # but we need consistent lookup keys
        direction = parts[2].lower() if len(parts) > 2 else "forward"

        if base_key not in pair_results:
            pair_results[base_key] = {}

        pair_results[base_key][direction] = result

    # map aggregated results back to m_ask_df rows
    ordered_llm_mapping_predictions = []
    ordered_llm_prediction_confidences = []
    ordered_token_usage_input = []
    ordered_token_usage_output = []
    skipped_count = 0

    for row in m_ask_df.iterrows():
        
        row_series = row[1]
        src_uri = str(row_series.iloc[0])
        tgt_uri = str(row_series.iloc[1])
        base_key = src_uri + PAIRS_SEPARATOR + tgt_uri

        if base_key in pair_results:
            
            pr = pair_results[base_key]
            fwd = pr.get("forward", {})
            rev = pr.get("reverse", {})
            fwd_pred = fwd.get("prediction", "error")
            rev_pred = rev.get("prediction", "error")

            # equivalence via mutual subsumption: true iff both hold

            if fwd_pred == "error" or rev_pred == "error":
                agg_pred = "error"
            else:
                agg_pred = bool(fwd_pred) and bool(rev_pred)

            # confidence is the minimum of the two directional confidences

            fwd_conf = fwd.get("confidence", float('nan'))
            rev_conf = rev.get("confidence", float('nan'))
            try:
                agg_conf = min(float(fwd_conf), float(rev_conf))
            except (ValueError, TypeError):
                agg_conf = float('nan')

            # token usage is the sum of both directions

            fwd_tu = fwd.get("token_usage", (0, 0))
            rev_tu = rev.get("token_usage", (0, 0))
            total_in = (fwd_tu[0] or 0) + (rev_tu[0] or 0)
            total_out = (fwd_tu[1] or 0) + (rev_tu[1] or 0)

            ordered_llm_mapping_predictions.append(agg_pred)
            ordered_llm_prediction_confidences.append(agg_conf)
            ordered_token_usage_input.append(total_in)
            ordered_token_usage_output.append(total_out)
        else:
            skipped_count += 1
            ordered_llm_mapping_predictions.append("skipped")
            ordered_llm_prediction_confidences.append(str(np.nan))
            ordered_token_usage_input.append(np.nan)
            ordered_token_usage_output.append(np.nan)

    if skipped_count > 0:
        print(f"Note: {skipped_count} mappings marked as 'skipped'.")

    m_ask_df_ext = m_ask_df.copy()
    m_ask_df_ext['Oracle_prediction'] = ordered_llm_mapping_predictions
    m_ask_df_ext['Oracle_confidence'] = ordered_llm_prediction_confidences
    m_ask_df_ext['Oracle_input_tokens'] = ordered_token_usage_input
    m_ask_df_ext['Oracle_output_tokens'] = ordered_token_usage_output

    return m_ask_df_ext, oracle_params
