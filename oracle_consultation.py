'''
This module contains functionality that supports consulting 
(interacting with) LLM Oracles.
'''
from __future__ import annotations

import functools
from typing import Any, Callable
from oracle_consultation_managers import OracleConsultationManager_OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from constants import BinaryOutputFormat, BinaryOutputFormatWithReasoning, YesNoOutputFormat, PAIRS_SEPARATOR, POSITIVE_TOKENS, NEGATIVE_TOKENS, RESPONSE_FORMAT_FOR_ANSWER
from tqdm import tqdm
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
        # converts string answer (Yes/No) to boolean
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

        # compute confidence from whichever of true/false was present
        probs = []
        if positive_logprob > float('-inf'):
            probs.append(np.exp(positive_logprob))
        if negative_logprob > float('-inf'):
            probs.append(np.exp(negative_logprob))

        return max(probs) if probs else float('nan')

    # no positive/negative token found in any position
    return float('nan')


@retry(max_retries=1)
def consult_oracle_for_mapping(entity_pair, prompt, llm_oracle,
                                developer_override=None,
                                per_query_few_shot=None):
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
    developer_override : str, optional
        If provided, replaces the developer/system message for this
        consultation only.  Used for entity-type-aware developer prompts.
    per_query_few_shot : list of (str, str) tuples, optional
        Per-query few-shot examples (hard-similar strategy).

    Returns
    -------
    mapping_prediction : list
        An LLM prediction for a mapping, formatted as
        [src_entity_uri, tgt_entity_uri, prediction, confidence]
    token_usage : tuple 
        A 2-tuple of (input tokens, output tokens) for the interaction
        with the LLM
    '''

    src_entity_uri, tgt_entity_uri = entity_pair.split('|')

    try:
        # consult an LLM Oracle regarding a candidate mapping
        response = llm_oracle.consult_oracle(
            prompt, developer_override,
            per_query_few_shot=per_query_few_shot,
        )
        
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
    Shared setup logic for oracle consultation functions.

    Parameters
    ----------
    m_ask_oracle_user_prompts : dict
        The prompts dict (used to compute failure thresholds from its size).
    api_key, model_name, max_workers, base_url, enable_thinking,
    interaction_style, developer_prompt_text, max_completion_tokens,
    failure_tolerance, temperature, top_p, reasoning_effort :
        See consult_oracle_for_mappings_to_ask for descriptions.
    mode : str
        'standard' or 'bidirectional' — recorded in oracle_params.

    Returns
    -------
    llm_oracle : OracleConsultationManager_OpenAI
        A configured and ready consultation manager.
    oracle_params : dict
        The effective configuration parameters for logging.
    effective_failure_tolerance : int
        The resolved cumulative failure threshold.
    """
    # resolve interaction style
    if interaction_style is None:
        interaction_style = 'auto'
    interaction_style_name = interaction_style

    # apply defaults for configurable parameters
    effective_max_tokens = max_completion_tokens if max_completion_tokens is not None else 1000

    # failure tolerance: abort if cumulative failures exceed 5% of total or a configured minimum, whichever is greater...
    n_total = len(m_ask_oracle_user_prompts)
    if failure_tolerance is not None:
        effective_failure_tolerance = max(failure_tolerance, int(n_total * 0.05))
    else:
        effective_failure_tolerance = max(5, int(n_total * 0.05))

    # LLM sampling & reasoning parameters
    effective_temperature = temperature if temperature is not None else 0
    effective_top_p = top_p if top_p is not None else 1
    effective_reasoning_effort = reasoning_effort if reasoning_effort is not None else 'minimal'

    # assemble kwargs for the consultation manager
    response_format = RESPONSE_FORMAT_FOR_ANSWER.get(answer_format, BinaryOutputFormat)
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

    # instantiate manager
    llm_oracle = OracleConsultationManager_OpenAI(api_key, model_name, interaction_style_name, **kwargs)

    # attach developer message if provided
    if developer_prompt_text is not None:
        llm_oracle.add_developer_message(developer_prompt_text)

    # inject few-shot examples if provided
    if few_shot_examples:
        llm_oracle.add_few_shot_examples(few_shot_examples)

    # capture effective configuration for logging
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


def _check_failure_abort(
    prediction_status,
    consecutive_failures,
    cumulative_failures,
    effective_failure_tolerance,
    consecutive_limit=5,
):
    """
    check whether oracle consultations should be aborted

    implements a dual-threshold policy:
    - **Consecutive failures:** If 5 failures in a row, abort.
    - **Cumulative failures:** If total failures reach effective_failure_tolerance
        (derived from 5% of total or config), abort.

    Parameters
    ----------
    prediction_status : str or bool
        The prediction value — checked for error.
    consecutive_failures : int
        Current consecutive failure count (before this result).
    cumulative_failures : int
        Current cumulative failure count (before this result).
    effective_failure_tolerance : int
        The cumulative threshold from _setup_oracle_consultation.
    consecutive_limit : int
        Maximum consecutive failures before abort (default 5).

    Returns
    -------
    should_abort : bool
    consecutive_failures : int (updated)
    cumulative_failures : int (updated)
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
        # Reset consecutive on success; cumulative is monotonic
        return False, 0, cumulative_failures


# TODO: consider patterns that simplify all of these params (use of unpacking, etc)
#       the amount of configurable parameters is still 'managable' but is it 
#       reasonable & maintainable to use this approach?
def consult_oracle_for_mappings_to_ask(m_ask_oracle_user_prompts, 
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
                                       few_shot_pool=None,
                                       pair_labels=None):
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
    base_url : str, optional
        Custom API base URL (e.g. for local vLLM)
    enable_thinking : bool, optional
        Control model thinking/CoT mode
    interaction_style : str, optional
        API interaction style: 'auto', 'openrouter', or 'vllm'.
        Defaults to 'auto' which discovers the right style on first call.
    developer_prompt_text : str or None, optional
        The developer/system prompt text to use for all consultations.
        If None, no developer message is added.
    developer_prompt_map : dict or None, optional
        Entity-type-aware developer prompts.  Maps entity type strings
        (e.g. 'CLS', 'OPROP', 'DPROP', 'INST') to developer prompt text.
        When provided and a mapping's entity type has an entry, that text
        overrides the default developer prompt for that consultation.
        If None, all consultations use the single developer_prompt_text.
    max_completion_tokens : int, optional
        Maximum number of tokens in the LLM completion response.
        Defaults to 1000.
    failure_tolerance : int, optional
        Cumulative oracle failure threshold before aborting.
        Defaults to max(5, 5% of total).
    temperature : float, optional
        LLM sampling temperature. Defaults to 0 (deterministic).
    top_p : float, optional
        LLM nucleus sampling top-p. Defaults to 1.
    reasoning_effort : str, optional
        LLM reasoning effort level. Defaults to 'minimal'.
        Options: none, minimal, low, medium, high, xhigh.

    Returns
    -------
    m_ask_df_ext : pandas DataFrame
        A copy of input m_ask_df extended with new columns for LLM
        predictions and related attributes
    oracle_params : dict
        The effective Oracle configuration parameters used for this run
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

    consecutive_failures = 0
    cumulative_failures = 0
    abort_consultations = False

    # freeze messages before multithreaded access
    llm_oracle.freeze_messages()

    # build entity_pair -> entityType lookup from m_ask_df for entity-type-aware developer prompt selection
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

            # per-query few-shot selection (hard-similar strategy)
            query_few_shot = None
            if few_shot_pool is not None and pair_labels is not None:
                labels = pair_labels.get(entity_pair)
                if labels:
                    src_uri, tgt_uri = entity_pair.split(PAIRS_SEPARATOR)
                    query_few_shot = few_shot_pool.select_for_query(
                        query_src_label=labels[0],
                        query_tgt_label=labels[1],
                        query_src_uri=src_uri,
                        query_tgt_uri=tgt_uri,
                    )

            futures.append(executor.submit(consult_oracle_for_mapping, 
                                           entity_pair, user_prompt, llm_oracle,
                                           developer_override=dev_override,
                                           per_query_few_shot=query_few_shot)
            )

        for future in tqdm(as_completed(futures), total=len(futures), 
                           desc="Oracle consultations"):
            # future.result() returns the outputs of the function being executed
            mapping_prediction, token_usage = future.result()

            # check for errors using dual-threshold policy
            should_abort, consecutive_failures, cumulative_failures = _check_failure_abort(
                mapping_prediction[2],
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
                print('Pending consultations will be cancelled')
                print('Running consultations will run to completion')
                print()
                abort_consultations = True
                executor.shutdown(wait=True, cancel_futures=True)
                break
            
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

    # iterate over the mappings_to_ask 
    skipped_count = 0
    for row in m_ask_df.iterrows():
        # get the URIs of the two entities involved in the current mapping
        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]
        
        # get the correct LLM prediction for the current mapping_to_ask
        # (some mappings may have been skipped during prompt building due
        # to unresolvable class URIs --- these won't have predictions)
        idx = entity_pair_to_idx.get((src_entity_uri, tgt_entity_uri))

        if idx is not None:
            mp = mapping_predictions[idx]   # [source, target, prediction, confidence]
            
            # store the binary prediction (True/False) and prediction confidence 
            ordered_llm_mapping_predictions.append(mp[2])
            ordered_llm_prediction_confidences.append(mp[3])
            
            # order the token usage correspondingly
            ordered_token_usage_input.append(tokens_usage[idx][0])
            ordered_token_usage_output.append(tokens_usage[idx][1])
        else:
            # Mapping was skipped during prompt building (e.g. class URI
            # not resolvable by owlready2). Mark as 'skipped' so it is
            # clearly distinguishable from 'error' (API failure) and from
            # True/False predictions in downstream processing.
            skipped_count += 1
            ordered_llm_mapping_predictions.append("skipped")
            ordered_llm_prediction_confidences.append(str(np.nan))
            ordered_token_usage_input.append(np.nan)
            ordered_token_usage_output.append(np.nan)

    if skipped_count > 0:
        print(f"Note: {skipped_count} mappings marked as 'skipped' "
              f"(no prompt built due to unresolvable class URIs).")

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


def _consult_oracle_for_directed_mapping(full_key, prompt, llm_oracle,
                                          developer_override=None,
                                          per_query_few_shot=None):
    '''
    Consult an LLM Oracle for one directed mapping (forward or reverse).

    Like consult_oracle_for_mapping but handles 3-part keys
    (src_uri|tgt_uri|direction) used by bidirectional subsumption prompts.

    Parameters
    ----------
    full_key : str
        'src_entity_uri|tgt_entity_uri|direction' where direction is
        'forward' or 'reverse'
    prompt : str
        A formatted user prompt for a particular directed mapping
    llm_oracle : OracleConsultationManager
        An OracleConsultationManager of some kind
    developer_override : str, optional
        If provided, replaces the developer/system message for this
        consultation only.
    per_query_few_shot : list of (str, str) tuples, optional
        Per-query few-shot examples (hard-similar strategy).

    Returns
    -------
    full_key : str
        The original key (passed through for result mapping)
    prediction : bool or str
        The LLM's prediction (True/False/error)
    confidence : float or str
        The prediction confidence
    token_usage : tuple
        A 2-tuple of (input tokens, output tokens)
    '''
    try:
        response = llm_oracle.consult_oracle(
            prompt, developer_override,
            per_query_few_shot=per_query_few_shot,
        )
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


def consult_oracle_bidirectional(m_ask_oracle_user_prompts,
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
                                  few_shot_pool=None,
                                  pair_labels=None):
    '''
    Consult an LLM Oracle with bidirectional subsumption prompts and aggregate 
    forward/reverse results into per-candidate equivalence predictions

    This function handles the full bidirectional workflow:
    
    1. Sends all prompts (forward + reverse) to the LLM
    2. Collects individual directional predictions
    3. Aggregates into per-candidate equivalence decisions using AND rule:
       predict equivalence only if BOTH directions return True

    The returned m_ask_df_ext has the same schema as the standard
    consult_oracle_for_mappings_to_ask output, ensuring downstream
    compatibility with bridging.py and LogMap refinement

    Parameters
    ----------
    m_ask_oracle_user_prompts : dict
        Bidirectional prompts with direction-annotated keys:
        'src_uri|tgt_uri|forward' and 'src_uri|tgt_uri|reverse'
    api_key : str
        An LLM API key
    model_name : str
        An LLM model name
    max_workers : int
        The maximum number of Oracle consultations to run in parallel
    m_ask_df : pandas DataFrame
        The original LogMap mappings_to_ask (used for result alignment)
    base_url : str, optional
        Custom API base URL (e.g. for local vLLM)
    enable_thinking : bool, optional
        Control model thinking/CoT mode
    interaction_style : str, optional
        API interaction style: 'auto', 'openrouter', or 'vllm'
    developer_prompt_text : str or None, optional
        The developer/system prompt text
    max_completion_tokens : int, optional
        Maximum number of tokens in the LLM completion response.
        Defaults to 1000.
    failure_tolerance : int, optional
        Number of consecutive oracle failures before aborting.
        Defaults to 3.
    temperature : float, optional
        LLM sampling temperature. Defaults to 0 (deterministic).
    top_p : float, optional
        LLM nucleus sampling top-p. Defaults to 1.
    reasoning_effort : str, optional
        LLM reasoning effort level. Defaults to 'minimal'.
        Options: none, minimal, low, medium, high, xhigh.

    Returns
    -------
    m_ask_df_ext : pandas DataFrame or None
        m_ask_df extended with columns:
        - Oracle_prediction_forward : bool or str
        - Oracle_prediction_reverse : bool or str
        - Oracle_confidence_forward : float or str
        - Oracle_confidence_reverse : float or str
        - Oracle_prediction : bool (aggregated: True iff both True)
        - Oracle_confidence : float (min of forward and reverse)
        - Oracle_input_tokens : int (sum of forward and reverse)
        - Oracle_output_tokens : int (sum of forward and reverse)
        Returns None if consultations were aborted due to failures.
    oracle_params : dict
        The effective Oracle configuration parameters
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
        mode="bidirectional",
        few_shot_examples=few_shot_examples,
        answer_format=answer_format,
    )

    # TODO: refactor to remove duplicated code

    # phase 1: send all directional prompts to the LLM

    # store results keyed by the full direction-annotated key
    results = {}
    consecutive_failures = 0
    cumulative_failures = 0
    abort_consultations = False

    # freeze messages before multithreaded access
    llm_oracle.freeze_messages()

    # build entity_pair -> entityType lookup for developer prompt selection
    pair_entity_types = {}
    if developer_prompt_map:
        for _, row in m_ask_df.iterrows():
            base_key = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
            etype = str(row.iloc[4]).strip() if len(row) > 4 else "CLS"
            pair_entity_types[base_key] = etype

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for full_key, user_prompt in m_ask_oracle_user_prompts.items():
            # Look up per-mapping developer prompt override
            dev_override = None
            if developer_prompt_map:
                # full_key is 'src|tgt|direction'; extract base pair
                parts = full_key.split(PAIRS_SEPARATOR)
                base_key = PAIRS_SEPARATOR.join(parts[:2])
                etype = pair_entity_types.get(base_key, "CLS")
                dev_override = developer_prompt_map.get(etype)

            # per-query few-shot selection (hard-similar strategy)
            query_few_shot = None
            if few_shot_pool is not None and pair_labels is not None:
                # extract base pair key (strip direction suffix)
                parts = full_key.split(PAIRS_SEPARATOR)
                base_key = PAIRS_SEPARATOR.join(parts[:2])
                labels = pair_labels.get(base_key)
                if labels:
                    src_uri, tgt_uri = parts[0], parts[1]
                    query_few_shot = few_shot_pool.select_for_query(
                        query_src_label=labels[0],
                        query_tgt_label=labels[1],
                        query_src_uri=src_uri,
                        query_tgt_uri=tgt_uri,
                    )

            futures.append(executor.submit(
                _consult_oracle_for_directed_mapping,
                full_key, user_prompt, llm_oracle,
                developer_override=dev_override,
                per_query_few_shot=query_few_shot,
            ))

        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Oracle consultations (bidirectional)"):
            full_key, prediction, confidence, token_usage = future.result()

            # check for errors using dual-threshold policy
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

    # phase 2: aggregate forward/reverse into per-candidate results

    # group results by base entity pair (strip direction suffix)
    pair_results = {}
    for full_key, result in results.items():
        parts = full_key.split(PAIRS_SEPARATOR)
        # parts: [src_uri, tgt_uri, direction]
        base_key = PAIRS_SEPARATOR.join(parts[:2])
        direction = parts[2] if len(parts) > 2 else "forward"

        if base_key not in pair_results:
            pair_results[base_key] = {}
        pair_results[base_key][direction] = result

    # phase 3: map aggregated results back to m_ask_df rows

    col_pred_fwd = []
    col_pred_rev = []
    col_conf_fwd = []
    col_conf_rev = []
    col_pred_agg = []
    col_conf_agg = []
    col_tokens_in = []
    col_tokens_out = []

    skipped_count = 0

    for row in m_ask_df.iterrows():
        row_series = row[1]
        src_entity_uri, tgt_entity_uri = row_series.iloc[0], row_series.iloc[1]
        base_key = src_entity_uri + PAIRS_SEPARATOR + tgt_entity_uri

        pair = pair_results.get(base_key)

        if pair is None:
            # this candidate was skipped (e.g., unresolvable relation / class URI)
            skipped_count += 1
            col_pred_fwd.append("skipped")
            col_pred_rev.append("skipped")
            col_conf_fwd.append(str(np.nan))
            col_conf_rev.append(str(np.nan))
            col_pred_agg.append("skipped")
            col_conf_agg.append(str(np.nan))
            col_tokens_in.append(np.nan)
            col_tokens_out.append(np.nan)
            continue

        fwd = pair.get("forward", {})
        rev = pair.get("reverse", {})

        fwd_pred = fwd.get("prediction", "skipped")
        rev_pred = rev.get("prediction", "skipped")
        fwd_conf = fwd.get("confidence", str(np.nan))
        rev_conf = rev.get("confidence", str(np.nan))
        fwd_tokens = fwd.get("token_usage", (np.nan, np.nan))
        rev_tokens = rev.get("token_usage", (np.nan, np.nan))

        col_pred_fwd.append(fwd_pred)
        col_pred_rev.append(rev_pred)
        col_conf_fwd.append(fwd_conf)
        col_conf_rev.append(rev_conf)

        # aggregate: AND rule — equivalence only if BOTH directions True
        if fwd_pred is True and rev_pred is True:
            agg_pred = True
        elif fwd_pred == "error" or rev_pred == "error":
            agg_pred = "error"
        elif fwd_pred == "skipped" or rev_pred == "skipped":
            agg_pred = "skipped"
        else:
            agg_pred = False

        col_pred_agg.append(agg_pred)

        # aggregated confidence: min of forward and reverse (both directions must be confident for equivalence)
        try:
            fwd_conf_f = float(fwd_conf)
            rev_conf_f = float(rev_conf)
            agg_conf = min(fwd_conf_f, rev_conf_f)
        except (ValueError, TypeError):
            agg_conf = str(np.nan)
        col_conf_agg.append(agg_conf)

        # token usage: sum of both directions
        try:
            tokens_in = (fwd_tokens[0] or 0) + (rev_tokens[0] or 0)
            tokens_out = (fwd_tokens[1] or 0) + (rev_tokens[1] or 0)
        except TypeError:
            tokens_in = np.nan
            tokens_out = np.nan
        col_tokens_in.append(tokens_in)
        col_tokens_out.append(tokens_out)

    if skipped_count > 0:
        print(f"Note: {skipped_count} mappings marked as 'skipped' "
              f"(non-equivalence relation or unresolvable class URIs).")

    # build extended DataFrame
    import pandas as pd
    m_ask_df_ext = m_ask_df.copy()
    m_ask_df_ext['Oracle_prediction_forward'] = col_pred_fwd
    m_ask_df_ext['Oracle_prediction_reverse'] = col_pred_rev
    m_ask_df_ext['Oracle_confidence_forward'] = col_conf_fwd
    m_ask_df_ext['Oracle_confidence_reverse'] = col_conf_rev
    m_ask_df_ext['Oracle_prediction'] = col_pred_agg
    m_ask_df_ext['Oracle_confidence'] = col_conf_agg
    m_ask_df_ext['Oracle_input_tokens'] = col_tokens_in
    m_ask_df_ext['Oracle_output_tokens'] = col_tokens_out

    # print aggregation summary
    n_both_true = sum(1 for p in col_pred_agg if p is True)
    n_at_least_one_false = sum(1 for p in col_pred_agg if p is False)
    n_error = sum(1 for p in col_pred_agg if p == "error")
    n_skipped = sum(1 for p in col_pred_agg if p == "skipped")
    print()
    print("Bidirectional aggregation summary:")
    print(f"  Both True  (-> equivalence) : {n_both_true}")
    print(f"  At least one False (-> not)  : {n_at_least_one_false}")
    print(f"  Errors                       : {n_error}")
    print(f"  Skipped                      : {n_skipped}")
    print()

    return m_ask_df_ext, oracle_params