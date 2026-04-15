'''
logmap_llm.oracle.consultation
This module contains functionality that supports consulting 
(interacting with) LLM Oracles.
'''
from __future__ import annotations

import functools
from typing import Any, Callable
from logmap_llm.oracle.manager import OracleConsultationManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from logmap_llm.config.schema import OracleConfig
from logmap_llm.constants import (
    BinaryOutputFormat,
    BinaryOutputFormatWithReasoning,
    YesNoOutputFormat,
    YesNoOutputFormatWithReasoning,
    TokensUsage,
    PAIRS_SEPARATOR,
    POSITIVE_TOKENS,
    NEGATIVE_TOKENS,
    DEFAULT_FAILURE_TOLERANCE_FLOOR,
    DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE,
    VERBOSE,
    VERY_VERBOSE,
)
from logmap_llm.utils.logging import (
    info,
    warn,
    warning,
    critical,
    debug,
)
from tqdm import tqdm
import numpy as np
from openai import BadRequestError
import pandas as pd


def retry(max_retries: int = 2, on: tuple = (Exception,), on_final_failure=None):
    """
    Retry the function up to ``max_retries`` times if an exception occurs.
    NOTE: this applies a decorator which wraps the function within a closure that
    reinvokes it upon failure, up until max_retries has been reached. It returns
    the first successful result, or re-raises the last exception if all attempts fail
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
            attempts = 0
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt < max_retries:
                        if VERBOSE:
                            debug(f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {type(e).__name__}: {e}")
                        continue
                    if on_final_failure is not None:
                        return on_final_failure(func, args, kwargs, e)
                    raise e
        return wrapper
    return decorator


def _consultation_failure_record(func, args, kwargs, exc):
    key = args[0] if args else kwargs.get("key", "<unknown>")
    warning(f"Consultation for {key} exhausted retries: {type(exc).__name__}: {exc}")
    return key, "error", float('nan'), TokensUsage(input_tokens=None, output_tokens=None)


def _debug_conversation_history(llm_oracle):
    print()
    debug("Conversation History:")
    for n_message, message in enumerate(llm_oracle.messages):
        debug(f"Message {n_message}: {message}")
    print()


def get_llm_mapping_prediction(response):
    """Extract the boolean prediction from a structured LLM response."""
    if isinstance(response.parsed, (BinaryOutputFormat, BinaryOutputFormatWithReasoning)):
        return response.parsed.answer
    if isinstance(response.parsed, (YesNoOutputFormat, YesNoOutputFormatWithReasoning)):
        answer_str = response.parsed.answer.strip().lower()
        if answer_str in POSITIVE_TOKENS:
            return True
        elif answer_str in NEGATIVE_TOKENS:
            return False
        else:
            raise ValueError(f"YesNoOutputFormat answer not recognised: '{response.parsed.answer}'")
    raise NotImplementedError()


def calculate_logprobs_confidence(log_probs: list) -> float:
    """
    Extract confidence from logprobs for a binary true/false prediction.
    Searches token logprobs for the true/false decision token, handling
    both bare tokens ("true", "false") and tokens with leading whitespace
    (" true", " false") as (possibly) produced by structured JSON output;
    will return the maximum probability between the true and false options,
    representing the model's confidence in whichever prediction it made.
    Returns NaN if logprobs are unavailable or contain no true/false token.
    """
    if not log_probs:
        if VERBOSE:
            debug("(calculate_logprobs_confidence) invalid argument, returning NaN (cast to float).")
        return float('nan')

    for token_info in log_probs:
        token_text = token_info["token"].strip().lower()
        if token_text not in POSITIVE_TOKENS and token_text not in NEGATIVE_TOKENS:
            if VERBOSE and VERY_VERBOSE:
                debug("(calculate_logprobs_confidence) token_text not in POSITIVE_TOKENS or NEGATIVE_TOKENS.")
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

    if VERBOSE:
        debug("(calculate_logprobs_confidence) No relevant tokens found at any position.")
    return float('nan')


def _check_failure_abort(prediction_status, consecutive_failures, cumulative_failures, failure_tolerance, 
                         consecutive_limit=DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE) -> tuple[bool, int, int]:
    if prediction_status != "error":
        return False, 0, cumulative_failures
    # else (an error has occured):
    warn("A consultation failure has been encountered.")
    consecutive_failures += 1
    cumulative_failures += 1
    should_abort: bool = (consecutive_failures >= consecutive_limit or cumulative_failures >= failure_tolerance)
    return should_abort, consecutive_failures, cumulative_failures


def _resolve_failure_tolerance(oracle_cfg: OracleConfig, n_total: int) -> int:
    """failure tolerance / abort policy"""
    failure_tolerance_floor = DEFAULT_FAILURE_TOLERANCE_FLOOR
    if oracle_cfg.failure_tolerance is not None:
        failure_tolerance_floor = oracle_cfg.failure_tolerance
    effective_faulure_tolerance = max(failure_tolerance_floor, int(n_total * 0.05))
    if VERBOSE:
        debug(f"Effective failure tolerance is set to: {effective_faulure_tolerance}.")
    return effective_faulure_tolerance


def _print_abort_message(consecutive_failures: int, cumulative_failures: int, failure_tolerance: int) -> None:
    critical("\nABORTING Oracle consultations prematurely! Error report:")
    if consecutive_failures >= DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE:
        critical(f"  {consecutive_failures} consecutive failures. Check your system setup.")
    critical(f"  {cumulative_failures} cumulative failures (threshold: {failure_tolerance})")
    warning("Pending consultations will be cancelled.")
    warning("Running consultations will run to completion.\n")


def _build_oracle_manager(oracle_cfg: OracleConfig, developer_prompt_text: str | None, 
                          few_shot_examples: list | None) -> OracleConsultationManager:
    llm_oracle = OracleConsultationManager(
        api_key=oracle_cfg.api_key,
        model_name=oracle_cfg.model_name,
        interaction_style=oracle_cfg.interaction_style,
        base_url=oracle_cfg.base_url,
        temperature=oracle_cfg.temperature,
        top_p=oracle_cfg.top_p,
        reasoning_effort=oracle_cfg.reasoning_effort,
        max_completion_tokens=oracle_cfg.max_completion_tokens,
        enable_thinking=oracle_cfg.enable_thinking,
        supports_chat_template_kwargs=oracle_cfg.supports_chat_template_kwargs,
        response_format=oracle_cfg.response_format,
    )
    if developer_prompt_text is not None:
        llm_oracle.add_developer_message(developer_prompt_text)
    if few_shot_examples:
        llm_oracle.add_few_shot_examples(few_shot_examples)
    if VERBOSE:
        _debug_conversation_history(llm_oracle=llm_oracle)
    return llm_oracle


def _build_pair_entity_types(m_ask_init_alignment_df: pd.DataFrame) -> dict[str, str]:
    """
    Build src|tgt -> entityType lookup from m_ask DataFrame; used for obtaining the neccesary 
    class (CLS), property (OPROP), and instance (INST) dev/system + user prompts for use during
    consultation. ie. dict: { "SRC_URI|TGT_URI": "OPROP" || "INST" || "CLS", ... (for all mappings in M_ask) }
    """
    pair_entity_types = {}
    for _, row in m_ask_init_alignment_df.iterrows():
        base_key = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
        etype = str(row.iloc[4]).strip() if len(row) > 4 else "CLS"
        pair_entity_types[base_key] = etype
        if VERBOSE and VERY_VERBOSE:
            debug(f"Attached Entity Type '{etype}' to Mapping '{base_key}'")
    
    return pair_entity_types


def _resolve_developer_override(full_key: str, developer_prompt_map: dict | None, pair_entity_types: dict[str, str]) -> str | None:
    if not developer_prompt_map:
        return None
    base_key = PAIRS_SEPARATOR.join(full_key.split(PAIRS_SEPARATOR)[:2])
    etype = pair_entity_types.get(base_key, "CLS")
    return developer_prompt_map.get(etype)


def _run_consultations(llm_oracle: OracleConsultationManager, m_ask_prompts: dict[str, str], 
                       pair_entity_types: dict[str, str], developer_prompt_map: dict | None, 
                       max_workers: int, failure_tolerance: int, desc: str) -> dict | None:
    """Dispatches all prompts in parallel; returns None if aborted"""
    results = {}
    consecutive_failures = 0
    cumulative_failures = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                consult_oracle_for_mapping,
                key, prompt, llm_oracle,
                developer_override=_resolve_developer_override(key, developer_prompt_map, pair_entity_types),
            )
            for key, prompt in m_ask_prompts.items()
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                key, prediction, confidence, usage = future.result()
            except Exception as e:
                should_abort, consecutive_failures, cumulative_failures = _check_failure_abort(
                    prediction, consecutive_failures, cumulative_failures, failure_tolerance,
                )
                if should_abort:
                    _print_abort_message(consecutive_failures, cumulative_failures, failure_tolerance)
                    executor.shutdown(wait=True, cancel_futures=True)
                    return None
                warning(f"Consultation exhausted retries and raised: {type(e).__name__}: {e}")

            results[key] = (prediction, confidence, usage)

    return results


@retry(max_retries=2, on_final_failure=_consultation_failure_record)
def consult_oracle_for_mapping(key, prompt, llm_oracle, developer_override=None):
    """obtain prediction, confidence and token usage for a given mapping (id'd via key)"""
    try:
        response = llm_oracle.consult_oracle(prompt, developer_override)
        prediction = get_llm_mapping_prediction(response)
        confidence = calculate_logprobs_confidence(response.logprobs)
        return key, prediction, confidence, response.usage

    except BadRequestError as bre:
        try:
            body = getattr(bre, "body", None) or {}
            error_msg = body.get("error", {}).get("message") or str(bre)
        except (AttributeError, TypeError):
            error_msg = str(bre)
        warning(f'BadRequestError: {error_msg} (for mapping: {key})')
        return key, "error", float("nan"), TokensUsage(input_tokens=None, output_tokens=None)


def consult_oracle_for_mappings_to_ask(
    m_ask_prompts: dict[str, str],
    m_ask_init_alignment_df: pd.DataFrame,
    oracle_cfg: OracleConfig,
    developer_prompt_text: str | None = None,
    developer_prompt_map: dict | None = None,
    few_shot_examples: list | None = None,
    **kwargs
) -> pd.DataFrame | None:

    failure_tolerance = _resolve_failure_tolerance(oracle_cfg, len(m_ask_prompts))
    llm_oracle = _build_oracle_manager(oracle_cfg, developer_prompt_text, few_shot_examples)
    llm_oracle.freeze_messages() # thread-safe (may need to adjust for RAG-based prompting)
    pair_entity_types = _build_pair_entity_types(m_ask_init_alignment_df) if developer_prompt_map else {}

    results = _run_consultations(
        llm_oracle=llm_oracle,
        m_ask_prompts=m_ask_prompts,
        pair_entity_types=pair_entity_types,
        developer_prompt_map=developer_prompt_map,
        max_workers=oracle_cfg.max_workers,
        failure_tolerance=failure_tolerance,
        desc="Oracle consultations",
    )

    if results is None:
        return None

    ordered_pred, ordered_conf, ordered_in, ordered_out = [], [], [], []
    skipped_count = 0

    for _, row in m_ask_init_alignment_df.iterrows():
        key = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
        if key in results:
            pred, conf, usage = results[key]
            ordered_pred.append(pred)
            ordered_conf.append(conf)
            ordered_in.append(usage.input_tokens)
            ordered_out.append(usage.output_tokens)
        else:
            skipped_count += 1
            ordered_pred.append("skipped")
            ordered_conf.append(float("nan"))
            ordered_in.append(None)
            ordered_out.append(None)

    if skipped_count > 0:
        warn(f"{skipped_count} mappings 'skipped' (no prompt built due to unresolvable class URIs).")

    m_ask_df_ext = m_ask_init_alignment_df.copy()

    m_ask_df_ext['Oracle_prediction'] = ordered_pred # bool | "error" | "skipped"
    m_ask_df_ext['Oracle_confidence'] = pd.Series(ordered_conf, dtype="float64")
    m_ask_df_ext['Oracle_input_tokens'] = pd.Series(ordered_in, dtype="Int64")      # nullable
    m_ask_df_ext['Oracle_output_tokens'] = pd.Series(ordered_out, dtype="Int64")    # nullable

    return m_ask_df_ext


def consult_oracle_bidirectional(
    m_ask_prompts: dict[str, str],
    m_ask_init_alignment_df: pd.DataFrame,
    oracle_cfg: OracleConfig,
    developer_prompt_text: str | None = None,
    developer_prompt_map: dict | None = None,
    few_shot_examples: list | None = None,
    **kwargs
) -> pd.DataFrame | None:
    """
    Consult an LLM Oracle with bidirectional subsumption prompts.
    --------------------------------------------------------------
    Sends forward and reverse subsumption queries, then aggregates into
    per-candidate equivalence predictions. A candidate is predicted as
    equivalent (ie. true) if and only if both forward and reverse subsumption 
    hold; otherwise false.
    """
    failure_tolerance = _resolve_failure_tolerance(oracle_cfg, len(m_ask_prompts))
    llm_oracle = _build_oracle_manager(oracle_cfg, developer_prompt_text, few_shot_examples)
    llm_oracle.freeze_messages() # again, adjust for RAG-based prompting
    pair_entity_types = _build_pair_entity_types(m_ask_init_alignment_df) if developer_prompt_map else {}

    results = _run_consultations(
        llm_oracle=llm_oracle,
        m_ask_prompts=m_ask_prompts,
        pair_entity_types=pair_entity_types,
        developer_prompt_map=developer_prompt_map,
        max_workers=oracle_cfg.max_workers,
        failure_tolerance=failure_tolerance,
        desc="Oracle consultations (bidirectional)",
    )

    if results is None:
        return None

    ordered_pred, ordered_conf, ordered_in, ordered_out = [], [], [], []
    skipped_count = 0

    for _, row in m_ask_init_alignment_df.iterrows():
        
        base = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
        rev_key = base + PAIRS_SEPARATOR + "REVERSE"

        if base not in results or rev_key not in results:
            skipped_count += 1
            ordered_pred.append("skipped")
            ordered_conf.append(float("nan"))
            ordered_in.append(None)
            ordered_out.append(None)
            if VERBOSE and VERY_VERBOSE:
                debug(f"(consult_oracle_bidirectional) skipping mapping in M_ask: {base} <> {rev_key}.")
            continue

        fwd_pred, fwd_conf, fwd_usage = results[base]
        rev_pred, rev_conf, rev_usage = results[rev_key]

        # true iff both hold (equivalence via mutual subsumption)
        if fwd_pred == "error" or rev_pred == "error":
            agg_pred = "error"
            warn("Encountered an 'error' when computing logical AND in bidirectional mode.")
        else:
            agg_pred = bool(fwd_pred) and bool(rev_pred)

        try: # we consider the confidence as the minimum of the two directional confidences
            agg_conf = min(float(fwd_conf), float(rev_conf))
        except (ValueError, TypeError):
            warn("Confidence value can not be resolved for an M_ask mapping in bidirectional mode.")
            agg_conf = float('nan')

        ordered_pred.append(agg_pred)
        ordered_conf.append(agg_conf)
        ordered_in.append((fwd_usage.input_tokens or 0) + (rev_usage.input_tokens or 0))
        ordered_out.append((fwd_usage.output_tokens or 0) + (rev_usage.output_tokens or 0))

    if skipped_count > 0:
        warning(f"{skipped_count} mappings marked as 'skipped'.")

    m_ask_df_ext = m_ask_init_alignment_df.copy()
    m_ask_df_ext['Oracle_prediction'] = ordered_pred # bool | "error" | "skipped"
    m_ask_df_ext['Oracle_confidence'] = pd.Series(ordered_conf, dtype="float64")
    m_ask_df_ext['Oracle_input_tokens'] = pd.Series(ordered_in, dtype="Int64")      # nullable
    m_ask_df_ext['Oracle_output_tokens'] = pd.Series(ordered_out, dtype="Int64")    # nullable

    return m_ask_df_ext
