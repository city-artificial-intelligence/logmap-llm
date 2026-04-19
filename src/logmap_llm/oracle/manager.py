'''
logmap_llm.oracle.manager
Contains OracleConsultationManager/s which manages LLM interactions via the OpenAI SDK. 
Supporting both OpenRouter and local endpoints (vLLM, SGLang).
'''
from __future__ import annotations
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from logmap_llm.constants import (
    BinaryOutputFormat,
    BinaryOutputFormatWithReasoning,
    YesNoOutputFormat,
    YesNoOutputFormatWithReasoning,
    LLMCallOutput,
    TokensUsage,
    POSITIVE_TOKENS,
    NEGATIVE_TOKENS,
    InteractionStyle,
    RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE,
    VERBOSE,
    VERY_VERBOSE,
)
from logmap_llm.utils.misc import resolve_response_format_to_str
from logmap_llm.utils.logging import (
    warning,
    debug,
)
import json


class OracleConsultationManager:
    """
    Manages consultations with an LLM Oracle via OpenAI-compatible API.
    Supports OpenRouter, vLLM, SGLang, and any OpenAI-compatible endpoint.
    """
    def __init__(self, api_key: str, model_name: str, interaction_style: str, base_url: str, temperature: float, top_p: float, 
                 reasoning_effort: str | None, max_completion_tokens: int, enable_thinking: bool, supports_chat_template_kwargs: bool | None = None,
                 response_format: BinaryOutputFormat | BinaryOutputFormatWithReasoning | YesNoOutputFormat | YesNoOutputFormatWithReasoning | None = None):

        if VERBOSE:
            debug(f"Initialising OracleConsultationManager ... with args:")
            debug(f"model={model_name}")
            debug(f"base_url={base_url}")
            debug(f"temp={str(temperature)}")
            debug(f"reasoning={str(enable_thinking)}")
            debug(f"response_format={resolve_response_format_to_str(response_format)}")

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

        self.interaction_style = self._resolve_interaction_style(
            interaction_style, self.base_url,
        )

        if VERBOSE:
            debug(f"(resolved argument) interaction_style={InteractionStyle(self.interaction_style)}")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=2,
        )

        self.messages = []
        self._frozen = False
        self._frozen_messages = ()

        self.response_format = response_format

        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.enable_thinking = enable_thinking

        #################################
        #-------------------------------#
        # CHECK THE FOLLOWING (TODO)    #
        #-------------------------------#
        #################################

        self.logprobs = True
        self.top_logprobs = 3

        #############################################################################
        # --------------------------------------------------------------------------#
        # chat_template_kwargs compatibility (Mistral support)                      #
        # --------------------------------------------------------------------------#
        # Does the model's tokenizer supports chat_template_kwargs in extra_body?   #
        # Mistral models served via vLLM use a proprietary "tekken" tokenizer...    #
        # this rejects any chat_template / chat_template_kwargs parameters.         #
        #                                                                           #
        # Resolution order:                                                         #
        #   1. Explicit config override (True/False) — takes priority               #
        #   2. Auto-detection from model_name                                       #
        #   3. Otherwise default to True                                            #
        # --------------------------------------------------------------------------#
        #############################################################################

        explicit = supports_chat_template_kwargs

        if explicit is not None:
            self.supports_chat_template_kwargs = explicit
        else:
            self.supports_chat_template_kwargs = (
                not self._is_mistral_family(self.model_name)
            )

        # TODO: maintain an official 'supported models' list


    @staticmethod
    def _is_mistral_family(model_name: str) -> bool:
        """Check whether a model name belongs to the Mistral family."""
        return "mistral" in model_name.lower()


    @staticmethod
    def _resolve_interaction_style(requested: str, base_url: str | None) -> InteractionStyle:
        """
        Translates the configured interaction_style into a concrete enum value (see constants.py).
        """
        if requested.lower() == InteractionStyle.AUTOMATIC:
            if 'openrouter.ai' in (base_url or ''):
                return InteractionStyle.OPEN_ROUTER
            # else (not openrouter):
            return InteractionStyle.LOCAL_VLLM # compatible \w vLLM & SGLang
        # else (not auto):
        try:
            return InteractionStyle(requested)
        except ValueError:
            raise ValueError(f"Unknown interaction_style '{requested}'.")


    def _resolve_system_role(self) -> str:
        """
        Return the appropriate role name for system-level instructions.
        OpenAI/OpenRouter accept 'developer'; vLLM expects 'system'.
        """
        # add the neccesary configuration options here when required (hence: `in`)
        if self.interaction_style in (InteractionStyle.LOCAL_VLLM, InteractionStyle.LOCAL_SG_LANG):
            return 'system'
        return 'developer'

    # -------------------
    # Message management:
    # -------------------

    def freeze_messages(self) -> None:
        """
        Freeze messages — no further modifications allowed.

        Call before entering multithreaded consultation to ensure the
        shared message list is not mutated during concurrent reads.
        """
        self._frozen = True
        self._frozen_messages = tuple(self.messages)


    def add_developer_message(self, message: str) -> None:
        """Add or replace the developer (system) message."""
        if self._frozen:
            raise RuntimeError("Cannot modify messages after freeze_messages().")
        
        role = self._resolve_system_role()
        system_roles = {"developer", "system"}

        if len(self.messages) == 0 or self.messages[0]["role"] not in system_roles:
            self.messages.insert(0, self.build_api_message(role, message))
        else:
            self.messages[0] = self.build_api_message(role, message)


    def add_message(self, role: str, message: str) -> None:
        """Add a message to the conversation history."""
        if self._frozen:
            raise RuntimeError("Cannot modify messages after freeze_messages().")
        self.messages.append(self.build_api_message(role, message))


    def add_few_shot_examples(self, examples: list) -> None:
        """
        Add few-shot example pairs as user/assistant message turns.
        Call after add_developer_message() and before freeze_messages().
        Each example is a (user_prompt, assistant_response) tuple.
        """
        for user_prompt, assistant_response in examples:
            self.add_message("user", user_prompt)
            self.add_message("assistant", assistant_response)


    def set_response_format(self, response_format: BaseModel | dict) -> None:
        self.response_format = response_format


    def build_api_message(self, role: str, message: str) -> dict:
        return {"role": role, "content": message}


    def set_interaction_style(self, interaction_style) -> None:
        self.interaction_style = interaction_style


    def clear_messages(self) -> None:
        """Clear all messages and unfreeze."""
        self._frozen = False
        self._frozen_messages = ()
        self.messages = []

    # --------------------
    # Oracle consultation
    # --------------------

    def consult_oracle(self, message, developer_override=None):
        """Consult the LLM Oracle, dispatching to the appropriate method."""

        if self.response_format is RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE:
            if VERBOSE and VERY_VERBOSE:
                debug("(consult_oracle) Consulting via plain.")
            return self._consult_via_plain(message, developer_override)

        elif self.interaction_style in (InteractionStyle.OPEN_ROUTER, InteractionStyle.OPEN_AI_CHAT_COMPLETIONS_PARSE):
            if VERBOSE and VERY_VERBOSE:
                debug("(consult_oracle) Consulting via parse.")
            return self._consult_via_parse(message, developer_override)
        
        elif self.interaction_style in (InteractionStyle.LOCAL_VLLM, InteractionStyle.LOCAL_SG_LANG):
            if VERBOSE and VERY_VERBOSE:
                debug("(consult_oracle) Consulting via create.")
            return self._consult_via_create(message, developer_override)

        else:
            raise ValueError(f"Interaction style not recognised: {self.interaction_style}")


    def _build_base_kwargs(self, prompt, developer_override=None):
        
        msgs = list(self._frozen_messages if self._frozen else self.messages)

        # Per-call developer prompt substitution (entity-type-aware)
        if developer_override is not None and msgs:
            system_roles = {"developer", "system"}
            if msgs[0]["role"] in system_roles:
                msgs[0] = self.build_api_message(msgs[0]["role"], developer_override)

        messages = [*msgs, self.build_api_message("user", prompt)]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "logprobs": self.logprobs,
            "top_logprobs": self.top_logprobs,
        }

        # For models with built-in thinking/CoT (e.g. Qwen3, etc), enable_thinking requires extra_body:
        if self.enable_thinking is not None and self.supports_chat_template_kwargs:
            # (skipped when supports_chat_template_kwargs is False)
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": self.enable_thinking
                }
            }

        return kwargs


    def _extract_response(self, response, parsed_output):
        """Extract common fields from an API response into LLMCallOutput."""
        output_message = response.choices[0].message.content        
        usage = TokensUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        try:
            logprobs = response.choices[0].logprobs.model_dump()["content"]
        except (AttributeError, KeyError, TypeError):
            logprobs = []

        return LLMCallOutput(
            message=output_message,
            usage=usage,
            logprobs=logprobs,
            parsed=parsed_output,
        )


    @staticmethod
    def _parse_plain_text_answer(raw_content: str) -> BinaryOutputFormat:
        """
        Parse a plain-text LLM response into a BinaryOutputFormat.
        Permissive parsing means lowercases, strips common punctuation/quoting,
        and matches against POSITIVE_TOKENS / NEGATIVE_TOKENS. Used both
        as the primary parser in plain mode and as a fallback inside
        _consult_via_create when guided JSON decoding fails
        """
        if raw_content is None:
            raise ValueError("LLM returned empty response")
        cleaned = raw_content.strip().strip(' .!?"\'\n\t').lower()
        tokens = set(cleaned.split())
        pos_match = bool(tokens & set(POSITIVE_TOKENS))
        neg_match = bool(tokens & set(NEGATIVE_TOKENS))
        if pos_match and not neg_match:
            return BinaryOutputFormat(answer=True)
        if neg_match and not pos_match:
            return BinaryOutputFormat(answer=False)
        if pos_match and neg_match:
            raise ValueError(f"Ambiguous LLM response (both pos and neg tokens): {raw_content[:200]!r}")        
        raise ValueError(f"Could not parse LLM response: {raw_content[:200]!r}")


    def _consult_via_plain(self, prompt, developer_override=None):
        """
        Consult via create() WITHOUT response_format constraint;
        if the model produces free-form text, we parse it ourselves
        against POSITIVE_TOKENS, NEGATIVE_TOKENS. It works identically
        against any OpenAI-compatible endpoint (OpenRouter, vLLM, SGLang)
        because no provider-specific schema mechanism is involved
        """
        kwargs = self._build_base_kwargs(prompt, developer_override)
        kwargs["max_tokens"] = self.max_completion_tokens        
        # deliberately do NOT set response_format
        # this is the whole point of plain mode
        response = self.client.chat.completions.create(**kwargs)
        raw_content = response.choices[0].message.content
        parsed_output = self._parse_plain_text_answer(raw_content)
        
        return self._extract_response(response, parsed_output)


    def _consult_via_parse(self, prompt, developer_override=None):
        """
        Consult via OpenAI's parse() method with structured outputs
        """
        kwargs = self._build_base_kwargs(prompt, developer_override)
        kwargs["max_completion_tokens"] = self.max_completion_tokens
        kwargs["response_format"] = self.response_format

        if self.reasoning_effort and self.reasoning_effort != 'none':
            kwargs["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.parse(**kwargs)
        parsed_output = response.choices[0].message.parsed

        return self._extract_response(response, parsed_output)


    def _consult_via_create(self, prompt, developer_override=None):
        """
        Consult via create() with JSON schema guided decoding (vLLM, SGLang)
        """
        kwargs = self._build_base_kwargs(prompt, developer_override)
        kwargs["max_tokens"] = self.max_completion_tokens

        if hasattr(self.response_format, 'model_json_schema'):
            schema = self.response_format.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_format.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            kwargs["response_format"] = self.response_format

        response = self.client.chat.completions.create(**kwargs)
        raw_content = response.choices[0].message.content

        try:
            parsed_dict = json.loads(raw_content)
            parsed_output = self.response_format(**parsed_dict)
        except (json.JSONDecodeError, ValidationError) as exc:
            warning(f"_consult_via_create JSON-schema parse failed ({type(exc).__name__}: {exc}); "
                    f"falling back to _parse_plain_text_answer.")
            parsed_output = self._parse_plain_text_answer(raw_content)

        return self._extract_response(response, parsed_output)
