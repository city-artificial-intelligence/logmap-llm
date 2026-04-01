from __future__ import annotations

from pydantic import BaseModel
from enum import Enum


import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)



class BinaryOutputFormat(BaseModel):
    answer: bool


# JD. structured answer format for yes/no response
class YesNoOutputFormat(BaseModel):
    answer: str


class BinaryOutputFormatWithReasoning(BaseModel):
    reasoning: str
    answer: bool


class TokensUsage(BaseModel):
    input_tokens: int | None
    output_tokens: int | None


class LLMCallOutput(BaseModel):
    message: str | bool
    usage: TokensUsage
    logprobs: list | None
    parsed: BaseModel | None


PAIRS_SEPARATOR = "|"


class EntityType(Enum):
    '''
    The symbols denoting entity types are the ones used by LogMap
    in its output files. We want LogMap-LLM to use these same symbols
    in its output files so that the LogMap-LLM user experiences an
    integrated and intuitive product. It made sense to also use these
    same symbols internally (in memory). So think carefully before
    making changes.    
    '''
    CLASS = 'CLS'
    DATAPROPERTY = 'DPROP'
    OBJECTPROPERTY = 'OPROP'
    INSTANCE = 'INST'
    UNKNOWN = 'UNKNO'


class EntityRelation(Enum):
    '''
    The symbols denoting entity relations are the ones used by LogMap
    in its output files. We want LogMap-LLM to use these same symbols
    in its output files so that the LogMap-LLM user experiences an
    integrated and intuitive product. It made sense to also use these
    same symbols internally (in memory). So think carefully before
    making changes.
    '''
    SUBCLASSOF = '<'
    SUPERCLASSOF = '>'
    EQUIVALENCE = '='



# token sets for recognising positive/negative oracle predictions
# to be used by logprobs extraction & text-fallback parsing

POSITIVE_TOKENS = frozenset({"true", "yes"})

NEGATIVE_TOKENS = frozenset({"false", "no"})

# answer format controls the response instruction wording in prompts and 
# the structured output schema sent to the LLM; options:
# 'true_false': prompt uses "true or false" & schema is BinaryOutputFormat
# 'yes_no': prompt uses "yes or no" & schema is YesNoOutputFormat

ANSWER_FORMATS = frozenset({"true_false", "yes_no"})

DEFAULT_ANSWER_FORMAT = "true_false"

RESPONSE_INSTRUCTION = {
    "true_false": 'Respond with "True" or "False".',
    "yes_no":     'Respond with "Yes" or "No".',
}

RESPONSE_FORMAT_FOR_ANSWER = {
    "true_false": BinaryOutputFormat,
    "yes_no":     YesNoOutputFormat,
}