from __future__ import annotations

from pydantic import BaseModel
from enum import Enum


import logging
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)



class BinaryOutputFormat(BaseModel):
    answer: bool


class BinaryOutputFormatWithReasoning(BaseModel):
    reasoning: str
    answer: bool


class YesNoOutputFormat(BaseModel):
    answer: str


class YesNoOutputFormatWithReasoning(BaseModel):
    reasoning: str
    answer: str


class TokensUsage(BaseModel):
    input_tokens: int | None
    output_tokens: int | None


class LLMCallOutput(BaseModel):
    message: str | bool
    usage: TokensUsage
    logprobs: list | None
    parsed: BaseModel | None


PAIRS_SEPARATOR = "|"

###
# positive & negative tokens for answer format
# (used by logprobs extraction, text-fallback parsing, and CSV normalisation)
###

POSITIVE_TOKENS = frozenset({"true", "yes"})
NEGATIVE_TOKENS = frozenset({"false", "no"})

###
# Answer format: controls the response instruction wording in prompts
# AND the structured output schema sent to the LLM.
####

ANSWER_FORMATS = frozenset({"true_false", "yes_no"})


DEFAULT_ANSWER_FORMAT = "true_false" # TODO: modify to yes_no? Depends on consensus


RESPONSE_INSTRUCTION = {
    "true_false": 'Respond with "True" or "False".',
    "yes_no":     'Respond with "Yes" or "No".',
}


RESPONSE_FORMAT_FOR_ANSWER = {
    "true_false": BinaryOutputFormat,
    "yes_no":     YesNoOutputFormat,
}


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


###
# PIPELINE CONFIGURATIONS
# -----------------------
# The original implementation, which is provided under:
#
#   https://github.com/jonathondilworth/logmap-llm/tree/jd-extended
#
# breaks the pipeline from logmap_llm.py down into composable
# function calls, where the output from fn_i is the input to fn_i+1
# (producer-consumer type pattern; these live within the `steps.py` module). 
# Within each 'step', rather than using if-else, we use match-case
# in an effort to make it a little more 'readable' (though, may be
# considered stylistic).
#
# The configuration enums provided below are used in the for each
# case in the switch.
###

# TODO: documentation from the outset, see TODOs.
# For now, let's just include comments.

# STEP ONE (INITIAL ALIGNMENT)

class AlignMode(str, Enum):
    ALIGN = 'align'
    REUSE = 'reuse'
    BYPASS = 'bypass'

# STEP TWO (BUILD PROMPTS)

class PromptBuildMode(str, Enum):
    BUILD = 'build'
    REUSE = 'reuse'
    BYPASS = 'bypass'

# STEP THREE (CONSULT ORACLE)

class ConsultMode(str, Enum):
    CONSULT = 'consult'
    REUSE = 'reuse'
    LOCAL = 'local'
    BYPASS = 'bypass'

# STEP FOUR (REFINEMENT)

class RefineMode(str, Enum):
    REFINE = 'refine'
    BYPASS = 'bypass'



###
# Constants (related to bridging.py)
# -----------------------------------
# NOTE: M_ASK_COLUMNS is an immutable tuple, rather than a list
# to obtain the list, you can call the same function as before
# in bridging.py, which will cast it as a list.
# 
# TODO: consider whether coupling bridging.py and constants.py
# is neccesarily appropriate, or if we might consider reversing
# this change. My thinking is that configurable 'stuff' should
# live here, rather than being hardcoded or scattered across the
# project. It's probably fine like this, actually.
###

COL_SOURCE_ENTITY_URI = 'source_entity_uri'
COL_TARGET_ENTITY_URI = 'target_entity_uri'
COL_RELATION = 'relation'
COL_CONFIDENCE = 'confidence'
COL_ENTITY_TYPE = 'entityType'

M_ASK_COLUMNS = (
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    COL_RELATION,
    COL_CONFIDENCE,
    COL_ENTITY_TYPE,
)
