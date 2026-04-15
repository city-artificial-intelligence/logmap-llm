from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel
from enum import Enum

###
# DEBUGGING
###

VERBOSE = False
VERY_VERBOSE = False

###
# ORACLE-RELATED PYDANTIC (SCHEMA) MODELS:
###

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

###
# ORACLE-RELATED ENUMS:
###

class AnswerFormat(str, Enum):
    TRUE_FALSE = "true_false"
    YES_NO     = "yes_no"

class PositiveToken(str, Enum):
    TRUE = "true"
    YES  = "yes"

class NegativeToken(str, Enum):
    FALSE = "false"
    NO    = "no"

class ResponseModes(str, Enum):
    STRUCTURED = "structured"
    PLAIN  = "plain"

class InteractionStyle(str, Enum):
    OPEN_ROUTER = 'openrouter',
    OPEN_AI_CHAT_COMPLETIONS_PARSE = 'openai_chat_completions_parse_structured_output',
    LOCAL_VLLM = 'vllm',
    LOCAL_SG_LANG = 'sglang',
    LOCAL_GENERIC = 'local',


###
# ORACLE-RELATED 'CONSTANTS'
###

PAIRS_SEPARATOR = "|"
ORACLE_PREDICTION_COLUMN = "Oracle_prediction"

###
# positive & negative tokens for answer format
# (used by logprobs extraction, text-fallback parsing, and CSV normalisation)
###

POSITIVE_TOKENS = frozenset({
    PositiveToken.TRUE, 
    PositiveToken.YES,
})

NEGATIVE_TOKENS = frozenset({
    NegativeToken.FALSE, 
    NegativeToken.NO,
})

#############################################################################################
#-------------------------------------------------------------------------------------------#
# Answer format and response mode                                                           #
#-------------------------------------------------------------------------------------------#
# answer_format controls the answer vocabulary: true/false vs yes/no                        #
# response_mode controls the output container: structured JSON vs plain text                #
# response_format depends on the answer_format and response_mode combination                #
#-------------------------------------------------------------------------------------------#
# ie. response_format = None                            # unstructured response             #
#                     | BinaryOutputFormat              # structured, boolean               #
#                     | BinaryOutputFormatWithReasoning # structured, boolean \w reasoning  #
#                     | YesNoOutputFormat               # structured, yes_no                #
#                     | YesNoOutputFormatWithReasoning  # structured, yes_no \w reasoning   #
#-------------------------------------------------------------------------------------------#
# NOTE: There is no support for unstructured responses with reasoning at present.           #
#-------------------------------------------------------------------------------------------#
#############################################################################################

ANSWER_FORMATS = frozenset({
    AnswerFormat.TRUE_FALSE,
    AnswerFormat.YES_NO,
})

RESPONSE_MODES = frozenset({
    ResponseModes.STRUCTURED,
    ResponseModes.PLAIN,
})

RESPONSE_INSTRUCTION = {
    (AnswerFormat.TRUE_FALSE, ResponseModes.STRUCTURED): 'Respond with a JSON object: {"answer": true} or {"answer": false}.',
    (AnswerFormat.TRUE_FALSE, ResponseModes.PLAIN     ): 'Respond with "True" or "False".',
    (AnswerFormat.YES_NO,     ResponseModes.STRUCTURED): 'Respond with a JSON object: {"answer": "Yes"} or {"answer": "No"}.',
    (AnswerFormat.YES_NO,     ResponseModes.PLAIN     ): 'Respond with "Yes" or "No".',
}

RESPONSE_FORMAT_FOR_ANSWER = {
    (AnswerFormat.TRUE_FALSE , False): BinaryOutputFormat,
    (AnswerFormat.TRUE_FALSE , True ): BinaryOutputFormatWithReasoning,
    (AnswerFormat.YES_NO     , False): YesNoOutputFormat,
    (AnswerFormat.YES_NO     , True ): YesNoOutputFormatWithReasoning,
}

RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE = None

DEFAULT_ANSWER_FORMAT = AnswerFormat.TRUE_FALSE
DEFAULT_RESPONSE_MODE = ResponseModes.STRUCTURED

###
# ONTOLOGICAL (ONTOLOGY-RELATED) ENUMS:
###

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
# (producer-consumer type pattern; these live within the `orchestration.py` 
# module). Within each 'step', rather than using if-else, we use match-case.
#
# The configuration enums provided below are used in the for each
# case in the switch.
###

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

# STEP FOUR (REFINEMENT STRATEGY)

class RefinementStrategy(str, Enum):
    '''
    When RefineMode.REFINE is matched in `runner.py` during the
    refinement step, we perform the refinement in one of two ways.
    Refinement via LogMap has the benefit of resolving any remaining
    conflicts. However, in some versions of LogMap, it cannot complete
    this step successfully for instances (such as during the OAEI 2025
    KG task). Therefore, we also implement a refinement strategy in 
    Python that uses an approximate refinement via a set union operation.
    Details are provided in `logmap_llm.pipeline.orchestration` under
    the method: `_kg_refine_in_python`. NOTE: (April 12th) 'logmap'
    does now also work for KG alignment (since patching and rebuilding
    the src) -- however, it can take quite a while to perform the 
    refinement, so the setunion operation in python is still a nice
    feature for quickly testing changes to see an approximate result.
    '''
    LOGMAP = 'logmap'
    PYTHON_SETUNION = 'python'


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

###
# CONSTANTS - ORACLE MANAGER / CONSULTATION:
###

DEFAULT_FAILURE_TOLERANCE_FLOOR       = 5
DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE = 5


###
# CONSTANTS - SIBLING SELECTION 
# ------------------------------------------
# defaults for SiblingSelector: some are (likely) tunable; see
# logmap_llm.ontology.sibling_retrieval for the strategy enum
###
 
# pretrained encoders (choices):
DEFAULT_SAPBERT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
DEFAULT_GENERAL_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
 
# effective cost cap wrt the candidate sibling set before scoring/embedding
# set this to max(sibling_count) measured over your ontologies to 'disable' 
# the cap entirely (at the price of additional embedding cost -- scales as a 
# fn of the largest number of children across all classes for each ontology, 
# ie. could be quite expensive for unclassified or 'flat' ontologies.
DEFAULT_MAX_SIBLING_CANDIDATES = 50
 
# the default number for top-k siblings (injected into prompts). 
# reccomended: keep small (1–3) to avoid bloating prompts that require siblings.
DEFAULT_TOP_K = 2
 
# default strategy name (ie. string form of SiblingSelectionStrategy)
# select from: "alphanumeric", "shortest_label", "sapbert", "sbert"
DEFAULT_SIBLING_STRATEGY = "sapbert"

###
# CONSTANTS - FEW SHOT
# --------------------
###

# in some cases, we may encounter examples that cause collisions with mappings 
# in M_ask, in which case, we retry until the set of examples has been obtained.
# In some (rare) cases, this may be impossible to satisfy; so we cap the max 
# retries to avoid an infinite loop (how likely are we to encounter N collisions?)
DEFAULT_MAX_SAMPLE_RETRIES = 50

###
# CONSTANTS - EVALUATION
# ----------------------
###

# used to identify files that satisfy the DeepOnto TSV convention for 
# ontology matching / mapping / alignment tasks, used in their own evaluation
# implementation (that we re-use, and borrow form)
# DeepOnto library: https://github.com/KRR-Oxford/DeepOnto
DEEPONTO_TSV_HEADER_PREFIXES: tuple[str, ...] = ("SrcEntity",)


###
# CONSTANTS - BRIDGING
# --------------------
###

###
# CONFIDENCE COERCION AT THE JAVA BOUNDARY
# ---------------------------------------- 
# when the oracle accepts a mapping but no usable logprob token was emitted,
# calculate_logprobs_confidence returns float('nan'); passing NaN to LogMap
# for a confidence value via MappingObjectStr(..., double conf, ...) is technically
# type-correct, but may notbe semantically coherant; and could possibly cause some
# kind of failure mode within LogMap; to protect agaisnt this, we coerce NaN to a 
# concrete default (ie. 1.0; confident 'True' -- or "yes, this mapping is correct")
# which I'm quite sure is the 'expected behaviour' (the oracle is assumed to produce
# a binary answer, either yes or no) ... and this only fires for oracle predictions
# that predict the mapping to be true; so, this should be fine.
###

DEFAULT_CONFIDENCE_FALLBACK = 1.0

