'''
This module contains OracleConsultationManagers.
'''

from openai import OpenAI
from pydantic import BaseModel
from constants import BinaryOutputFormat, LLMCallOutput, TokensUsage


class OracleConsultationManager_OpenAI:
    '''
    An OracleConsultationManager to manage consultations with an
    Oracle (i.e. interactions with an LLM).

    This OracleConsultationManager uses the OpenAI API to connect
    with LLM endpoints.

    Using this OracleConsultationManager, LogMap-LLM users can
    connect and interact with any LLM provider that supports the
    OpenAI API. This includes OpenAI itself, of course.

    Importantly, the LLM model aggregation platform OpenRouter
    (www.openrouter.ai) supports the OpenAI API. When using
    OpenRouter, LogMap-LLM interacts only with OpenRouter, and
    OpenRouter handles the redirection of LogMap-LLM conversations
    to whichever of 500+ different LLM models the LogMap-LLM user
    selects.
    '''

    def __init__(self, api_key, model_name, interaction_style_name, **kwargs):
        '''
        Instantiate the OracleConsultationManager.
        
        Parameters
        ----------
        api_key : str
            An OpenRouter API key
        model_name : str
            And OpenRouter LLM model name
        interaction_style_name : str
            The style to be used for interacting with the LLM model
        kwargs : various
            see __init__() method below for details
        '''
        
        # OpenRouter's base url
        # (Note: This is the base URL for using OpenRouter's support
        #  for OpenAI's Chat Completions API.
        #  The base URL for using OpenRouter's support for OpenAI's
        #  Responses API is 'https://openrouter.ai/api/v1/responses'.)
        # TODO: externalise the base_url in config-basic.toml and let the user decide;
        # we don't want or need to force them to use OpenRouter
        self.base_url = "https://openrouter.ai/api/v1"

        if api_key is None:
            raise ValueError('OpenRouter API key required')
        self.api_key = api_key

        if model_name is None:
            raise ValueError('OpenRouter LLM model name required')
        self.model_name = model_name

        if interaction_style_name is None:
            raise ValueError('Interaction style name required')
        self.interaction_style_name = interaction_style_name

        # Instantiate a client through which to conduct actual LLM Oracle
        # interactions. The client is an OpenAI SDK client, which
        # OpenRouter supports as a kind of defacto standard. OpenRouter 
        # forwards the requests it receives from this client to (some 
        # provider of) the LLM model designated within the requests.
        self.client = OpenAI(api_key=self.api_key, 
                             base_url=self.base_url, 
                             max_retries=2)

        # - - - - - - -
        # messages
        # - - - - - - -

        # The messages for a given interaction with some LLM.
        # A list of dictionaries. Each 'message' is a dictionary that
        # associates a 'role' with some 'content' (a text message).
        # 
        # Valid roles:
        # - 'developer' : A developer message provides instructions to an LLM
        #   model regarding the supposed situation, the task at hand, how it
        #   should conduct itself, what persona to adopt, etc.. Developer 
        #   messages have priority and are processed ahead of user messages.
        #   For example: "Talk like a pirate."
        # - 'user' : A user message represents a particular user question
        #   or prompt to be put to an LLM, within the context of the situation
        #   established by the developer instructions. A common scenario is
        #   for one fixed 'developer' message to be used in multiple LLM
        #   interactions, each one involving a particular (unique) 'user'
        #   message (prompt).
        # - 'assistant' : An assistant message represents an LLM's reply to
        #   a particular user message. Pairs of 'user' and 'assistant' messages
        #   can be assembled to provide/simulate statefulness across a
        #   multi-interaction conversation.
        self.messages = []

        # - - - - - - - - - - -
        # LLM response format
        # - - - - - - - - - - -
        
        # NOTE:
        # The latest method for supporting Structured Outputs available
        # in the OpenAI SDK involves the use of explicit class structures
        # based on the pydantic BaseModel class. Many LLM models
        # (typically earlier, older ones) do not support this latest
        # method. LLM interactions will fail if the LogMap-LLM user
        # chooses an OpenRouter LLM model that does not support this
        # advanced method for ensuring LLM outputs are structured as
        # requested.
        #
        # TODO:
        # Allow the LogMap-LLM user to choose between two forms of
        # supporting Structured Outputs, via the config.toml file.
        # Make the strong method (pydantic BaseModel) the default,
        # but allow the user to switch to the weaker but more
        # widely supported method that uses a JSON schema to 
        # express the desired output structure. This should make it
        # more likely that LLM interactions with the user's desired
        # LLM model will succeed rather than fail.
        # IE set 'response_format' to 
        # { "type": "json_schema", "json_schema": {...} } 
        #
        # For example: consider a new config.toml parameter:
        # 'structured_output_support_level' = 'strong' or 'weak'
        # And setting self.response_format would become
        # conditional upon the value of this parameter, with an
        # if/else
        #

        # An object specifying the format that the model must output.
        self.response_format = kwargs.get("response_format", BinaryOutputFormat)
        
        # - - - - - - - - -
        # LLM sampling
        # - - - - - - - - -

        # What sampling temperature to use, between 0 and 2. 
        # Higher values like 1.2 will make the output more random, 
        # while lower values like 0.2 will make it more focused 
        # and deterministic. 
        #
        # OpenAI recommends altering 'temperature' or 'top_p' but not both.
        #
        # OpenAI default: 1
        # 
        # For ontology alignment, LogMap-LLM wants an LLM's decisions regarding
        # a given mapping to be as deterministic as possible. Suppose the user
        # performs 'n' runs of the same alignment task with identical
        # configurations. For mapping 'x' in that task, we want the given
        # LLM model to reach the same decision (make the same prediction),
        # whether True or False, for all 'n' consultations (or get as close
        # to this ideal as we can). In general, for ontology alignment, we want 
        # to discourage randomness in LLM responses. 
        # Hence LogMap-LLM encourages low temperatures.
        #
        # LogMap-LLM default: 0
        #
        self.temperature = kwargs.get("temperature", 0)

        # An alternative to sampling with temperature, called nucleus sampling, 
        # where the model considers the results of the tokens with top_p 
        # probability mass. So 0.1 means only the tokens comprising the top 
        # 10% probability mass are considered.
        #
        # OpenAI recommends altering 'top_p' or 'temperature' but not both.
        #
        # OpenAI default: 1
        #
        # LogMap-LLM default: 1
        self.top_p = kwargs.get("top_p", 1)

        # Given LogMap-LLM's defaults of 0 for 'temperature' and '1' for
        # 'top_p', we see that LogMap-LLM defaults to using 'temperature'
        # to control LLM sampling (and, hence, randomness) rather than 'top_p'. 

        # TODO: Consider how to better police the setting of these two
        # parameters, given OpenAI's recommendation that only one of them
        # vary from the OpenAI default values at any one time.
        # At the moment, we rely only on user knowledge to these details
        # to avoid violating OpenAI's recommendations.

        # - - - - - - - - - - -
        # LLM reasoning 
        # - - - - - - - - - - -

        # Constrains effort on reasoning for reasoning models. 
        # Currently supported values:
        #   none, minimal, low, medium, high, and xhigh
        # Reducing reasoning effort can result in faster responses 
        # and fewer tokens used on reasoning in a response.
        #
        # LogMap-LLM default: 'none'
        self.reasoning_effort = kwargs.get("reasoning_effort", 'none')

        # - - - - - - - - - - - - -
        # LLM max generated tokens
        # - - - - - - - - - - - - -

        # The maximum number of tokens that can be generated in the chat 
        # completion. This value can be used to control costs for text 
        # generated via API.
        # 
        # But 'max_tokens' is now deprecated by OpenAI (largely, it
        # seems, because it pertains only to visible output tokens,
        # not internal 'reasoning tokens' as well).
        # OpenAI's o-series reasoning models don't even support the 
        # max_tokens parameter; they use the successor parameter,
        # max_completion_tokens, instead.
        #
        # LogMap-LLM respects the deprecation and uses the successor
        # parameter instead.
        #self.max_tokens = kwargs.get("max_tokens", 1000)

        # An upper bound for the number of tokens that can be generated for 
        # a completion, including visible output tokens and reasoning tokens.
        self.max_completion_tokens = kwargs.get("max_completion_tokens", 100)

        # TODO: Consider enforcing a relation between the level of reasoning
        # effort requested and max_completion_tokens.
        # For example, if reasoning_effort == 'none' (either because the LLM 
        # model does not support reasoning, or because reasoning has been
        # deliberately turned off), then max_completion_tokens should be 
        # small, because, after all, at the end of the day we only want a 
        # one-word, binary response of True or False.
        # But if reasoning_effort != 'none', then max_completion_tokens
        # needs to be larger, to allow the reasoning to happen. And the
        # max_completion_tokens should grow with the level of the
        # reasoning effort, to allow the reasoning to happen.

        # - - - - - - - - - - - - - - - - - - - -
        # LLM generated token log probabilities
        # - - - - - - - - - - - - - - - - - - - - 

        # Whether to return log probabilities of the output tokens or not. 
        # If true, the log probabilities of each output token in
        # completion.choices[0].message.content are returned.
        # Boolean; default is False
        self.logprobs = True

        # An integer between 0 and 20 specifying the number of most likely 
        # tokens to return at each token position, each with an associated 
        # log probability.
        self.top_logprobs = 3

        # TODO: Review the use of log probabilities and verify empirically 
        # that LogMap-LLM is getting value from doing so.
        # In RAI4Ukraine results.csv files, the
        # confidence reported there is virtually always 1.0. I can't 
        # remember seeing a value other than 1.0. That confidence is
        # derived from the log probability information. Perhaps only some
        # models return them? Are we extracting top_logprobs info
        # correctly?  If the resulting LLM confidence is always 1.0, 
        # is all this stuff with log probabilities not redundant? 
        # Wouldn't it be simpler and cleaner to remove it and set the
        # LLM confidence to 1.0 manually?

        # - - - - - - - - - - - - - - - - - - - - - - - - - -
        # further LLM interaction configuration flexibility
        # - - - - - - - - - - - - - - - - - - - - - - - - - -

        # TODO: At some point, review the LLM interaction configuration
        # flexibility and decide if we need greater flexibility such
        # as via the following mechanism which sets `self.key = value`
        # for whatever is in kwargs. Caution: this much flexibility
        # could also be dangerous and invite mis-configuration.

        # set self.key = value
        #for key, value in kwargs.items():
        #    setattr(self, key, value)

    def add_developer_message(self, message: str) -> None:
        '''
        Add a developer message (an instructions message) to the list of API messages.

        Place the developer message at the front of the list. If one already exists,
        overwrite it.
        '''

        if len(self.messages) == 0 or self.messages[0]["role"] != "developer":
            self.messages.insert(0, self.build_api_message("developer", message))
        else:
            self.messages[0] = self.build_api_message("developer", message)

    def add_message(self, role: str, message: str) -> None:
        '''
        Add a message to the list of API messages
        '''
        self.messages.append(self.build_api_message(role, message))

    def set_response_format(self, response_format: BaseModel | dict) -> None:
        """Set the response format for the server."""
        self.response_format = response_format

    def build_api_message(self, role: str, message: str) -> dict:
        """Build an API message."""
        return {"role": role, "content": message}
    
    def set_interaction_style(self, interaction_style_name):
        """Set the style to be used for interacting with the LLM Oracle."""
        self.interaction_style_name = interaction_style_name

    def consult_oracle(self, message):
        '''
        Consult an Oracle using a particular style or approach.
        
        This function simply delegates a consultation request to the
        appropriate approach-specific consultation function. This level
        of indirection permits LogMap-LLM to explore and support multiple
        different interaction approaches (or styles), as LLM APIs evolve
        and mature over time.

        For example, Open AI has a Chat Completions API and a Responses API.
        And there are multiple ways of using each of the two. The former API
        is stateless; the latter is stateful.

        OpenAI is pushing the Responses API hard, but OpenRouter's support
        for it is currently limited and in Beta status. Crucially, 
        OpenRouter's support for the Responses API is currently 'stateless',
        which means the main feature of the Responses API is unavailable.

        Also, OpenRouter has its own API SDK, but it is in Beta status 
        and subject to breaking changes. One day, when it has matured,
        there may be reason to support it as well.

        Parameters
        ----------
        message : str
            A user prompt (aka user message).

        Returns
        -------
        LLMCallOutput : Wrapper for the response message, usage, logprobs, and parsed output.
        '''

        if self.interaction_style_name == 'openai_chat_completions_parse_structured_output':
             return self.consult_oracle_openai_chat_completions_parse_structured_output(message)
        else:
            raise ValueError(f'Interaction style name not recognised: {self.interaction_style_name}')


    def consult_oracle_openai_chat_completions_parse_structured_output(self, prompt):
        '''
        Consult an LLM Oracle using OpenAI's 'chat completions' API and
        structured outputs.
        
        For structured outputs, we use the API's parse() method rather
        than its create() method, and we specify a 'response format'
        defined using a class that specialises the 'pydantic' BaseModel 
        class.

        Parameters
        ----------
        prompt : str
            Text content for a user api message.

        Returns
        -------
        LLMCallOutput : 
            Wrapper for the response message, usage, logprobs, and parsed output.
        '''
        try:
            # finalise the list of api messages for the interaction with an LLM
            messages = [*self.messages, self.build_api_message("user", prompt)]

            # assemble the configuration parameter settings
            # for the LLM interaction
            llm_interaction_kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "reasoning_effort": self.reasoning_effort,
                "max_completion_tokens": self.max_completion_tokens,
                "response_format": self.response_format,
                "logprobs": self.logprobs,
                "top_logprobs": self.top_logprobs
            }

            # consult the LLM Oracle regarding the current mapping 
            response = self.client.chat.completions.parse(**llm_interaction_kwargs)

            #
            # extract and prepare the elements of the LLM's response
            # that are of interest
            #

            # get the raw response message from the LLM Oracle; this raw 
            # response may not satisfy our requirement for a one-word, 
            # binary, true/false prediction regarding the candidate mapping;
            # we focus more intently on the parsed_output(see below)
            output_message = response.choices[0].message.content

            usage = TokensUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
            try:
                logprobs = response.choices[0].logprobs.model_dump()["content"]
            except AttributeError:
                logprobs = []
            
            parsed_output = response.choices[0].message.parsed

            return LLMCallOutput(message=output_message, 
                                 usage=usage, 
                                 logprobs=logprobs, 
                                 parsed=parsed_output)

        except Exception as e:
            raise e


    def clear_messages(self) -> None:
        '''
        Clear all messages (developer and user)
        '''
        self.messages = []
