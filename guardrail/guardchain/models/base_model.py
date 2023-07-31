import logging
import json 
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from colorama import Fore

from pydantic import Extra, Field, BaseModel, PrivateAttr
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from guardrail.guardchain.agent.message import BaseMessage, AIMessage
from guardrail.guardchain.tools.base_tool import Tool

from guardrail.guardchain.utils import print_with_color

from guardrail.guardchain.models.jsonformer import Jsonformer

logger = logging.getLogger(__name__)


class Generation(BaseModel):
    """Output of a single generation."""

    message: BaseMessage
    """Generated text output."""

    generation_info: Optional[Dict[str, Any]] = None
    """Raw generation info response from the provider."""
    """May include things like reason for finishing (e.g., in OpenAI)."""
    # TODO: add log probs


class LLMResult(BaseModel):
    """Class that contains all relevant information for an LLM Result."""

    generations: List[Generation]
    """List of the things generated. This is List[List[]] because
    each input could have multiple generations."""
    llm_output: Optional[dict] = None
    """For arbitrary LLM provider specific output."""


class EmbeddingResult(BaseModel):
    texts: List[str]
    embeddings: List[List[float]]


class BaseLanguageModel():
    """Wrapper around HuggingFace's Transformers language models.

    To use, you should have the `transformers` library installed, and the
    appropriate model and tokenizer should be loaded.

    Any parameters that are valid to be passed to the `generate` call can be passed
    in, even if not explicitly saved on this class.
    """

    model_name: str = "guardrail/llama-2-7b-guanaco-instruct-sharded"
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    model_kwargs: Dict[str, Any] = None
    """Holds any model parameters valid for `generate` call not explicitly specified."""

    tokenizer_kwargs: Dict[str, Any] = None
    
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    n: int = 1
    """Number of completions to generate for each prompt."""
    max_tokens: Optional[int] = 512
    """Maximum number of tokens to generate."""
    max_length: int = 512
    """Tokenizer maximum length."""
    truncation: bool = True
    """Tokenizer truncation"""
    padding: bool = True 
    """Padding truncation"""
    max_array_length: int = 64
    max_number_tokens: int = 512

    _tokenizer: Optional[AutoTokenizer] = Field(default=None)
    """The AutoTokenizer instance for this model."""
    
    _model: Optional[AutoModelForCausalLM] = Field(default=None)
    """The AutoModelForCausalLM instance for this model."""

    conversation_schema: json = {}

    default_stop_tokens: List[str] = ["."]
    
    def __init__(
        self,
        model_name: str = None,
        temperature: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        padding: Optional[bool] = None,
        max_array_length: Optional[int] = None,
        max_number_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        if model_name is not None:
            self.model_name = model_name

        if temperature is not None:
            self.temperature = temperature

        if model_kwargs is not None:
            self.model_kwargs = model_kwargs

        if max_length is not None:
            self.max_length = max_length

        if truncation is not None:
            self.truncation = truncation

        if padding is not None:
            self.padding = padding

        if self.model_kwargs is not None:
            if torch.cuda.is_available():
                self.model_kwargs["device_map"] = "auto"
        else:
            # Initialize self.model_kwargs as an empty dictionary and assign "device_map" value
            self.model_kwargs = {}
            if torch.cuda.is_available():
                self.model_kwargs["device_map"] = "auto"

        # Initialize tokenizer and model here
        if self.tokenizer_kwargs is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, **self.tokenizer_kwargs
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        if self.model_kwargs is not None:
            kwargs = self.model_kwargs.copy()
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **kwargs,
            )
        else: 
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                              load_in_8bit=True)
        
        self.agent_should_respond = {
            "type": "object",
            "properties": {
                "ai_agent_should_respond": {"type": "boolean"}
            },
        }

        self.agent_planning = {
            "type": "object",
            "properties": {
                "thoughts": {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string"},
                        "need_use_tool": {"type": "boolean"},
                    },
                },
                "tool": {
                    "type": "object",
                    "properties": { 
                        "name": {"type": "string"},
                        "args": {
                            "type": "object",
                            "properties": {
                                "arg_name": {"type": "string"}
                            }
                        }
                    },
                },
                "assistant": {
                    "type": "object",
                    "properties": {
                        "ai_response": {"type": "string"},
                    },
                },
            },
        }

    # Create properties for tokenizer and model so they can be accessed externally
    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        # You can add custom logic here if needed
        self._tokenizer = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        # You can add custom logic here if needed
        self._model = value
        
    def generate(
        self,
        messages: List[BaseMessage],
        functions: Optional[List[Tool]] = None,
        stop: Optional[List[str]] = None,
        planning: Optional[bool] = False
    ) -> LLMResult:
        prompt = self._construct_prompt_from_message(messages)
        print("Prompt:", prompt)

        if planning:
            json_schema = self.agent_planning
        else:
            json_schema = self.agent_should_respond

        generator = Jsonformer(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        json_schema=json_schema,
                        prompt=prompt,
                        max_string_token_length=self.max_length,
                        max_array_length=self.max_array_length,
                        max_number_tokens=self.max_number_tokens,
                        temperature=self.temperature,
                    )
        generation = generator()
        return self._create_llm_result(generation=generation, prompt=prompt, stop=stop, planning=planning)

    def encode(self, texts: List[str]) -> EmbeddingResult:
        # Tokenize the input texts using the tokenizer
        inputs_encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
        )

        # Generate embeddings with the model
        with torch.no_grad():
            embeddings = self.model.base_model(input_ids=inputs_encoded.input_ids).last_hidden_state

        return EmbeddingResult(texts=texts, embeddings=embeddings.tolist())

    class Config:
        """Configuration for this Pydantic object."""
        arbitrary_types_allowed = True
        extra = Extra.ignore
        validate_assignment = False  # Turn off Pydantic validation errors

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the language model API."""
        return {
            "model": self.model_name,
            "request_timeout": self.request_timeout,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "temperature": self.temperature,
            **self.model_kwargs,
        }

    def _create_retry_decorator(self) -> Callable[[Any], Any]:
        min_seconds = 1
        max_seconds = 60
        # Wait 2^x * 1 second between each retry starting with
        # 4 seconds, then up to 10 seconds, then 10 seconds afterward
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
            retry=(
                retry_if_exception_type(
                    Exception
                )  # Replace this with specific exception types from the provider.
            ),
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )

    def generate_with_retry(self, **kwargs: Any) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = self._create_retry_decorator()

        @retry_decorator
        def _generate_with_retry(**kwargs: Any) -> Any:
            # Replace this with the actual generate function from the provider.
            return self.model.generate(**kwargs)

        return _generate_with_retry(**kwargs)
    
    @staticmethod
    def _construct_prompt_from_message(messages: List[BaseMessage]):
        prompt = ""
        for msg in messages:
            prompt += msg.content
        return prompt

    @staticmethod
    def _enforce_stop_tokens(text: str, stop: List[str]) -> str:
        """Cut off the text as soon as any stop words occur."""
        first_index = len(text)
        for s in stop:
            if s in text:
                first_index = min(text.index(s), first_index)

        return text[:first_index].strip()

    def _create_llm_result(
        self, generation: Any, prompt: str, stop: List[str], planning=False
    ) -> LLMResult:
        print("Generation: ", str(generation))
        print("Planning: ", planning)
        
        if not planning:
            ai_response = generation["ai_agent_should_respond"]
            if ai_response: ai_response = "True"
            else: ai_response = "False"
        else: 
            ai_response = json.dumps(generation)
            
        print("AI Response: ", ai_response)

        return LLMResult(
                generations=[Generation(message=AIMessage(content=ai_response))],
                llm_output={
                    "token_usage": len(str(generation).split()),
                    "model_name": self.model_name,
                },
        )
        """
            

        # text = generation[0]["generated_text"][len(prompt) :]
        text = generation[0]["generated_text"][len(prompt) :]
        print("Generated text: ", str(text))
        print_with_color(str(text), Fore.GREEN)

        if self.max_tokens:
            token_ids = self.tokenizer.encode(text)[: self.max_tokens]
            text = self.tokenizer.decode(token_ids)

        # it is better to have a default stop token so model does not always generate to max
        # sequence length
        stop = stop or self.default_stop_tokens
        print("Before enforce...", stop)
        # json_output = self._enforce_stop_tokens(text=json_output, stop=stop)

        return LLMResult(
            generations=[Generation(message=AIMessage(content=text))],
            llm_output={
                "token_usage": len(text.split()),
                "model_name": self.model_name,
            },
        )
        """

