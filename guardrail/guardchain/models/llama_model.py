import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

from pydantic import Extra, Field, BaseModel, PrivateAttr
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import LlamaTokenizer, LlamaForCausalLM

from guardrail.guardchain.agent.message import BaseMessage
from guardrail.guardchain.tools.base_tool import Tool

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
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout for requests to completion API. Default is 600 seconds."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    n: int = 1
    """Number of completions to generate for each prompt."""
    max_tokens: Optional[int] = None
    """Maximum number of tokens to generate."""
    max_length: int = 512
    """Tokenizer maximum length."""
    truncation: bool = False
    """Tokenizer truncation"""
    padding: bool = False 
    """Padding truncation"""

    _tokenizer: Optional[LlamaTokenizer] = Field(default=None)
    """The LlamaTokenizer instance for this model."""
    
    _model: Optional[LlamaForCausalLM] = Field(default=None)
    """The LlamaForCausalLM instance for this model."""

    def __init__(
        self,
        model_name: str = None,
        temperature: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        max_length: Optional[int] = None,
        truncation: Optional[bool] = None,
        padding: Optional[bool] = None,
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

        # Initialize tokenizer and model here
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.model_name,
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
        )
        if self.model_kwargs is not None:
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                device_map=self.model_kwargs.get("device_map"),
                quantization_config=self.model_kwargs.get("bnb_config"),
            )
        else: 
            self.model = LlamaForCausalLM.from_pretrained(self.model_name)


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
    ) -> LLMResult:
        inputs = [msg.content for msg in messages]

        # Encode the input text using the tokenizer
        inputs_encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=self.truncation,
            padding=self.padding,
        )

        # Generate text with the model
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_encoded.input_ids,
                max_length=self.max_length,  # Set the maximum length of the generated text
                num_return_sequences=self.n,  # Number of completions to generate for each prompt
                temperature=self.temperature,  # Set the sampling temperature
            )

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Create Generation objects for each generated text
        generations = [Generation(message=BaseMessage(content=text)) for text in generated_texts]

        # Return the result
        return LLMResult(generations=generations)

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
