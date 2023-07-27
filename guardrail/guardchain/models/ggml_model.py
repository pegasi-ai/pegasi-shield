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
from ctransformers import AutoModelForCausalLM

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

class GGMLHuggingFaceModel():
    """Wrapper around HuggingFace's Transformers language models.

    To use, you should have the `transformers` library installed, and the
    appropriate model and tokenizer should be loaded.

    Any parameters that are valid to be passed to the `generate` call can be passed
    in, even if not explicitly saved on this class.
    """

    model_name: str = "TheBloke/WizardLM-13B-V1.2-GGML"
    """Model name to use."""
    model_file: str = "wizardlm-13b-v1.2.ggmlv3.q4_1.bin"
    """GGML Model Bin File"""
    gpu_layers: int = 50
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
    max_length: int = 1024
    """Tokenizer maximum length."""
    truncation: bool = False
    """Tokenizer truncation"""
    padding: bool = False 
    """Padding truncation"""
    gpu_layers: int = 0
    """GGML """
    
    _model: Optional[AutoModelForCausalLM] = Field(default=None)
    """The LlamaForCausalLM instance for this model."""

    def __init__(
        self,
        model_name: str = None,
        model_file: str = None,
        gpu_layers: int = None,
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

        if model_file is not None:
            self.model_file = model_file
        
        if gpu_layers is not None:
            self.gpu_layers = gpu_layers

        print("Loading in GGML Model with GPU layers")

        self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                model_file=self.model_file,
                gpu_layers=self.gpu_layers
        )

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
            planning: Optional[bool] = True
        ) -> LLMResult:
            prompt = self._construct_prompt_from_message(messages)
            print("Prompt:", prompt)

            # inputs = [msg.content for msg in messages]
            # Concatenate the inputs into a single string
            # concatenated_string = "".join(prompt)
            # Remove leading and trailing whitespaces
            # result = concatenated_string.strip()
            
            text = self.model(prompt)
        
            generation = [Generation(message=BaseMessage(content=text))]
            # Return the result
            return self._create_llm_result(generation=generation, prompt=prompt, stop=stop, planning=planning)

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
         
        ai_response = generation[0].message.content
                  
        print("AI Response: ", ai_response)

        return LLMResult(
                generations=[Generation(message=AIMessage(content=ai_response))],
                llm_output={
                    "token_usage": len(str(generation).split()),
                    "model_name": self.model_name,
                },
        )