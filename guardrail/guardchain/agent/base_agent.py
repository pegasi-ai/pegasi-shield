from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Union
from string import Template

from pydantic import BaseModel, Extra, Field
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from transformers import AutoTokenizer, AutoModelForCausalLM

from guardrail.guardchain.agent.message import BaseMessage, UserMessage
from guardrail.guardchain.agent.agent_structs import AgentAction, AgentFinish, AgentOutputParser
from guardrail.guardchain.models.base_model import BaseLanguageModel
from guardrail.guardchain.tools.base_tool import Tool


logger = logging.getLogger(__name__)


class JSONPromptTemplate(BaseModel):
    """Format prompt with string Template and dictionary of variables."""

    template: Template
    """The prompt template."""
    
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True
        validate_assignment = False  # Turn off Pydantic validation errors

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    def format_prompt(self, **kwargs: Any) -> List[BaseMessage]:
        variables = {v: "" for v in self.input_variables}
        variables.update(kwargs)
        prompt = self.template.substitute(**variables)
        return [UserMessage(content=prompt)]


class BaseAgent(BaseModel, ABC):
    output_parser: AgentOutputParser = None
    llm: BaseLanguageModel = None
    tools: Sequence[Tool] = []

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True
        validate_assignment = False  # Turn off Pydantic validation errors

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[Tool],
        prompt: str,
        output_parser: Optional[AgentOutputParser] = None,
        input_variables: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> BaseAgent:
        """Construct an agent from an LLM and tools."""
        pass

    def should_answer(
        self, should_answer_prompt_template: str = "", **kwargs
    ) -> Optional[AgentFinish]:
        """Determine if agent should continue to answer user questions based on the latest user query."""
        return None

    @abstractmethod
    def plan(
        self,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Plan the next step, either taking an action with AgentAction or respond to the user with AgentFinish.
        Args:
            history: Entire conversation history between user and agent including the latest query.
            intermediate_steps: List of AgentAction that has been performed with outputs.
            **kwargs: Key-value pairs from chain, which contains query and other stored memories.

        Returns:
            AgentAction or AgentFinish
        """
        pass

    def clarify_args_for_agent_action(
        self,
        agent_action: AgentAction,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        """
        Ask clarifying question if needed. When the agent is about to perform an action, we could
        use this function with a different prompt to ask clarifying questions for input if needed.
        Sometimes the planning response would already have the clarifying question, but we found
        it is more precise if there is a different prompt just for clarifying args.

        Args:
            agent_action: Agent action about to take.
            history: Conversation history including the latest query.
            intermediate_steps: List of agent actions taken so far.
            **kwargs:

        Returns:
            Either a clarifying question (AgentFinish) or take the planned action (AgentAction)
        """
        pass

    def fix_action_input(
        self, tool: Tool, action: AgentAction, error: str
    ) -> Optional[AgentAction]:
        """If the tool failed due to an error, what should be the fix for inputs."""
        pass

    @staticmethod
    def get_prompt_template(
        prompt: str = "",
        input_variables: Optional[List[str]] = None,
    ) -> JSONPromptTemplate:
        """
        Create a prompt in the style of the zero-shot agent.

        Args:
            prompt: Message to be injected between prefix and suffix.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        pass


class ConvoJSONOutputParser(AgentOutputParser):
    def parse(self, message: BaseMessage, tool_names) -> Union[AgentAction, AgentFinish]:
        response = self.load_json_output(message)

        action_name = response.get("tool", {}).get("name")
        action_args = response.get("tool", {}).get("args")

        if (
            not response.get("thoughts", {}).get("need_use_tool")
            or not action_name
            or action_name not in tool_names
        ):
            output_message = response.get("response").get("ai_response")
            if output_message:
                return AgentFinish(message=output_message, log=output_message)
            else:
                return AgentFinish(message="Sorry, I don't understand", log=output_message)

        return AgentAction(
            tool=action_name,
            tool_input=action_args,
            model_response=response.get("response", ""),
        )

    def parse_clarification(
        self, message: BaseMessage, agent_action: AgentAction
    ) -> Union[AgentAction, AgentFinish]:
        response = self.load_json_output(message)

        has_arg_value = response.get("has_arg_value", "")
        clarifying_question = response.get("clarifying_question", "")

        if "no" in has_arg_value.lower() and clarifying_question:
            return AgentFinish(message=clarifying_question, log=clarifying_question)
        else:
            return agent_action
