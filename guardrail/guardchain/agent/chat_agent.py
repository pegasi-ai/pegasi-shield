from __future__ import annotations

import logging
from string import Template
from typing import Any, Dict, List, Optional, Union

from colorama import Fore

from guardrail.guardchain.agent.base_agent import BaseAgent
from guardrail.guardchain.agent.prompts import (
    CLARIFYING_QUESTION_PROMPT_TEMPLATE,
    PLANNING_PROMPT_TEMPLATE,
    SHOULD_ANSWER_PROMPT_TEMPLATE,
    FIX_TOOL_INPUT_PROMPT_TEMPLATE,
)
from guardrail.guardchain.agent.message import BaseMessage, ChatMessageHistory, UserMessage
from guardrail.guardchain.agent.base_agent import ConvoJSONOutputParser, JSONPromptTemplate
from guardrail.guardchain.agent.agent_structs import AgentAction, AgentFinish
from guardrail.guardchain.models.base_model import BaseLanguageModel, Generation
from guardrail.guardchain.tools.base_tool import Tool
from guardrail.guardchain.utils import print_with_color

logger = logging.getLogger(__name__)


class ChatAgent(BaseAgent):
    output_parser: ConvoJSONOutputParser = ConvoJSONOutputParser()
    llm: BaseLanguageModel = None
    prompt_template: JSONPromptTemplate = None
    allowed_tools: Dict[str, Tool] = {}
    tools: List[Tool] = []
    prompt: str = ""

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Optional[List[Tool]] = None,
        output_parser: Optional[ConvoJSONOutputParser] = None,
        prompt_template: str = PLANNING_PROMPT_TEMPLATE,
        input_variables: Optional[List[str]] = None,
        prompt: str = "",
        **kwargs: Any,
    ) -> ChatAgent:
        tools = tools or []
        template = cls.get_prompt_template(
            template=prompt_template,
            input_variables=input_variables,
        )
        allowed_tools = {tool.name: tool for tool in tools}
        _output_parser = output_parser or ConvoJSONOutputParser()
        return cls(
            llm=llm,
            allowed_tools=allowed_tools,
            output_parser=_output_parser,
            prompt_template=template,
            tools=tools,
            prompt=prompt,
            **kwargs,
        )

    def should_answer(
        self,
        should_answer_prompt_template: str = SHOULD_ANSWER_PROMPT_TEMPLATE,
        **kwargs,
    ) -> Optional[AgentFinish]:
        if "history" not in kwargs or not kwargs["history"]:
            return None
        history = kwargs.pop("history")
        inputs = {
            "history": history.format_message(),
            **kwargs,
        }

        def _parse_response(res: str):
            if "yes" in res.lower():
                return AgentFinish(
                    message="Thank you for contacting",
                    log="Thank you for contacting",
                )
            else:
                return None

        prompt = Template(should_answer_prompt_template).substitute(**inputs)
        response = self.llm.generate([UserMessage(content=prompt)]).generations[0].message.content
        return _parse_response(response)

    @staticmethod
    def format_prompt(
        template: JSONPromptTemplate,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> List[BaseMessage]:
        def _construct_scratchpad(actions: List[AgentAction]) -> Union[str, List[BaseMessage]]:
            scratchpad = ""
            for action in actions:
                scratchpad += action.response
            return scratchpad

        thoughts = _construct_scratchpad(intermediate_steps)
        new_inputs = {"agent_scratchpad": thoughts}
        full_inputs = {**kwargs, **new_inputs}
        prompt = template.format_prompt(**full_inputs)
        return prompt

    @staticmethod
    def get_prompt_template(
        template: str = "",
        input_variables: Optional[List[str]] = None,
    ) -> JSONPromptTemplate:
        template = Template(template)
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return JSONPromptTemplate(template=template, input_variables=input_variables)

    def plan(
        self,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ) -> Union[AgentAction, AgentFinish]:
        print_with_color("Planning the next steps...", Fore.LIGHTYELLOW_EX)
        tool_names = ", ".join([tool.name for tool in self.tools])
        tool_strings = "\n\n".join([f"> {tool.name}: \n{tool.description}" for tool in self.tools])
        inputs = {
            "tool_names": tool_names,
            "tools": tool_strings,
            "history": history.format_message(),
            "prompt": self.prompt,
            **kwargs,
        }
        final_prompt = self.format_prompt(self.prompt_template, intermediate_steps, **inputs)
        logger.info(f"\nPlanning Input: {final_prompt[0].content} \n")
        full_output: Generation = self.llm.generate(final_prompt).generations[0]
        agent_output: Union[AgentAction, AgentFinish] = self.output_parser.parse(
            full_output.message
        )

        print(f"Planning output: \n{repr(full_output.message.content)}", Fore.YELLOW)
        if isinstance(agent_output, AgentAction):
            print_with_color(
                f"Plan to take the following action: '{agent_output.tool}'", Fore.LIGHTYELLOW_EX
            )

        return agent_output

    def clarify_args_for_agent_action(
        self,
        agent_action: AgentAction,
        history: ChatMessageHistory,
        intermediate_steps: List[AgentAction],
        **kwargs: Any,
    ):
        print_with_color("Deciding if I need additional clarification...", Fore.LIGHTYELLOW_EX)
        if not self.allowed_tools.get(agent_action.tool):
            return agent_action
        else:
            inputs = {
                "tool_name": agent_action.tool,
                "tool_desp": self.allowed_tools.get(agent_action.tool).description,
                "history": history.format_message(),
                **kwargs,
            }
            clarifying_template = self.get_prompt_template(
                template=CLARIFYING_QUESTION_PROMPT_TEMPLATE
            )
            final_prompt = self.format_prompt(clarifying_template, intermediate_steps, **inputs)
            logger.info(f"\nClarification inputs: {final_prompt[0].content}")
            full_output: Generation = self.llm.generate(final_prompt).generations[0]
            print(f"Clarification outputs: {repr(full_output.message.content)}")
            return self.output_parser.parse_clarification(
                full_output.message, agent_action=agent_action
            )

    def fix_action_input(self, tool: Tool, action: AgentAction, error: str) -> AgentAction:
        prompt = FIX_TOOL_INPUT_PROMPT_TEMPLATE.format(
            tool_description=tool.description, inputs=action.tool_input, error=error
        )
        logger.info(f"\nFixing tool input prompt: {prompt}")
        messages = UserMessage(content=prompt)
        output = self.llm.generate([messages]).generations[0]
        new_tool_inputs = self.output_parser.load_json_output(output.message)

        logger.info(f"\nFixed tool output: {new_tool_inputs}")
        new_action = AgentAction(tool=action.tool, tool_input=new_tool_inputs)
        return new_action
