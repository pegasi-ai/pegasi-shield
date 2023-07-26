import logging
import inspect
import traceback
from typing import Dict, Union

from colorama import Fore
from guardrail.guardchain.agent.message import MessageType
from guardrail.guardchain.agent.agent_structs import AgentAction, AgentFinish
from guardrail.guardchain.chain.base_chain import BaseChain
from guardrail.guardchain.errors import ToolRunningError
from guardrail.guardchain.tools.base_tool import Tool
from guardrail.guardchain.tools.simple_handoff import HandOffToAgent

logger = logging.getLogger(__name__)


class Chain(BaseChain):
    """
    Default chain with take_next_step implemented.
    It handles a few common error cases with the agent, such as taking repeated action with the same
    inputs and whether the agent should continue the conversation.
    """

    return_intermediate_steps: bool = False
    handle_parsing_errors: bool = True  # Add type annotation 'bool'
    graceful_exit_tool: Tool = HandOffToAgent()

    def handle_repeated_action(self, agent_action: AgentAction) -> AgentFinish:
        print(f"Action taken before: {agent_action.tool}, " f"input: {agent_action.tool_input}")
        if agent_action.model_response:
            return AgentFinish(
                message=agent_action.response,
                log=f"Action taken before: {agent_action.tool}, "
                f"input: {agent_action.tool_input}",
            )
        else:
            print("No response from the agent. Gracefully exit due to repeated action")
            return AgentFinish(
                message=self.graceful_exit_tool.run(),
                log=f"Gracefully exit due to repeated action",
            )

    def take_next_step(
        self,
        name_to_tool_map: Dict[str, Tool],
        inputs: Dict[str, str],
    ) -> Union[AgentFinish, AgentAction]:
        """
        How the agent determines the next step after observing the inputs and intermediate steps.
        Args:
            name_to_tool_map: a map of tool name to the actual tool object.
            inputs: a dictionary of all inputs, such as user query, past conversation, and
                tool outputs.

        Returns:
            Either an AgentFinish to respond to the user or an AgentAction to take the next action.
        """

        try:
            # Call the LLM to see what to do.
            output = self.agent.plan(
                **inputs,
            )
        except Exception as e:
            if not self.handle_parsing_errors:
                raise e
            tool_output = f"Invalid or incomplete response due to {e}"
            traceback.print_exc()  # This will print the traceback with file name and line number.            

            current_frame = inspect.currentframe()
            line_number = current_frame.f_lineno
            file_name = inspect.getfile(current_frame)
            print(tool_output)
            print(f"File: {file_name}, Line: {line_number}, {tool_output}")
            output = AgentFinish(message=self.graceful_exit_tool.run(), log=tool_output)
            return output

        if isinstance(output, AgentAction):
            output = self.agent.clarify_args_for_agent_action(output, **inputs)

        # If the agent plans to respond with AgentFinish or there is a clarifying question, respond to
        # the user by returning AgentFinish.
        if isinstance(output, AgentFinish):
            return output

        if isinstance(output, AgentAction):
            tool_output = ""
            # Check if the tool is supported.
            if output.tool in name_to_tool_map:
                tool = name_to_tool_map[output.tool]

                # How to handle the case where the same action with the same input is taken before.
                if output.tool_input == self.memory.load_memory(tool.name):
                    return self.handle_repeated_action(output)

                self.memory.save_memory(tool.name, output.tool_input)
                # We then call the tool on the tool input to get a tool_output.
                try:
                    tool_output = tool.run(output.tool_input)
                except ToolRunningError as e:
                    new_agent_action = self.agent.fix_action_input(tool, output, error=str(e))
                    if new_agent_action and new_agent_action.tool_input != output.tool_input:
                        tool_output = tool.run(output.tool_input)

                print(
                    f"Took action '{tool.name}' with inputs '{output.tool_input}', "
                    f"and the tool_output is {tool_output}"
                )
            else:
                tool_output = f"Tool {output.tool} is not supported"

            output.tool_output = tool_output
            return output
        else:
            raise ValueError(f"Unsupported action: {type(output)}")
