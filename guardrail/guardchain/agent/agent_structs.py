import json
from abc import abstractmethod
from typing import Any, Dict, Union, List

from pydantic import BaseModel

from guardrail.guardchain.models.base_model import Generation, BaseLanguageModel
from guardrail.guardchain.agent.message import BaseMessage, UserMessage
from guardrail.guardchain.chain import constants
from guardrail.guardchain.errors import OutputParserException


class AgentAction(BaseModel):
    """Agent's action to take."""

    tool: str
    tool_input: Union[str, dict]
    tool_output: str = ""
    log: str = ""
    model_response: str = ""

    @property
    def response(self) -> str:
        """Get the message to be stored in memory and shared with the next prompt."""
        if self.model_response and not self.tool_output:
            return self.model_response
        return (
            f"Outputs from using tool '{self.tool}' for inputs {self.tool_input} "
            f"is '{self.tool_output}'\n"
        )


class AgentFinish(BaseModel):
    """Agent's return value."""

    message: str
    log: str
    intermediate_steps: List[AgentAction] = []

    def format_output(self) -> Dict[str, Any]:
        final_output = {
            "message": self.message,
            constants.INTERMEDIATE_STEPS: self.intermediate_steps,
        }
        return final_output


class AgentOutputParser(BaseModel):
    @staticmethod
    def load_json_output(message: BaseMessage) -> Dict[str, Any]:
        """If the message contains a json response, try to parse it into dictionary"""
        text = message.content
        clean_text = ""

        try:
            clean_text = text[text.index("{") : text.rindex("}") + 1].strip()
            response = json.loads(clean_text)
        except Exception:
            llm = BaseLanguageModel("OpenAssistant/falcon-7b-sft-mix-2000")
            message = [
                UserMessage(
                    content=f"""Fix the following json into correct format
                                ```json
                                {clean_text}
                                ```
                            """
                )
            ]
            full_output: Generation = llm.generate(message).generations[0]
            response = json.loads(full_output.message.content)

        return response

    @abstractmethod
    def parse(self, message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse text into agent action/finish."""

    def parse_clarification(
        self, message: BaseMessage, agent_action: AgentAction
    ) -> Union[AgentAction, AgentFinish]:
        """Parse clarification outputs"""
        return agent_action
