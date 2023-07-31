from typing import Any

from guardrail.guardchain.tools.base_tool import Tool

class HandOffToAgent(Tool):
    name: str = "Hand off" # Add type annotation 'str'
    description: str = "Hand off to a human agent"  # Add type annotation 'str'
    handoff_msg: str = "Let me hand you off to an agent now"  # Add type annotation 'str'

    def _run(self, *args: Any, **kwargs: Any) -> str:
        return self.handoff_msg
