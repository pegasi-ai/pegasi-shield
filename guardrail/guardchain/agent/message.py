import enum
from typing import List, Any, Dict, Optional

from pydantic import BaseModel, Field


class MessageType(enum.Enum):
    UserMessage = enum.auto()
    AIMessage = enum.auto()
    SystemMessage = enum.auto()
    FunctionMessage = enum.auto()


class BaseMessage(BaseModel):
    """Message object."""

    content: str
    additional_kwargs: Optional[Dict[str, Any]] = None  # Use Optional with default None

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        raise NotImplementedError("Subclasses must implement this property.")


class UserMessage(BaseMessage):
    """Type of message that is spoken by the human."""

    content: str  # Ensure that content is defined with type str
    example: bool = False

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "user"


class AIMessage(BaseMessage):
    """Type of message that is spoken by the AI."""

    example: bool = False
    function_call: Dict[str, Any] = {}

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "ai"


class SystemMessage(BaseMessage):
    """Type of message that is a system message."""

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "system"


class FunctionMessage(BaseMessage):
    """Type of message that is a function message."""

    name: str

    @property
    def type(self) -> str:
        """Type of the message, used for serialization."""
        return "function"


class ChatMessageHistory(BaseModel):
    messages: List[BaseMessage] = []

    def save_message(self, message: str, message_type: MessageType, **kwargs):
        message_class = {
            MessageType.AIMessage: AIMessage,
            MessageType.UserMessage: UserMessage,
            MessageType.FunctionMessage: FunctionMessage,
            MessageType.SystemMessage: SystemMessage,
        }.get(message_type)

        if message_class:
            self.messages.append(message_class(content=message, **kwargs))

    def format_message(self) -> str:
        string_messages = []
        for m in self.messages:
            role = {
                UserMessage: "User",
                AIMessage: "Assistant",
                SystemMessage: "System",
            }.get(type(m))
            if role:
                string_messages.append(f"{role}: {m.content}")
        return "\n".join(string_messages) + "\n"

    def get_latest_user_message(self) -> UserMessage:
        for message in reversed(self.messages):
            if isinstance(message, UserMessage):
                return message
        return UserMessage(content="n/a")

    def clear(self) -> None:
        self.messages = []
