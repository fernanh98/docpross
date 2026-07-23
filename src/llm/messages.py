from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict

class Message(BaseModel):
    role: str
    content: str | None

class MessageStructured(Message):
    st_response: BaseModel | None
    tool_calls: list[ToolCall]