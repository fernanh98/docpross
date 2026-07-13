from pydantic import BaseModel


class MessageContentItem(BaseModel):
    type: str  # "text" or "image_url"
    text: str | None = None
    image_url: dict | None = None  # Expected structure: {"url": "..."} or {"url": "data:image/jpeg;base64,..."}


class ChatMessage(BaseModel):
    role: str
    content: list[MessageContentItem] | str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    stream: bool | None = False

