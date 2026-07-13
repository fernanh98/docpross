from ollama import chat
from ollama import ChatResponse

def call_ollama(model: str, messages: list[dict]) -> str | None:
    response: ChatResponse = chat(model=model, messages=messages)
    return response.message.content