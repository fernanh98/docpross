from abc import ABC, abstractmethod

from src.llm.messages import Message, MessageStructured


class BaseLLM(ABC):
    def __init__(
            self,
            model: str,
            api_key: str
        ):
        self.model: str = model
        self.api_key: str = api_key

    @abstractmethod
    def chat(
        self,
        messages,
        *args, 
        **kwargs
    ) -> Message:
        pass

    @abstractmethod
    def chat_structured(
        self,
        messages,
        *args, 
        **kwargs
    ) -> MessageStructured:
        pass