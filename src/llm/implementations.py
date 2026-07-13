

from openai import OpenAI

from src.llm.base_llm import BaseLLM
from src.llm.messages import Message, MessageStructured, ToolCall


class LLM(BaseLLM):
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "none"
    ):
        super().__init__(
            model=model,
            api_key=api_key
        )
        self.base_url: str = base_url
        self._set_openai_client()
    
    def _set_openai_client(self) -> None:
        self.client: OpenAI = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def chat(
        self,
        messages
    ) -> Message:
        self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )
        pass

    def chat_structured(
        self,
        messages,
        tools,
        tool_choice,
        response_format
    ) -> MessageStructured:
        if tools:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice
            )
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                tools = []
                for call in tool_calls:
                    tools.append(
                        ToolCall(
                            name=call.function.name,
                            arguments=eval(call.function.arguments)
                        )
                    )
                return MessageStructured(
                    role="assistant",
                    content=response.choices[0].message.content,
                    st_response=None,
                    tool_calls=tools
                )
            else:
                return MessageStructured(
                    role="assistant",
                    content=response.choices[0].message.content,
                    st_response=None,
                    tool_calls=[]
                )

    