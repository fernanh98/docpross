from typing import Annotated, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from logging import Logger

from langgraph.graph import StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from src.llm.base_llm import BaseLLM


@dataclass
class NodeInfo:
    name: str
    function_name: str
    tool_node: bool


class NodeRegistry:
    """
    Class to registry all the graph nodes. It allows to declare nodes with a decorator and automatically
    include them into the langgraph workflow
    """

    def __init__(self):
        self.nodes: list[NodeInfo] = []

    def node(
        self, 
        name: str | list[str] | None = None,
        tool_node: bool = False
    ) -> Callable:
        def decorator(func: Callable):
            node_name = name or func.__name__
            if isinstance(node_name, str):
                self.nodes.append(
                    NodeInfo(
                        name=node_name,
                        function_name=func.__name__,
                        tool_node=tool_node
                    )
                )
            elif isinstance(node_name, list):
                for node in node_name:
                    self.nodes.append(
                        NodeInfo(
                            name=node,
                            function_name=func.__name__,
                            tool_node=tool_node
                        )
                    )
            return func
        return decorator
    


class BaseGraph(ABC):
    _node_registry = NodeRegistry()

    def __init__(
        self,
        llm_model: BaseLLM,
        logger: Logger = Logger("default")
    ):
        self.llm_model: BaseLLM = llm_model
        self.graph: CompiledStateGraph = self.create_graph()
        self.logger = logger

    def _add_nodes(
        self,
        builder: StateGraph
    ) -> None:
        for node in self._node_registry.nodes:
            if node.tool_node:
                builder.add_node(node.name, ToolNode([getattr(self, node.function_name)]))
            else:
                builder.add_node(node.name, getattr(self, node.function_name))

    @abstractmethod
    def create_graph(self) -> CompiledStateGraph:
        pass

    @abstractmethod
    def invoke(
        self, 
        **kwargs
    ) -> dict[str, Any] | Any:
        pass
        

# class Agent(BaseAgent):
#     _node_registry = NodeRegistry()

#     def __init__(
#         self,
#         llm_model: LLM,
#         logger: Logger = Logger("default"),
#         ...
#     ):
#         ...
#         super().__init__(llm_model, logger)

#     def create_graph(self) -> CompiledStateGraph:
#         builder = StateGraph(State)
#         self._add_nodes(builder)
#         ...
#         return builder.compile()