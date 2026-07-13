import operator
from logging import Logger
from typing import TypedDict, Any, Annotated

from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from src.langgraph.base_graph import BaseGraph, NodeRegistry
from src.llm.base_llm import BaseLLM
from src.llm.implementations import LLM
from src.entities_extractor.extraction import (
    extract_with_langextract,
    extract_with_simple_llm
)

class EntitiesState(TypedDict):
    metadata: Annotated[dict, operator.ior]
    text: str
    entity: str


class EntitiesExtractorGraph(BaseGraph):
    _node_registry = NodeRegistry()

    def __init__(
        self,
        llm_model: LLM,
        logger: Logger = Logger("default")
    ):
        super().__init__(llm_model, logger)

    def create_graph(self) -> CompiledStateGraph:
        builder = StateGraph(EntitiesState)
        self._add_nodes(builder)
        builder.add_conditional_edges(START, self.perform_parallel_extractions)
        builder.add_edge("extract_metadata", END)
        return builder.compile()
    
    @_node_registry.node("set_extractions")
    def perform_parallel_extractions(
        self,
        state: EntitiesState
    ) -> list[Send]:
        entities_to_extract = [
            Send(
                "extract_metadata",
                {
                   "entity": entity,
                   "text": state["text"],
                   "metadata": state["metadata"]
                }
            )
            for entity in state["metadata"]
        ]
        return entities_to_extract
    
    @_node_registry.node()
    def extract_metadata(
        self,
        state: EntitiesState
    ) -> dict:
        """
        Performs the extraction of one metadata concept or entity
        """
        metadata = state["metadata"]
        entitity_key = state["entity"]
        entity_name = metadata[entitity_key]["name"]
        entity_description = metadata[entitity_key]["description"]
        # entity_keywords = metadata[entitity_key]["keywords"]
        text = state["text"]
        entity = extract_with_simple_llm(
            entity_name,
            entity_description,
            text,
            self.llm_model
        )
        metadata[entitity_key]["values"] = [
            {
                "value": ext.value,
                "attributes": ext.attributes
            }
             for ext in entity.extraction
        ]
        return {"metadata": metadata}
    
    def invoke(
        self,
        metadata: dict,
        text: str,
        **kwargs
    ) -> dict[str, Any] | Any:
        return self.graph.invoke(
            {
                "metadata": metadata,
                "text": text
            }
        )
    

# from transformers import LightOnOcrConditionalGeneration, LightOnOcrProcessor
# from transformers import AutoProcessor, AutoModelForMultimodalLM, AutoModelForSeq2SeqLM, AutoModelForImageTextToText

# lighton_processor = AutoProcessor.from_pretrained("lightonai/LightOnOCR-2-1B")
# lighton_model = AutoModelForImageTextToText.from_pretrained(
#     "lightonai/LightOnOCR-2-1B",
#     dtype="auto",
#     device_map="auto"
# )
# lighton_processor.apply_chat_template()