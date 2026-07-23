import textwrap

import langextract as lx
from langextract.core import data

from src.entities_extractor.extraction_prompts import extraction_system_prompt
from src.entities_extractor.datamodel import Entity, ExtractionValue
from src.llm.implementations import LLM
from src.entities_extractor.tools import extraction_tool
from src.llm.messages import MessageStructured

from langextract.factory import ModelConfig


def extract_with_langextract(
    entity_name: str,
    entity_description: str,
    text: str,
    examples: list,
    llm_model: LLM
) -> Entity:
    """
    Extracts an entity from text using the langextract library.

    Args:
        entity_name: The name of the entity to extract.
        entity_description: A description of the entity.
        text: The input text to extract from.

    Returns:
        Entity: The extracted entity information.
    """
    entity_extraction_prompt = extraction_system_prompt.format(
        entity_name=entity_name,
        entity_description=entity_description
    )
    prompt = textwrap.dedent(entity_extraction_prompt)
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        config=ModelConfig(
            model_id=llm_model.model,
            provider="openai",
            provider_kwargs={
                "api_key": llm_model.api_key, 
                "base_url": llm_model.base_url
            },
        ),
        debug=True
    )
    return format_langextract_results(entity_name, result)

def extract_with_simple_llm(
    entity_name: str,
    entity_description: str,
    text: str,
    llm_model: LLM
) -> Entity:
    """
    Extracts an entity from text using a structured LLM call with tools.

    Args:
        entity_name: The name of the entity to extract.
        entity_description: A description of the entity.
        text: The input text to extract from.
        llm_model: The LLM implementation to use for extraction.

    Returns:
        Entity: The extracted entity information.
    """
    entity_extraction_prompt = extraction_system_prompt.format(
        entity_name=entity_name,
        entity_description=entity_description
    )
    messages = [
        dict(role="system", content=entity_extraction_prompt),
        dict(role="user", content=text)
    ]
    ai_response = llm_model.chat_structured(
        messages,
        tools=[extraction_tool],
        tool_choice="extract_entity",
        response_format=None
    )
    return format_simplellm_results(entity_name, ai_response)
    
    
def format_simplellm_results(
    entity_type: str,
    simplellm_result: MessageStructured
) -> Entity:
    """
    Formats the result from a simple LLM extraction call into an Entity object.

    Args:
        entity_type: The type of entity being extracted.
        simplellm_result: The structured response from the LLM.

    Returns:
        Entity: The resulting Entity object.
    """
    if simplellm_result.st_response:
        return simplellm_result.st_response
    elif simplellm_result.tool_calls:
        extractions = []
        for tool in simplellm_result.tool_calls:
            if tool.name == "extract_entity":
                for value in tool.arguments["values"]:
                    extractions.append(
                        ExtractionValue(
                            value=value,
                            attributes={}
                        )
                    )
        return Entity(
            type=entity_type,
            extraction=extractions,
            metadata={}
        )
    else:
        return Entity(
            type=entity_type,
            extraction=[],
            metadata={}
        )

def format_langextract_results(
    entity_type: str,
    langextract_result: list[data.AnnotatedDocument] | data.AnnotatedDocument
) -> Entity:
    """
    Formats the result from a langextract call into an Entity object.

    Args:
        entity_type: The type of entity being extracted.
        langextract_result: The result from langextract (list of documents or a single document).

    Returns:
        Entity: The resulting Entity object.
    """
    
    if not isinstance(langextract_result, list):
        documents = [langextract_result]
    else:
        documents = langextract_result

    extraction_values = []
    for doc in documents:
        if doc.extractions is not None:
            for ext in doc.extractions:
                if ext.extraction_class == entity_type:
                    val = ExtractionValue(
                        value=getattr(ext, "extraction_text", None),
                        attributes={
                            "char_interval": ext.char_interval
                        } | getattr(ext, "attributes", {})
                    )
                    extraction_values.append(val)
    return Entity(
        type=entity_type,
        extraction=extraction_values,
        metadata={}
    )