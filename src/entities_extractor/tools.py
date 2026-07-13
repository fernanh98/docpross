from pydantic import BaseModel, Field
from openai import pydantic_function_tool

class Entity(BaseModel):
    name: str = Field(
        description="Name of the entity to extract",
    )
    values: list[str] = Field(
        description="List of values or value of the entity to extract"
    )

extraction_tool = pydantic_function_tool(
    Entity,
    name="extract_entity",
    description="Extracts an entity value"
)