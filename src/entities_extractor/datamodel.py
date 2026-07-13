from pydantic import BaseModel, Field
from typing import Optional


class ExtractionValue(BaseModel):
    value: str | None = Field(
        description="Entity value extracted"
    )
    attributes: dict = Field(
        {},
        description="Additional attributes related to an entity extraction value"
    )

class Entity(BaseModel):
    type: str = Field(
        description="Type or category of the entity"
    )
    extraction: list[ExtractionValue] = Field(
        description="List of values related to an entity type"
    )
    metadata: dict = Field(
        {},
        description="Metadata extraction related to an entity"
    )