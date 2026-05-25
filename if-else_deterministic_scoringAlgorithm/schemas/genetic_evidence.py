from typing import Optional, List
from pydantic import BaseModel, Field
from .variant import Variant


class GeneticEvidence(BaseModel):
    case_id: int = Field(description="The ID of the case")
    variant_id: int = Field(description="The ID of the variant")
    score: float = Field(description="The score of the evidence")
    rationale: Optional[str] = Field(description="The rationale for the evidence")
    variant: Optional[Variant] = Field(description="The variant")

    class Config:
        use_enum_values = True
