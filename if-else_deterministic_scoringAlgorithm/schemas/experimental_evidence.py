import logging
from typing import Optional, List
from enum import Enum

from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExperimentalEvidenceTypeEnum(str, Enum):
    """Based on EAGLE Guidelines Table 4"""

    BIOCHEMICAL_FUNCTION = "Biochemical Function"
    PROTEIN_INTERACTION = "Protein Interaction"
    EXPRESSION = "Expression"
    FUNCTIONAL_ALTERATION_PATIENT_CELLS = "Functional Alteration - Patient cells"
    FUNCTIONAL_ALTERATION_NON_PATIENT_CELLS = (
        "Functional Alteration - Non-patient cells"
    )
    MODEL_NON_HUMAN = "Model - Non-human model organism"
    MODEL_CELL_CULTURE = "Model - Cell culture model"
    RESCUE_HUMAN = "Rescue - Rescue in human"
    RESCUE_NON_HUMAN = "Rescue - Rescue in non-human model organism"
    RESCUE_CELL_CULTURE = "Rescue - Rescue in cell culture model"
    RESCUE_PATIENT_CELLS = "Rescue - Rescue in patient cells"


class ExperimentalEvidence(BaseModel):
    evidence_type: ExperimentalEvidenceTypeEnum = Field(
        description="The type of experimental evidence."
    )
    exp_description: Optional[str] = Field(
        description="A description of the experiment."
    )
    exp_quote: Optional[str] = Field(
        description="A relevant quote from the experiment source."
    )

    quotes: Optional[List[str]] = Field(
        description="Direct quotes from the paper supporting the experimental evidence, including details about experimental methods, results, functional effects, model systems used, and other relevant experimental findings. Must extract the complete sentence from the paper where experimental evidence details are mentioned for the specific experimental evidence being extracted."
    )

    class Config:
        use_enum_values = True


class ExperimentalEvidenceWithScore(ExperimentalEvidence):
    score: Optional[float] = Field(
        description="The score of the experimental evidence."
    )
    rationale: Optional[str] = Field(
        description="The rationale for the score of the experimental evidence."
    )

    class Config:
        use_enum_values = True
