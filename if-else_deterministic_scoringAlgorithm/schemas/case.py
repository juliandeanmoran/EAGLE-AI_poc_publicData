import logging
from pydantic import BaseModel, Field

from .variant import VariantWithQuotes
from .genetic_evidence import GeneticEvidence
from .experimental_evidence import (
    ExperimentalEvidence,
    ExperimentalEvidenceWithScore,
)
from .phenotypic_evidence import (
    PhenotypicEvidence,
    PhenotypicEvidenceWithQuotes,
)


from typing import Optional, List
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SexEnum(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    UNKNOWN = "Unknown"


class EAGLECase(BaseModel):
    """
    Base Pydantic schema for the Case model, mirroring SQLAlchemy fields.
    """

    case_id: str = Field(
        description="Unique identifier for the case within the study or database"
    )
    description: str = Field(
        description="Comprehensive description of the case including key clinical features, symptoms, diagnoses, and relevant medical history that provides context for the genetic findings"
    )
    cohort_name: Optional[str] = Field(
        description="Name of the study cohort or group the case belongs to"
    )
    age: Optional[str] = Field(
        description="Age of the individual at time of assessment or diagnosis"
    )
    sex: Optional[SexEnum] = Field(
        description="Biological sex of the individual (e.g., male, female, other)"
    )
    ethnicity: Optional[str] = Field(
        description="Ethnic background or ancestry of the individual"
    )
    family_history: Optional[str] = Field(
        description="Relevant family medical history, particularly regarding genetic conditions"
    )
    notes: Optional[str] = Field(
        description="Additional clinical notes or observations not captured in other fields"
    )

    genetic_evidence: Optional[GeneticEvidence] = Field(
        description="Genetic evidence supporting the case's genetic findings"
    )
    experimental_evidence: Optional[List[ExperimentalEvidenceWithScore]] = Field(
        description="Laboratory or experimental evidence supporting the case's genetic findings"
    )
    phenotypic_evidence: Optional[PhenotypicEvidence] = Field(
        description="Clinical phenotype information including symptoms, diagnostic criteria, and assessment results"
    )

    class Config:
        use_enum_values = True


class Case(BaseModel):
    """
    A structured representation of a case with genetic variants and autism features.

    This schema captures detailed information about individual cases extracted from
    scientific literature that have both genetic variants in a specific gene and
    autism spectrum disorder (ASD) features.
    """

    case_id: str = Field(..., description="Unique identifier for the case")
    gene_symbol: str = Field(..., description="Gene symbol associated with the case")
    description: str = Field(
        description="Quoted description of the case from the paper"
    )
    cohort_name: Optional[str] = Field(
        description="Name of the study cohort or group the case belongs to"
    )
    age: Optional[str] = Field(
        description="Age of the individual at time of assessment or diagnosis"
    )
    sex: Optional[SexEnum] = Field(
        description="Biological sex of the individual (e.g., male, female, other)"
    )
    ethnicity: Optional[str] = Field(
        description="Ethnic background or ancestry of the individual"
    )
    family_history: Optional[str] = Field(
        description="Relevant family medical history, particularly regarding genetic conditions"
    )
    notes: Optional[str] = Field(
        description="Additional clinical notes or observations not captured in other fields"
    )

    quotes: Optional[List[str]] = Field(
        description="Direct quotes from the paper describing the case, including all relevant clinical details, presentation, history, and other information as presented in the original text. Must extract the complete sentence from the paper where case details are mentioned for the specific case being extracted."
    )

    variant: Optional[VariantWithQuotes] = Field(
        description="Genetic variant(s) identified in the case, including genomic coordinates and annotations"
    )
    experimental_evidence: Optional[List[ExperimentalEvidence]] = Field(
        description="Laboratory or experimental evidence supporting the case's genetic findings"
    )
    phenotypic_evidence: Optional[PhenotypicEvidenceWithQuotes] = Field(
        description="Clinical phenotype information including symptoms, diagnostic criteria, and assessment results"
    )

    asd_phenotype_confidence: str = Field(
        ..., description="Confidence level for ASD phenotype (High/Medium/Low)"
    )


class CaseList(BaseModel):
    """
    A collection of cases extracted from a scientific paper.

    This model is the main return type for the case extraction process, containing
    all identified cases from the paper that have both genetic variants and autism features.
    """

    cases: List[Case] = Field(..., description="List of cases extracted from the paper")


class ExtractionPlan(BaseModel):
    """
    The extraction plan and identified relevant genes from a scientific paper.

    This model is returned by the planner agent and contains both the detailed
    extraction plan and a list of relevant genes identified in the paper.
    """

    extraction_plan: str = Field(
        ..., description="Detailed plan for extracting case information"
    )


class RelevantGenes(BaseModel):
    """
    A list of relevant genes extracted from a scientific paper.
    """

    relevant_genes: List[str] = Field(
        ..., description="List of relevant genes extracted from the paper"
    )
