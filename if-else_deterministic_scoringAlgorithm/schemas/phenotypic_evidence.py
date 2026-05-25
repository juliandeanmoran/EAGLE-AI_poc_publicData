from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class PhenotypeConfidenceEnum(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class IACategoryEnum(str, Enum):
    A = "a"  # No ID, mild ID, or moderate ID
    B = "b"  # Severe ID
    C = "c"  # Profound ID
    D = "d"  # No, or insufficient, information on intellectual ability


class PhenotypicEvidence(BaseModel):
    diagnostic_criteria_used: Optional[str] = Field(
        description="The diagnostic criteria used"
    )
    phenotype_source_indicator_a: Optional[bool] = Field(
        description="Guideline A: Expert clinician or multidisciplinary team assigned diagnosis",
    )
    phenotype_source_indicator_b: Optional[bool] = Field(
        description="Guideline B: Validated assessment methods used (e.g. ADI-R, ADOS)",
    )
    phenotype_source_indicator_c: Optional[bool] = Field(
        description="Guideline C: Explicit mention of 'meeting DSM or ICD [criteria]'",
    )
    phenotype_source_indicator_d: Optional[bool] = Field(
        description="Guideline D: Description of symptoms in social/communicative AND repetitive domain indicative of ASD",
    )
    phenotype_source_indicator_e: Optional[bool] = Field(
        description="Guideline E: Only mentions 'ASD', 'Autism', 'PDD', etc.",
    )
    phenotype_source_indicator_f: Optional[bool] = Field(
        description="Guideline F: Only mentions 'Autistic features', 'traits', etc. or insufficient description",
    )
    phenotype_confidence: Optional[PhenotypeConfidenceEnum] = Field(
        description="Confidence level in the ASD diagnosis based on available evidence (High, Medium, Low)"
    )
    assessment_tools_used: Optional[str] = Field(
        description="Specific diagnostic instruments or assessment tools used for evaluation (e.g., ADOS, ADI-R, CARS)"
    )
    age_at_diagnosis: Optional[str] = Field(
        description="Age of the individual when ASD diagnosis was first established"
    )
    core_asd_symptoms: Optional[str] = Field(
        description="Description of primary autism spectrum disorder symptoms observed in social communication and repetitive behaviors"
    )
    cognitive_assessment_results: Optional[str] = Field(
        description="Results from IQ testing or other cognitive assessments, including scores and interpretation"
    )
    ia_category: Optional[IACategoryEnum] = Field(
        description="Intellectual ability category classification (a: No/mild/moderate ID, b: Severe ID, c: Profound ID, d: Insufficient information)"
    )
    cognitive_ability_cautionary_comment: Optional[str] = Field(
        description="Important notes or limitations regarding the cognitive assessment results or interpretation"
    )
    developmental_milestones: Optional[str] = Field(
        description="Information about achievement of key developmental stages, including any delays or regressions"
    )
    comorbidities: Optional[str] = Field(
        description="Additional medical or psychiatric conditions present alongside ASD diagnosis"
    )
    phenotypes: Optional[str] = Field(
        description="Observable physical, behavioral, or biochemical characteristics associated with the condition"
    )

    class Config:
        use_enum_values = True


class PhenotypicEvidenceWithScore(PhenotypicEvidence):
    score: Optional[float] = Field(description="The score of the phenotypic evidence.")
    rationale: Optional[str] = Field(
        description="The rationale for the score of the phenotypic evidence."
    )

    class Config:
        use_enum_values = True


class PhenotypicEvidenceWithQuotes(PhenotypicEvidence):
    quotes: Optional[List[str]] = Field(
        description="Direct quotes from the paper supporting the phenotypic evidence, including diagnostic criteria, assessment tools, age at diagnosis, core ASD symptoms, cognitive assessment results, intellectual ability category, and other relevant information. Must extract the complete sentence from the paper where phenotypic evidence details are mentioned for the specific phenotypic evidence being extracted."
    )

    class Config:
        use_enum_values = True
