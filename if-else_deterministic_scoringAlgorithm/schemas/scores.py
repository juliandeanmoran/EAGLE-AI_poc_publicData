from pydantic import BaseModel
from typing import Optional


class Scores(BaseModel):
    case_id: str
    genetic_evidence_score: float
    genetic_evidence_score_rationale: str
    score_adjustment: float
    score_adjustment_rationale: str
    cognitive_assessment_results: str
    cognitive_ability_cautionary_comment: str
    phenotype_quality: Optional[str]
    phenotype_quality_rationale: str
    experimental_evidence_score: float
    experimental_evidence_score_rationale: str
    total_case_score: float
    additional_notes: str
