import json
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, NamedTuple

from eagle.schemas.case import Case, EAGLECase
from eagle.schemas.phenotypic_evidence import (
    PhenotypeConfidenceEnum,
    IACategoryEnum,
)
from eagle.schemas.experimental_evidence import (
    ExperimentalEvidence,
)
from eagle.schemas.scores import Scores
from eagle.schemas.genetic_evidence import GeneticEvidence, Variant


# Constants for score tables and guideline references
class ScoreConstants:
    # Default Genetic Score Constants (Table 3)
    DEFAULT_SCORE_DE_NOVO_CANONICAL = 2.0
    DEFAULT_SCORE_DE_NOVO_NON_CANONICAL = 1.0
    DEFAULT_SCORE_DE_NOVO_MISSENSE = 0.5
    DEFAULT_SCORE_DE_NOVO_OTHER = 2.0
    DEFAULT_SCORE_INHERITED_NULL = 1.5
    DEFAULT_SCORE_INHERITED_CANONICAL = 1.5
    DEFAULT_SCORE_INHERITED_NON_CANONICAL = 0.5
    DEFAULT_SCORE_INHERITED_OTHER = 0.1
    
    # Functional Evidence Upgrade Constants (Table 2)
    FUNCTIONAL_UPGRADE_NULL = 0.5
    FUNCTIONAL_UPGRADE_OTHER = 0.4
    
    # Phenotype Adjustment Constants (Figure 2)
    PHENOTYPE_ADJ_DEFAULT_2_LOW = -1.0
    PHENOTYPE_ADJ_DEFAULT_2_MEDIUM = -0.5
    PHENOTYPE_ADJ_DEFAULT_1_5_LOW = -0.5
    PHENOTYPE_ADJ_DEFAULT_1_5_MEDIUM = -0.25
    PHENOTYPE_ADJ_DEFAULT_0_5_LOW = -0.25
    PHENOTYPE_ADJ_DEFAULT_0_5_MEDIUM = -0.1
    
    # Experimental Evidence Constants (Table 4)
    MAX_MODELS_RESCUE_COMBINED = 4.0
    MAX_EXPERIMENTAL_TOTAL = 6.0


# Guideline reference constants
class GuidelineReferences:
    TABLE_3 = "EAGLE Table 3: Genetic Evidence Score Matrix"
    TABLE_2 = "EAGLE Table 2: Genetic Evidence Summary Matrix (Functional Data Upgrade)"
    PHENOTYPE_ADJUSTMENTS = "EAGLE Phenotype Scoring Adjustments (after Figure 2)"
    IA_CHECK = "EAGLE Figure 2: Workflow for Assessing ASD Phenotype (2.2 Cognitive Ability Cautionary Comment)"
    TABLE_4 = "EAGLE Table 4: Experimental Evidence Matrix"


# Step names for trace log
class StepNames:
    INITIAL_CHECK = "Initial Check"
    DEFAULT_GENETIC_SCORE = "1. Default Genetic Score"
    FUNCTIONAL_EVIDENCE = "1.5. Functional Evidence Upgrade"
    PHENOTYPE_ADJUSTMENT = "2. Phenotype Adjustment"
    IA_CHECK = "3. Intellectual Ability Check"
    EXPERIMENTAL_EVIDENCE = "4. Experimental Evidence Scoring"
    FINAL_SCORE = "5. Final Score Calculation"


# Result type for genetic score calculation
class GeneticScoreResult(NamedTuple):
    score: float
    category_description: str
    guideline_reference: str


# Map from evidence type to category for experimental evidence
EVIDENCE_TYPE_TO_CATEGORY = {
    # Function (Max Score: 2)
    "Biochemical Function": "Function",
    "Protein Interaction": "Function",
    "Expression": "Function",
    # Functional Alteration (Max Score: 2)
    "Functional Alteration - Patient cells": "Functional Alteration",
    "Functional Alteration - Non-patient cells": "Functional Alteration",
    # Models (Combined Max with Rescue: 4)
    "Model - Non-human model organism": "Models",
    "Model - Cell culture model": "Models",
    # Rescue (Combined Max with Models: 4)
    "Rescue - Human": "Rescue",
    "Rescue - Non-human model organism": "Rescue",
    "Rescue - Cell culture model": "Rescue",
    "Rescue - Patient cells": "Rescue",
}

# Functional evidence keyword indicators
FUNCTIONAL_EVIDENCE_KEYWORDS = [
    "functional", "function", "impact", "mechanism", "effect"
]

# Terms that indicate a null variant type
NULL_VARIANT_TERMS = [
    "nonsense", "frameshift", "stop_gained", "stop_loss", 
    "start_loss", "null", "lof", "loss-of-function"
]


def _calculate_default_genetic_score(
    genetic_evidence: Optional[GeneticEvidence],
) -> GeneticScoreResult:
    """
    Calculates the default genetic score based on EAGLE Guidelines Table 3.
    This is only the base score before any upgrades or adjustments.

    Args:
        genetic_evidence: The genetic evidence object.

    Returns:
        A GeneticScoreResult containing:
            - The default score based on variant type and inheritance pattern.
            - A description of the classified variant category.
            - A reference to the specific rule in Table 3.
    """
    # Check if valid evidence exists
    if not _is_valid_genetic_evidence(genetic_evidence):
        return GeneticScoreResult(
            score=0.0,
            category_description="Unknown/Insufficient Info",
            guideline_reference="Table 3 (Unable to determine category)",
        )

    # Extract and normalize variant information
    variant_info: Variant = genetic_evidence.variant
    variant_type = variant_info.variant_type.lower()
    impact = variant_info.impact.lower() if variant_info.impact else ""
    inheritance = variant_info.inheritance_pattern.lower()
    
    # Determine variant characteristics
    is_de_novo = inheritance == "de_novo"
    is_canonical_splice = "canonical splice" in variant_type
    is_non_canonical_splice = "non-canonical splice" in variant_type or (
        "splice" in variant_type and not is_canonical_splice
    )
    is_missense = "missense" in variant_type
    is_null = _is_null_variant(variant_type, impact)

    # Apply Table 3 scoring logic
    if is_de_novo:
        return _score_de_novo_variant(is_canonical_splice, is_non_canonical_splice, is_missense)
    else:
        return _score_inherited_variant(is_null, is_canonical_splice, is_non_canonical_splice)


def _is_valid_genetic_evidence(genetic_evidence: Optional[GeneticEvidence]) -> bool:
    """
    Checks if the genetic evidence contains valid variant information.
    
    Args:
        genetic_evidence: The genetic evidence to check.
        
    Returns:
        True if evidence contains valid variant information, False otherwise.
    """
    return (
        genetic_evidence is not None
        and genetic_evidence.variant is not None
        and genetic_evidence.variant.variant_type is not None
        and genetic_evidence.variant.inheritance_pattern is not None
    )


def _is_null_variant(variant_type: str, impact: str) -> bool:
    """
    Determines if a variant is a null (loss-of-function) variant.
    
    Args:
        variant_type: The type of variant.
        impact: The impact description of the variant.
        
    Returns:
        True if the variant is a null variant, False otherwise.
    """
    return any(term in variant_type or term in impact for term in NULL_VARIANT_TERMS)


def _score_de_novo_variant(
    is_canonical_splice: bool, is_non_canonical_splice: bool, is_missense: bool
) -> GeneticScoreResult:
    """
    Scores a de novo variant based on EAGLE Table 3 criteria.
    
    Args:
        is_canonical_splice: Whether the variant is a canonical splice site.
        is_non_canonical_splice: Whether the variant is a non-canonical splice site.
        is_missense: Whether the variant is a missense variant.
        
    Returns:
        A GeneticScoreResult containing the score, category description, and guideline reference.
    """
    if is_canonical_splice:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_DE_NOVO_CANONICAL,
            category_description="Variant is de novo canonical splice site",
            guideline_reference="Table 3 (De Novo Canonical Splice)",
        )
    elif is_non_canonical_splice:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_DE_NOVO_NON_CANONICAL,
            category_description="Variant is de novo non-canonical splice site",
            guideline_reference="Table 3 (De Novo Non-Canonical Splice)",
        )
    elif is_missense:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_DE_NOVO_MISSENSE,
            category_description="Variant is de novo missense",
            guideline_reference="Table 3 (De Novo Missense)",
        )
    else:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_DE_NOVO_OTHER,
            category_description="Variant is de novo (Other/Null)",
            guideline_reference="Table 3 (Variant is de novo)",
        )


def _score_inherited_variant(
    is_null: bool, is_canonical_splice: bool, is_non_canonical_splice: bool
) -> GeneticScoreResult:
    """
    Scores an inherited variant based on EAGLE Table 3 criteria.
    
    Args:
        is_null: Whether the variant is a null variant.
        is_canonical_splice: Whether the variant is a canonical splice site.
        is_non_canonical_splice: Whether the variant is a non-canonical splice site.
        
    Returns:
        A GeneticScoreResult containing the score, category description, and guideline reference.
    """
    if is_null:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_INHERITED_NULL,
            category_description="Variant is predicted/proven null (not de novo)",
            guideline_reference="Table 3 (Predicted/Proven Null)",
        )
    elif is_canonical_splice:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_INHERITED_CANONICAL,
            category_description="Variant is inherited canonical splice site",
            guideline_reference="Table 3 (Inherited Canonical Splice)",
        )
    elif is_non_canonical_splice:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_INHERITED_NON_CANONICAL,
            category_description="Variant is inherited non-canonical splice site",
            guideline_reference="Table 3 (Inherited Non-Canonical Splice)",
        )
    else:
        return GeneticScoreResult(
            score=ScoreConstants.DEFAULT_SCORE_INHERITED_OTHER,
            category_description="Other variant type (not de novo, not null/splice)",
            guideline_reference="Table 3 (Other variant type)",
        )


def _calculate_phenotype_adjustment(
    case: Case, default_score: float
) -> Tuple[float, str, str, Optional[PhenotypeConfidenceEnum]]:
    """
    Calculates phenotype confidence adjustments based on EAGLE guidelines.
    
    Args:
        case: The case with phenotypic evidence.
        default_score: The default genetic score.
        
    Returns:
        A tuple containing:
            - The adjustment to apply to the score.
            - A detailed reason for the adjustment.
            - The applied adjustment rule.
            - The phenotype confidence value.
    """
    adjustment = 0.0
    reason = "No adjustment needed (High Confidence or rule not applicable)."
    adjustment_rule = "N/A"
    phenotype_confidence = None
    
    # Check if phenotypic evidence exists
    if not case.phenotypic_evidence:
        reason = "Phenotypic evidence not provided."
        return adjustment, reason, adjustment_rule, phenotype_confidence
    
    # Get confidence level
    phenotype_confidence = case.phenotypic_evidence.phenotype_confidence
    if not phenotype_confidence:
        reason = "Phenotype confidence level not provided."
        return adjustment, reason, adjustment_rule, phenotype_confidence
    
    # Apply adjustment rules from EAGLE guidelines
    if default_score == ScoreConstants.DEFAULT_SCORE_DE_NOVO_CANONICAL:  # 2.0
        if phenotype_confidence == PhenotypeConfidenceEnum.LOW:
            adjustment = ScoreConstants.PHENOTYPE_ADJ_DEFAULT_2_LOW
            adjustment_rule = "Default 2, Low Confidence -> -1.0"
        elif phenotype_confidence == PhenotypeConfidenceEnum.MEDIUM:
            adjustment = ScoreConstants.PHENOTYPE_ADJ_DEFAULT_2_MEDIUM
            adjustment_rule = "Default 2, Medium Confidence -> -0.5"
    
    elif default_score == ScoreConstants.DEFAULT_SCORE_INHERITED_NULL:  # 1.5
        if phenotype_confidence == PhenotypeConfidenceEnum.LOW:
            adjustment = ScoreConstants.PHENOTYPE_ADJ_DEFAULT_1_5_LOW
            adjustment_rule = "Default 1.5, Low Confidence -> -0.5"
        elif phenotype_confidence == PhenotypeConfidenceEnum.MEDIUM:
            adjustment = ScoreConstants.PHENOTYPE_ADJ_DEFAULT_1_5_MEDIUM
            adjustment_rule = "Default 1.5, Medium Confidence -> -0.25"
    
    elif default_score == ScoreConstants.DEFAULT_SCORE_DE_NOVO_MISSENSE:  # 0.5
        if phenotype_confidence == PhenotypeConfidenceEnum.LOW:
            adjustment = ScoreConstants.PHENOTYPE_ADJ_DEFAULT_0_5_LOW
            adjustment_rule = "Default 0.5, Low Confidence -> -0.25"
        elif phenotype_confidence == PhenotypeConfidenceEnum.MEDIUM:
            adjustment = ScoreConstants.PHENOTYPE_ADJ_DEFAULT_0_5_MEDIUM
            adjustment_rule = "Default 0.5, Medium Confidence -> -0.1"
    
    # Construct reason if adjustment is applied
    if adjustment != 0.0:
        reason = (
            f"Applied {adjustment} due to {phenotype_confidence} confidence "
            f"(based on default score {default_score}). Rule: '{adjustment_rule}'."
        )
    else:
        reason = (
            f"No phenotype adjustment rule applicable for {phenotype_confidence} "
            f"confidence with default score {default_score}."
        )
    
    return adjustment, reason, adjustment_rule, phenotype_confidence


def score_reported_case(
    case: Case,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Scores a case based on EAGLE Guidelines and provides traceability steps.

    Args:
        case: An instance of Case populated with extracted data.

    Returns:
        A tuple containing:
            - The final calculated genetic score for the case (float).
            - A list of dictionaries detailing the scoring steps and warnings,
              referencing EAGLE guidelines.
    """
    trace_log: List[Dict[str, Any]] = []
    current_score = 0.0
    
    # Initial check for valid genetic evidence
    if not case.genetic_evidence or not case.genetic_evidence.variant:
        trace_log.append({
            "step": StepNames.INITIAL_CHECK,
            "outcome": "Error",
            "reason": "Case cannot be scored: Missing genetic evidence or variant information.",
            "score_impact": 0.0,
            "guideline_reference": "N/A - Prerequisite for scoring",
        })
        return 0.0, trace_log
    
    # Step 1: Calculate default genetic score
    genetic_score_result = _calculate_default_genetic_score(case.genetic_evidence)
    default_score = genetic_score_result.score
    current_score = default_score
    
    guideline_reason = f"Default score based on variant classification: '{genetic_score_result.category_description}'."
    
    trace_log.append({
        "step": StepNames.DEFAULT_GENETIC_SCORE,
        "variant_type_input": case.genetic_evidence.variant.variant_type,
        "impact_input": case.genetic_evidence.variant.impact,
        "inheritance_input": case.genetic_evidence.variant.inheritance_pattern,
        "classified_category": genetic_score_result.category_description,
        "score": default_score,
        "reason": guideline_reason,
        "guideline_reference": genetic_score_result.guideline_reference,
    })
    
    # Step 1.5: Apply functional evidence upgrade
    functional_adjustment, functional_reason, functional_details = _assess_functional_evidence(
        case, genetic_score_result.category_description
    )
    
    current_score += functional_adjustment
    has_functional_evidence = functional_adjustment > 0
    
    trace_log.append({
        "step": StepNames.FUNCTIONAL_EVIDENCE,
        "adjustment_applied": functional_adjustment,
        "score_after_step": round(current_score, 3),
        "reason": functional_reason,
        "has_functional_evidence": has_functional_evidence,
        "functional_evidence_details": functional_details if has_functional_evidence else "N/A",
        "guideline_reference": GuidelineReferences.TABLE_2,
    })
    
    # Step 2: Apply phenotype confidence adjustments
    pheno_adjustment, pheno_reason, adjustment_rule, phenotype_confidence = _calculate_phenotype_adjustment(
        case, default_score
    )
    
    if phenotype_confidence:
        # Apply adjustment if one was calculated
        if pheno_adjustment != 0.0:
            current_score += pheno_adjustment
            # Ensure score doesn't go below zero
            if current_score < 0.0:
                pheno_reason += f" Score capped at 0.0 (was {current_score:.2f})."
                current_score = 0.0
        
        trace_log.append({
            "step": StepNames.PHENOTYPE_ADJUSTMENT,
            "confidence_level": phenotype_confidence,
            "adjustment_applied": pheno_adjustment,
            "score_after_step": round(current_score, 3),
            "reason": pheno_reason,
            "guideline_reference": GuidelineReferences.PHENOTYPE_ADJUSTMENTS,
            "adjustment_rule": adjustment_rule,
        })
    else:
        # Log that phenotype adjustment was skipped
        trace_log.append({
            "step": StepNames.PHENOTYPE_ADJUSTMENT,
            "outcome": "Skipped",
            "score_after_step": round(current_score, 3),
            "reason": pheno_reason,  # Will contain info about why it was skipped
            "guideline_reference": GuidelineReferences.PHENOTYPE_ADJUSTMENTS,
            "adjustment_rule": "N/A",
        })
    
    # Step 3: Apply Intellectual Ability (IA) check
    current_score, ia_reason, guideline_rule, ia_warning = _perform_ia_check(case, current_score)
    
    # Determine whether this step was performed or skipped
    if case.phenotypic_evidence and case.phenotypic_evidence.ia_category:
        trace_log.append({
            "step": StepNames.IA_CHECK,
            "ia_category": case.phenotypic_evidence.ia_category,
            "criteria_A_met": bool(case.phenotypic_evidence.phenotype_source_indicator_a),
            "criteria_B_met": bool(case.phenotypic_evidence.phenotype_source_indicator_b),
            "score_after_step": round(current_score, 3),
            "reason": ia_reason,
            "guideline_reference": GuidelineReferences.IA_CHECK,
            "guideline_rule_applied": guideline_rule,
            "warning_added": ia_warning,
        })
    else:
        trace_log.append({
            "step": StepNames.IA_CHECK,
            "outcome": "IA Category Missing (Treated as Insufficient Info)",
            "criteria_A_met": bool(case.phenotypic_evidence.phenotype_source_indicator_a) if case.phenotypic_evidence else False,
            "criteria_B_met": bool(case.phenotypic_evidence.phenotype_source_indicator_b) if case.phenotypic_evidence else False,
            "score_after_step": round(current_score, 3),
            "reason": ia_reason,
            "guideline_reference": GuidelineReferences.IA_CHECK,
            "guideline_rule_applied": guideline_rule,
            "warning_added": ia_warning,
        })
    
    # Round the final genetic score
    final_genetic_score = round(current_score, 3)
    
    logging.info(
        f"Case {case.case_id}: Genetic score calculated: {final_genetic_score:.3f}. Trace log generated."
    )
    
    return final_genetic_score, trace_log


def format_score_output(
    case: Case,
    final_genetic_score: float,
    experimental_score: Optional[float],
    experimental_rationale: Optional[str],
    trace_log: List[Dict[str, Any]],
) -> Scores:
    """
    Formats the scoring results and trace log into the Scores structure.

    Args:
        case: The original Case object used for scoring.
        final_genetic_score: The final calculated genetic score after adjustments.
        experimental_score: The calculated experimental evidence score (can be None).
        experimental_rationale: The rationale for the experimental score (can be None).
        trace_log: The detailed traceability log.

    Returns:
        A Scores object populated with the scoring results.
    """
    # Extract information from trace log
    trace_info = _extract_trace_information(trace_log)
    
    # Calculate final scores
    final_experimental_score = experimental_score if experimental_score is not None else 0.0
    total_case_score = final_genetic_score + final_experimental_score
    
    # Get cognitive assessment results
    cognitive_assessment = _get_cognitive_assessment(case)
    
    # Create the Scores instance
    score_output = Scores(
        case_id=case.case_id,
        # Genetic Score section
        genetic_evidence_score=final_genetic_score,
        genetic_evidence_score_rationale=trace_info.genetic_rationale,
        # Adjustment section
        score_adjustment=trace_info.phenotype_adj,
        score_adjustment_rationale=trace_info.score_adjustment_rationale,
        # Phenotype Quality section
        phenotype_quality=trace_info.phenotype_confidence_value,
        phenotype_quality_rationale=trace_info.phenotype_reason,
        # Cognitive Ability section
        cognitive_assessment_results=cognitive_assessment,
        cognitive_ability_cautionary_comment=(
            trace_info.ia_comment if trace_info.ia_comment else "No cautionary comment provided."
        ),
        # Experimental Score section
        experimental_evidence_score=final_experimental_score,
        experimental_evidence_score_rationale=(
            experimental_rationale if experimental_rationale else "No scorable experimental evidence found or provided."
        ),
        # Final Score
        total_case_score=round(total_case_score, 3),
        # Store Trace Log
        additional_notes=json.dumps(trace_log, indent=2),
    )
    
    return score_output


class TraceInformation(NamedTuple):
    """Container for information extracted from trace log."""
    genetic_rationale: str
    phenotype_adj: float
    phenotype_reason: str
    phenotype_confidence_value: Optional[PhenotypeConfidenceEnum]
    ia_comment: Optional[str]
    functional_evidence_upgrade: float
    functional_evidence_rationale: str
    score_adjustment_rationale: str


def _extract_trace_information(trace_log: List[Dict[str, Any]]) -> TraceInformation:
    """
    Extracts relevant information from the trace log.
    
    Args:
        trace_log: The trace log from scoring.
        
    Returns:
        A TraceInformation object containing extracted data.
    """
    # Initialize values
    genetic_rationale = "N/A"
    phenotype_adj = 0.0
    phenotype_reason = "N/A"
    phenotype_confidence_value = None
    ia_comment = None
    adjustment_reasons = []
    functional_evidence_upgrade = 0.0
    functional_evidence_rationale = "N/A"
    
    # Find relevant steps in trace log
    step1_info = next((step for step in trace_log if step["step"] == StepNames.DEFAULT_GENETIC_SCORE), {})
    step1_5_info = next((step for step in trace_log if step["step"] == StepNames.FUNCTIONAL_EVIDENCE), {})
    step2_info = next((step for step in trace_log if step["step"] == StepNames.PHENOTYPE_ADJUSTMENT), {})
    step3_info = next((step for step in trace_log if step["step"] == StepNames.IA_CHECK), {})
    
    # Process genetic score info
    if step1_info:
        cat = step1_info.get("classified_category", "Unknown")
        ref = step1_info.get("guideline_reference", "N/A")
        genetic_rationale = f"Default score based on variant classification: '{cat}'. (Ref: {ref})"
    
    # Process functional evidence upgrade
    if step1_5_info:
        functional_evidence_upgrade = step1_5_info.get("adjustment_applied", 0.0)
        functional_evidence_rationale = step1_5_info.get("reason", "No functional evidence upgrade information available.")
        
        if functional_evidence_upgrade > 0:
            genetic_rationale += f" With functional evidence upgrade: +{functional_evidence_upgrade} points."
            adjustment_reasons.append(f"Functional Evidence Upgrade: {functional_evidence_rationale}")
    
    # Process phenotype adjustment
    if step2_info:
        phenotype_adj = step2_info.get("adjustment_applied", 0.0)
        phenotype_reason = step2_info.get("reason", "Phenotype adjustment step not fully processed.")
        ref = step2_info.get("guideline_reference")
        rule = step2_info.get("adjustment_rule")
        
        # Build full phenotype reason
        phenotype_reason_full = phenotype_reason
        if ref:
            phenotype_reason_full += f" (Ref: {ref}"
            if rule and rule != "N/A":
                phenotype_reason_full += f", Rule: {rule}"
            phenotype_reason_full += ")"
        phenotype_reason = phenotype_reason_full
        
        # Add to adjustment reasons if applicable
        if phenotype_adj != 0.0:
            adjustment_reasons.append(f"Phenotype Adjustment: {phenotype_reason}")
        
        # Extract phenotype confidence value
        if step2_info.get("confidence_level") and step2_info["confidence_level"] != "N/A":
            try:
                phenotype_confidence_value = PhenotypeConfidenceEnum(step2_info["confidence_level"])
            except ValueError:
                logging.warning(f"Could not parse phenotype confidence '{step2_info['confidence_level']}' back to Enum.")
                phenotype_confidence_value = step2_info["confidence_level"]  # Store as string if parse fails
    
    # Process IA check
    if step3_info:
        ia_reason = step3_info.get("reason", "IA check step not fully processed.")
        ref = step3_info.get("guideline_reference")
        rule = step3_info.get("guideline_rule_applied")
        
        # Build IA log entry
        ia_log_entry = f"IA Check: {ia_reason}"
        if ref:
            ia_log_entry += f" (Ref: {ref}"
            if rule and rule != "N/A":
                ia_log_entry += f", Rule: {rule}"
            ia_log_entry += ")"
        
        # Add warning if applicable
        ia_warning_added = step3_info.get("warning_added")
        ia_comment = ia_warning_added
        
        if "Score reduced to 0" in ia_reason or ia_warning_added:
            adjustment_reasons.append(ia_log_entry)
    
    # Combine all adjustment rationales
    score_adjustment_rationale = (
        "; ".join(adjustment_reasons) if adjustment_reasons 
        else "No score adjustments applied or cautionary comments generated based on phenotype confidence or IA category."
    )
    
    return TraceInformation(
        genetic_rationale=genetic_rationale,
        phenotype_adj=phenotype_adj,
        phenotype_reason=phenotype_reason,
        phenotype_confidence_value=phenotype_confidence_value,
        ia_comment=ia_comment,
        functional_evidence_upgrade=functional_evidence_upgrade,
        functional_evidence_rationale=functional_evidence_rationale,
        score_adjustment_rationale=score_adjustment_rationale
    )


def _get_cognitive_assessment(case: Case) -> str:
    """
    Gets cognitive assessment results from the case if available.
    
    Args:
        case: The case to get cognitive assessment from.
        
    Returns:
        A string containing cognitive assessment results or default message.
    """
    if (
        case.phenotypic_evidence 
        and case.phenotypic_evidence.cognitive_assessment_results
    ):
        return case.phenotypic_evidence.cognitive_assessment_results
    
    return "No cognitive assessment provided."


def score_case(case: Case) -> Scores:
    """
    Scores a case using EAGLE guidelines and returns the results formatted
    according to the Scores schema.

    The scoring process follows these steps:
    1. Calculate the default genetic score based on variant type (Table 3)
    2. Apply functional evidence upgrades if available (Table 2):
       - For null/canonical splice variants: +0.5 for functional evidence
       - For other variant types: +0.4 for functional evidence
    3. Apply phenotype confidence adjustments if needed
    4. Apply intellectual ability (IA) checks and adjustments
    5. Calculate experimental evidence score (Table 4)
    6. Combine genetic and experimental scores for the total case score

    Args:
        case: The Case object to score.

    Returns:
        A Scores object containing the formatted results.
    """
    # Validate the case object using pydantic
    case_object = EAGLECase.model_validate(case)
    
    # Calculate genetic score and get trace log
    genetic_score, trace = score_reported_case(case_object)
    
    # Calculate experimental evidence score
    experimental_score, experimental_rationale = score_experimental_evidence(case_object)
    
    # Format final output
    formatted_output = format_score_output(
        case_object, genetic_score, experimental_score, experimental_rationale, trace
    )
    
    return formatted_output


# --- Experimental Evidence Scoring (Based on Table 4) ---
def score_experimental_evidence(
    case: Case,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Scores experimental evidence based on EAGLE Table 4 guidelines.
    
    Args:
        case: The Case object with experimental evidence.
        
    Returns:
        A tuple containing the calculated experimental score and a rationale string.
    """
    # Validate the experimental evidence data structure
    if not _has_valid_experimental_evidence(case):
        logging.info(f"Case {case.case_id}: No experimental evidence list found or attribute is invalid.")
        return None, "Experimental evidence attribute missing or not a list."
    
    if not case.experimental_evidence:
        logging.info(f"Case {case.case_id}: Experimental evidence list is empty.")
        return 0.0, "No experimental evidence provided in the list."
    
    # Process evidence and calculate scores
    calculated_scores, rationale_parts, evidence_processed_count = _process_experimental_evidence(case)
    
    # If no processable items were found
    if evidence_processed_count == 0:
        logging.info(f"Case {case.case_id}: No scorable experimental evidence items found.")
        return 0.0, "No scorable experimental evidence found in the list."
    
    # Apply scoring caps and calculate final score
    final_score, rationale = _calculate_final_experimental_score(calculated_scores, rationale_parts)
    
    logging.info(f"Case {case.case_id}: Experimental score calculated: {final_score:.2f}")
    return final_score, rationale


def _process_experimental_evidence(case: Case) -> Tuple[Dict[str, float], List[str], int]:
    """
    Processes experimental evidence items and calculates raw scores.
    
    Args:
        case: The case with experimental evidence.
        
    Returns:
        A tuple containing:
            - Dictionary of calculated scores by category.
            - List of rationale parts.
            - Count of evidence items processed.
    """
    # Initialize score tracking
    calculated_scores = {
        "Function": 0.0,
        "Functional Alteration": 0.0,
        "Models": 0.0,
        "Rescue": 0.0,
    }
    rationale_parts = []
    evidence_processed_count = 0
    
    # Process each evidence item
    for evidence_item in case.experimental_evidence:
        # Basic validation
        if not isinstance(evidence_item, ExperimentalEvidence):
            logging.warning(
                f"Case {case.case_id}: Item in experimental_evidence list is not valid: {evidence_item}. Skipping."
            )
            continue
        
        # Check required fields
        if evidence_item.evidence_type is None or evidence_item.score is None:
            logging.warning(
                f"Case {case.case_id}: Skipping evidence item due to missing type or score."
            )
            continue
        
        # Map to category and process
        evidence_type_str = str(evidence_item.evidence_type)
        category = EVIDENCE_TYPE_TO_CATEGORY.get(evidence_type_str)
        
        if category:
            evidence_processed_count += 1
            
            # Add score to the appropriate category
            score = float(evidence_item.score)
            calculated_scores[category] += score
            
            # Build rationale for this item
            rationale_part = _build_evidence_item_rationale(evidence_item, score)
            rationale_parts.append(rationale_part)
        else:
            logging.warning(
                f"Case {case.case_id}: Unknown evidence type '{evidence_type_str}'. Skipping."
            )
    
    return calculated_scores, rationale_parts, evidence_processed_count


def _build_evidence_item_rationale(evidence_item: ExperimentalEvidence, score: float) -> str:
    """
    Builds a rationale string for an evidence item.
    
    Args:
        evidence_item: The evidence item.
        score: The score for this item.
        
    Returns:
        A string describing the evidence and score.
    """
    evidence_type_str = str(evidence_item.evidence_type)
    
    # Use description or fallback to type
    if evidence_item.exp_description:
        item_detail = f"{evidence_item.exp_description}"
    elif evidence_item.rationale:
        item_detail = f"{evidence_item.rationale}"
    else:
        item_detail = evidence_type_str
    
    # Add quote if available
    if evidence_item.exp_quote:
        item_detail += f' (Quote: "{evidence_item.exp_quote[:50]}...")'
    
    return f"{item_detail}: +{score:.2f}"


def _calculate_final_experimental_score(
    calculated_scores: Dict[str, float], rationale_parts: List[str]
) -> Tuple[float, str]:
    """
    Applies score caps and finalizes the experimental evidence score.
    
    Args:
        calculated_scores: Dictionary of raw scores by category.
        rationale_parts: List of rationale parts for each evidence item.
        
    Returns:
        A tuple with the final score and rationale string.
    """
    # Apply cap for Models + Rescue combined (max 4 points)
    models_rescue_raw_score = calculated_scores["Models"] + calculated_scores["Rescue"]
    models_rescue_capped_score = min(models_rescue_raw_score, ScoreConstants.MAX_MODELS_RESCUE_COMBINED)
    
    if models_rescue_raw_score > ScoreConstants.MAX_MODELS_RESCUE_COMBINED:
        cap_msg = (
            f"Models+Rescue subtotal ({models_rescue_raw_score:.2f}) capped at "
            f"{ScoreConstants.MAX_MODELS_RESCUE_COMBINED} (Ref: EAGLE Table 4 Note)."
        )
        rationale_parts.append(cap_msg)
    
    # Calculate total score before overall cap
    total_score_pre_cap = (
        calculated_scores["Function"]
        + calculated_scores["Functional Alteration"]
        + models_rescue_capped_score
    )
    
    # Apply overall cap (max 6 points)
    final_score = min(total_score_pre_cap, ScoreConstants.MAX_EXPERIMENTAL_TOTAL)
    
    if total_score_pre_cap > ScoreConstants.MAX_EXPERIMENTAL_TOTAL:
        cap_msg = (
            f"Total experimental score ({total_score_pre_cap:.2f}) capped at "
            f"{ScoreConstants.MAX_EXPERIMENTAL_TOTAL} (Ref: EAGLE Table 4 Maximum)."
        )
        rationale_parts.append(cap_msg)
    
    # Construct final rationale
    raw_scores_str = (
        f"Raw Scores - Function: {calculated_scores['Function']:.2f}, "
        f"Functional Alteration: {calculated_scores['Functional Alteration']:.2f}, "
        f"Models: {calculated_scores['Models']:.2f}, "
        f"Rescue: {calculated_scores['Rescue']:.2f}"
    )
    
    rationale_prefix = "Experimental Evidence Scoring: "
    
    if not rationale_parts:
        final_rationale = f"{rationale_prefix}Single item contributes {final_score:.2f}. {raw_scores_str}."
    else:
        final_rationale = f"{rationale_prefix}{'; '.join(rationale_parts)}. {raw_scores_str}. Final Score: {final_score:.2f}"
    
    return final_score, final_rationale


def _has_valid_experimental_evidence(case: Case) -> bool:
    """
    Checks if the case has valid experimental evidence.
    
    Args:
        case: The case to check.
        
    Returns:
        True if case has valid experimental evidence, False otherwise.
    """
    return (
        hasattr(case, "experimental_evidence")
        and isinstance(case.experimental_evidence, list)
        and bool(case.experimental_evidence)
    )


def _assess_functional_evidence(
    case: Case, category_desc: str
) -> Tuple[float, str, str]:
    """
    Assesses the functional evidence and calculates score adjustments.
    
    Args:
        case: The case containing experimental evidence.
        category_desc: The variant category description from default scoring.
        
    Returns:
        A tuple containing:
            - The adjustment to apply to the score.
            - A detailed reason for the adjustment.
            - Evidence details as a string.
    """
    adjustment = 0.0
    reason = "No functional evidence upgrade applied."
    evidence_details = ""
    has_evidence = False
    
    # Check if experimental evidence exists and is valid
    if not _has_valid_experimental_evidence(case):
        return adjustment, reason, evidence_details
    
    # Search for functional evidence in experimental data
    for evidence in case.experimental_evidence:
        # Skip invalid evidence
        if not evidence.evidence_type or not evidence.score:
            continue
            
        evidence_type_str = str(evidence.evidence_type)
        evidence_desc = evidence.exp_description if evidence.exp_description else evidence_type_str
        
        # Check if evidence demonstrates functional impact
        if _is_functional_evidence(evidence):
            has_evidence = True
            evidence_details += f"{evidence_desc} (Score: {evidence.score}); "
    
    evidence_details = evidence_details.rstrip("; ")
    
    # Apply adjustment if functional evidence exists
    if has_evidence:
        is_null_variant = _is_null_variant_category(category_desc)
        
        if is_null_variant:
            adjustment = ScoreConstants.FUNCTIONAL_UPGRADE_NULL
            reason = f"Applied +{adjustment} upgrade for functional evidence with null/canonical splice variant: {evidence_details}"
        else:
            adjustment = ScoreConstants.FUNCTIONAL_UPGRADE_OTHER
            reason = f"Applied +{adjustment} upgrade for functional evidence with non-null variant: {evidence_details}"
    
    return adjustment, reason, evidence_details


def _is_functional_evidence(evidence: ExperimentalEvidence) -> bool:
    """
    Determines if experimental evidence demonstrates functional impact.
    
    Args:
        evidence: The experimental evidence to check.
        
    Returns:
        True if the evidence demonstrates functional impact, False otherwise.
    """
    evidence_type_str = str(evidence.evidence_type)
    
    # Direct function-related evidence
    if "Biochemical Function" in evidence_type_str or "Protein Interaction" in evidence_type_str:
        return True
        
    # Functional alteration evidence
    if "Functional Alteration" in evidence_type_str:
        return True
        
    # Model or Rescue evidence with functional impact indicators
    if any(term in evidence_type_str for term in ["Model", "Rescue"]):
        # Check description for functional impact terms
        if evidence.exp_description and any(
            term in evidence.exp_description.lower() for term in FUNCTIONAL_EVIDENCE_KEYWORDS
        ):
            return True
            
        # Check rationale for functional impact terms
        if evidence.rationale and any(
            term in evidence.rationale.lower() for term in FUNCTIONAL_EVIDENCE_KEYWORDS
        ):
            return True
            
    return False


def _is_null_variant_category(category_desc: str) -> bool:
    """
    Determines if a variant category description indicates a null variant.
    
    Args:
        category_desc: The category description.
        
    Returns:
        True if the category indicates a null variant, False otherwise.
    """
    return "null" in category_desc.lower() or "canonical splice" in category_desc.lower()


def _perform_ia_check(
    case: Case, current_score: float
) -> Tuple[float, str, str, Optional[str]]:
    """
    Performs the Intellectual Ability (IA) check and applies adjustments.
    
    Args:
        case: The case with phenotypic evidence.
        current_score: The current score after previous adjustments.
        
    Returns:
        A tuple containing:
            - The adjusted score after IA check.
            - A detailed reason for any adjustments.
            - The guideline rule applied.
            - A warning message if applicable.
    """
    ia_category = None
    criterion_A_met = False
    criterion_B_met = False
    ia_impact_reason = "No score impact or warning needed based on IA category."
    ia_warning = None
    guideline_rule_applied = "N/A"
    
    # Extract criteria from phenotypic evidence
    if case.phenotypic_evidence:
        ia_category = case.phenotypic_evidence.ia_category
        criterion_A_met = bool(case.phenotypic_evidence.phenotype_source_indicator_a)
        criterion_B_met = bool(case.phenotypic_evidence.phenotype_source_indicator_b)
    
    # If no IA category provided, treat as Insufficient Info
    if not ia_category:
        guideline_rule_applied = "2.2.d (Implicit - IA Category not provided)"
        ia_warning = "Uncertainty regarding validity of ASD diagnosis in light of insufficient information regarding intellectual ability."
        ia_impact_reason = f"Intellectual ability category not provided. Applying guideline {guideline_rule_applied} implicitly."
        return current_score, ia_impact_reason, guideline_rule_applied, ia_warning
    
    # Apply guideline rules based on IA category
    if ia_category == IACategoryEnum.C:  # Profound ID
        guideline_rule_applied = "2.2.c (Profound ID)"
        ia_warning = "Case not counted towards evidence in light of profound ID."
        ia_impact_reason = f"IA Category '{ia_category}' (Profound ID). Guideline {guideline_rule_applied} applied."
        
        if current_score > 0:
            ia_impact_reason += f" Score reduced to 0 from {current_score:.3f}."
            current_score = 0.0
        else:
            ia_impact_reason += " Score remains 0."
            
    elif ia_category == IACategoryEnum.B:  # Severe ID
        ia_impact_reason = f"IA Category '{ia_category}' (Severe ID)."
        
        # Check criteria A and B for warning
        if not (criterion_A_met and criterion_B_met):
            guideline_rule_applied = "2.2.b (Severe ID, without criteria A and B)"
            ia_warning = "Some uncertainty regarding validity of ASD diagnosis in light of severe ID and insufficient information on ASD phenotyping methods."
            ia_impact_reason += f" Warning added per guideline {guideline_rule_applied} (Criteria A/B not met)."
        else:
            guideline_rule_applied = "2.2.b (Severe ID, with criteria A and B)"
            ia_impact_reason += f" No warning needed per guideline {guideline_rule_applied} (Criteria A/B met)."
            
    elif ia_category == IACategoryEnum.A:  # No/Mild/Moderate ID
        guideline_rule_applied = "2.2.a (No/Mild/Moderate ID)"
        ia_impact_reason = f"IA Category '{ia_category}'. No cautionary comment required per guideline {guideline_rule_applied}."
        
    elif ia_category == IACategoryEnum.D:  # Insufficient Info
        guideline_rule_applied = "2.2.d (Insufficient Info)"
        ia_warning = "Uncertainty regarding validity of ASD diagnosis in light of insufficient information regarding intellectual ability."
        ia_impact_reason = f"IA Category '{ia_category}'. Warning added per guideline {guideline_rule_applied}."
    
    return current_score, ia_impact_reason, guideline_rule_applied, ia_warning
