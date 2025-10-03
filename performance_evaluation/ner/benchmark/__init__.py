"""EAGLE benchmark utilities package."""

from .models import MatchResult
from .text import TextNormalizer
from .publication import PublicationParser
from .similarity import SimilarityCalculator
from .matching import PublicationMatcher, CaseMatcher
from .processing import DataProcessor
from .scoring import ScoreComparator, ScoreVerifier
from .visualization import compare_manual_vs_automatic_scores_with_viz
from .f1 import (
    PhenotypeConfidenceParser,
    VariantNormalizer,
    F1Calculator,
    F1Evaluator,
    evaluate_f1_between_manual_and_automatic,
)

__all__ = [
    "MatchResult",
    "TextNormalizer",
    "PublicationParser",
    "SimilarityCalculator",
    "PublicationMatcher",
    "CaseMatcher",
    "DataProcessor",
    "ScoreComparator",
    "ScoreVerifier",
    "PhenotypeConfidenceParser",
    "VariantNormalizer",
    "F1Calculator",
    "F1Evaluator",
    "evaluate_f1_between_manual_and_automatic",
    "compare_manual_vs_automatic_scores_with_viz",
]
