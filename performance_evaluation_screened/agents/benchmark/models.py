from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class MatchResult:
    """Data class to hold match results with confidence and details."""

    manual_index: int
    auto_index: int
    confidence: float
    match_type: str
    details: Dict[str, Any]

__all__ = [
    'MatchResult',
]