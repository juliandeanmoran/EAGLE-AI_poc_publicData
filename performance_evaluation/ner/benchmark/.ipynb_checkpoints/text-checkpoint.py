import pandas as pd
import numpy as np
from typing import Dict, Any
import re
import warnings

warnings.filterwarnings("ignore")

class TextNormalizer:
    """Handles all text normalization tasks."""

    STOP_WORDS = {
        "the",
        "a",
        "an",
        "of",
        "in",
        "to",
        "for",
        "with",
        "and",
        "or",
        "on",
        "at",
        "by",
    }

    @staticmethod
    def normalize_case_id(case_id: str) -> Dict[str, Any]:
        """Normalize case ID for better matching."""
        if pd.isna(case_id) or not case_id:
            return {
                "original": "",
                "normalized": "",
                "numeric_parts": [],
                "first_numeric": None,
            }

        case_id = str(case_id).strip()
        numeric_parts = re.findall(r"\d+", case_id)
        normalized = case_id.lower().replace(" ", "").replace("-", "").replace("_", "")

        return {
            "original": case_id,
            "normalized": normalized,
            "numeric_parts": numeric_parts,
            "first_numeric": numeric_parts[0] if numeric_parts else None,
        }

    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize title for comparison."""
        if pd.isna(title) or not title:
            return ""

        title = str(title).strip().lower()
        title = re.sub(r"\s+", " ", title)
        title = re.sub(r"[^\w\s]", "", title)
        return title

    @staticmethod
    def normalize_author(author_str: str) -> str:
        """Normalize author string for comparison."""
        if pd.isna(author_str) or not author_str:
            return ""

        author_str = str(author_str).strip().lower()
        author_str = re.sub(r"\s+", " ", author_str)

        # Extract main author (before 'et al' or comma)
        if "et al" in author_str:
            main_author = author_str.split("et al")[0].strip()
        elif "," in author_str:
            main_author = author_str.split(",")[0].strip()
        else:
            main_author = author_str

        # Remove parentheses content
        main_author = re.sub(r"\(.*?\)", "", main_author).strip()
        return main_author

    @staticmethod
    def get_title_words(title: str) -> set:
        """Extract meaningful words from title."""
        normalized = TextNormalizer.normalize_title(title)
        words = set(normalized.split()) - TextNormalizer.STOP_WORDS
        return {word for word in words if len(word) > 2}


__all__ = [
    'TextNormalizer',
]