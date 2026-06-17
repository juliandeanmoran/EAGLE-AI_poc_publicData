import pandas as pd
from typing import Tuple, Any
import re
import warnings

warnings.filterwarnings("ignore")


class PublicationParser:
    """Handles parsing of publication fields."""

    @staticmethod
    def parse_publication_field(publication_str: str) -> Tuple[str, str]:
        """Parse publication field to extract author and title."""
        if pd.isna(publication_str) or not publication_str:
            return "", ""

        publication_str = str(publication_str).strip()

        # Pattern 1: Author et al (YEAR): Title
        pattern1 = r"^(.*?)\s*\(\d{4}\)\s*[:\.]?\s*(.*)"
        match1 = re.match(pattern1, publication_str)

        if match1:
            author_part = match1.group(1).strip().rstrip(".")
            title_part = match1.group(2).strip()
            return author_part, title_part

        # Pattern 2: Colon separator
        if ":" in publication_str:
            parts = publication_str.split(":", 1)
            return parts[0].strip(), parts[1].strip()

        # Pattern 3: Et al pattern
        pattern3 = r"^(.*?(?:et al\.?|and colleagues)(?:\s*\(\d{4}\))?)\s*(.*)"
        match3 = re.match(pattern3, publication_str, re.IGNORECASE)

        if match3:
            author_part = match3.group(1).strip()
            title_part = re.sub(r"^[:\.\-\s]+", "", match3.group(2).strip())
            return author_part, title_part

        return "", publication_str

    @staticmethod
    def extract_pmid(pmid_field: Any) -> str:
        """Extract PMID from field, handling various formats."""
        if pd.isna(pmid_field) or not pmid_field:
            return ""

        pmid_str = str(pmid_field).strip()
        if pmid_str.endswith(".0"):
            pmid_str = pmid_str[:-2]

        return re.sub(r"[^\d]", "", pmid_str)
