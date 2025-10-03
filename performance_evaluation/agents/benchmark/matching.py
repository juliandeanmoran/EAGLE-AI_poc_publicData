import pandas as pd
from typing import Dict, List, Optional
import warnings

from .models import MatchResult
from .publication import PublicationParser
from .similarity import SimilarityCalculator
from .text import TextNormalizer

warnings.filterwarnings("ignore")


class PublicationMatcher:
    """Handles matching of publications between manual and automatic datasets."""

    def __init__(
        self, min_pmid_confidence: float = 1.0, min_title_confidence: float = 0.8
    ):
        self.min_pmid_confidence = min_pmid_confidence
        self.min_title_confidence = min_title_confidence

    def find_matches(
        self, manual_df: pd.DataFrame, auto_data: Dict
    ) -> List[MatchResult]:
        """Find matching publications with improved confidence scoring."""
        matches = []

        # Extract auto publication data
        auto_title = auto_data.get("title", "")
        auto_author = auto_data.get("author", "")
        auto_pmid = PublicationParser.extract_pmid(auto_data.get("pmid", ""))

        auto_title_words = TextNormalizer.get_title_words(auto_title)
        auto_author_norm = TextNormalizer.normalize_author(auto_author)

        for idx, row in manual_df.iterrows():
            confidence = 0.0
            details = {}
            match_type = "NO_MATCH"

            # Extract manual publication data
            manual_pub = row.get("publication", "")
            manual_pmid = PublicationParser.extract_pmid(row.get("pmid", ""))
            manual_author_raw, manual_title_raw = (
                PublicationParser.parse_publication_field(manual_pub)
            )

            manual_title_words = TextNormalizer.get_title_words(manual_title_raw)
            manual_author_norm = TextNormalizer.normalize_author(manual_author_raw)

            # PMID matching (highest priority)
            if auto_pmid and manual_pmid and auto_pmid == manual_pmid:
                confidence = 1.0
                match_type = "PMID_EXACT"
                details.update({"pmid_match": True, "matched_pmid": auto_pmid})
            else:
                # Title similarity
                title_similarity = SimilarityCalculator.jaccard_similarity(
                    auto_title_words, manual_title_words
                )

                # Author similarity
                author_similarity = SimilarityCalculator.author_similarity(
                    auto_author, manual_author_raw
                )

                # Combined scoring with stricter thresholds
                if title_similarity >= 0.7 and author_similarity >= 0.8:
                    confidence = (
                        0.6 + (title_similarity * 0.3) + (author_similarity * 0.1)
                    )
                    match_type = "TITLE_AUTHOR_HIGH"
                elif title_similarity >= 0.8:  # Very high title similarity alone
                    confidence = 0.5 + (title_similarity * 0.3)
                    match_type = "TITLE_HIGH"
                elif title_similarity >= 0.6 and author_similarity >= 0.7:
                    confidence = (
                        0.4 + (title_similarity * 0.2) + (author_similarity * 0.1)
                    )
                    match_type = "TITLE_AUTHOR_MEDIUM"

                details.update(
                    {
                        "title_similarity": title_similarity,
                        "author_similarity": author_similarity,
                        "auto_title_words": len(auto_title_words),
                        "manual_title_words": len(manual_title_words),
                        "common_words": len(
                            auto_title_words.intersection(manual_title_words)
                        ),
                    }
                )

            # Only add matches above minimum confidence
            min_confidence = (
                self.min_pmid_confidence
                if match_type == "PMID_EXACT"
                else self.min_title_confidence
            )
            if confidence >= min_confidence:
                matches.append(
                    MatchResult(
                        manual_index=idx,
                        auto_index=0,  # Auto data doesn't have an index in this context
                        confidence=confidence,
                        match_type=match_type,
                        details=details,
                    )
                )

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches


class CaseMatcher:
    """Handles matching of individual cases within publications."""

    def __init__(self, require_gene_match: bool = True, min_confidence: float = 0.4):
        self.require_gene_match = require_gene_match
        self.min_confidence = min_confidence

    def find_matches(
        self, manual_rows: List[Dict], auto_cases: List[Dict]
    ) -> List[MatchResult]:
        """Find matching cases with improved gene-based matching."""
        matches = []
        used_auto_indices = set()

        for manual_idx, manual_row in enumerate(manual_rows):
            manual_data = self._extract_manual_case_data(manual_row)

            # Skip if manual case is NOT SCORED
            if manual_data["final_score"] is None:
                continue

            # Skip if gene matching required but manual gene is empty
            if self.require_gene_match and not manual_data["gene"]:
                continue

            best_match = None

            for auto_idx, auto_case in enumerate(auto_cases):
                if auto_idx in used_auto_indices:
                    continue

                auto_data = self._extract_auto_case_data(auto_case)

                # Calculate match confidence
                match_result = self._calculate_case_match(
                    manual_data, auto_data, manual_idx, auto_idx
                )

                if match_result and match_result.confidence >= self.min_confidence:
                    if (
                        not best_match
                        or match_result.confidence > best_match.confidence
                    ):
                        best_match = match_result

            if best_match:
                matches.append(best_match)
                used_auto_indices.add(best_match.auto_index)

        return matches

    def _extract_manual_case_data(self, manual_row: Dict) -> Dict:
        """Extract and normalize manual case data."""
        # Handle final_score conversion
        final_score = manual_row.get("final_score")
        if isinstance(final_score, str):
            final_score_str = final_score.strip().upper()
            if final_score_str in ["NOT SCORED", "NOT_SCORED", "N/A", "NA", ""]:
                final_score = None
            else:
                try:
                    final_score = float(final_score)
                except (ValueError, TypeError):
                    final_score = None
        elif pd.isna(final_score):
            final_score = None

        return {
            "gene": str(manual_row.get("gene", "")).strip().upper(),
            "case_id": TextNormalizer.normalize_case_id(str(manual_row.get("id", ""))),
            "sex": str(manual_row.get("sex", "")).lower().strip(),
            "inheritance": str(manual_row.get("inheritance", "")).lower().strip(),
            "final_score": final_score,
            "phenotype": str(manual_row.get("phenotype", "")).strip(),
        }

    def _extract_auto_case_data(self, auto_case: Dict) -> Dict:
        """Extract and normalize automatic case data."""
        variant_info = auto_case.get("variant_info", {})

        # Handle final_score conversion
        final_score = auto_case.get("final_score", 0)
        try:
            final_score = float(final_score) if final_score is not None else 0.0
        except (ValueError, TypeError):
            final_score = 0.0

        return {
            "gene": str(auto_case.get("gene_symbol", "")).strip().upper(),
            "case_id": TextNormalizer.normalize_case_id(
                str(auto_case.get("case_id", ""))
            ),
            "inheritance": str(variant_info.get("inheritance_pattern", ""))
            .lower()
            .strip(),
            "final_score": final_score,
        }

    def _calculate_case_match(
        self, manual_data: Dict, auto_data: Dict, manual_idx: int, auto_idx: int
    ) -> Optional[MatchResult]:
        """Calculate case match confidence with detailed scoring."""
        confidence = 0.0
        details = {}
        match_reasons = []

        # Gene matching
        if self.require_gene_match:
            if not auto_data["gene"]:
                return None  # Skip if auto gene is empty

            if manual_data["gene"] == auto_data["gene"]:
                confidence += 0.6
                match_reasons.append("gene_exact")
            elif (
                SimilarityCalculator.substring_similarity(
                    manual_data["gene"], auto_data["gene"]
                )
                >= 0.9
            ):
                confidence += 0.4
                match_reasons.append("gene_fuzzy")
            else:
                return None  # No gene match when required
        else:
            # Legacy: gene matching adds confidence but isn't required
            if manual_data["gene"] and auto_data["gene"]:
                if manual_data["gene"] == auto_data["gene"]:
                    confidence += 0.5
                    match_reasons.append("gene_exact")
                elif (
                    SimilarityCalculator.substring_similarity(
                        manual_data["gene"], auto_data["gene"]
                    )
                    >= 0.9
                ):
                    confidence += 0.3
                    match_reasons.append("gene_fuzzy")

        # Case ID matching with improved logic
        manual_case_id = manual_data["case_id"]
        auto_case_id = auto_data["case_id"]

        if manual_case_id["original"] and auto_case_id["original"]:
            if manual_case_id["normalized"] == auto_case_id["normalized"]:
                confidence += 0.25
                match_reasons.append("case_id_exact")
            elif (
                manual_case_id["numeric_parts"] == auto_case_id["numeric_parts"]
                and manual_case_id["numeric_parts"]
            ):
                confidence += 0.2
                match_reasons.append("case_id_numeric")
            elif (
                manual_case_id["original"].lower() in auto_case_id["original"].lower()
                or auto_case_id["original"].lower()
                in manual_case_id["original"].lower()
            ):
                confidence += 0.15
                match_reasons.append("case_id_contains")
            elif (
                manual_case_id["first_numeric"]
                and auto_case_id["first_numeric"]
                and manual_case_id["first_numeric"] == auto_case_id["first_numeric"]
            ):
                confidence += 0.1
                match_reasons.append("case_id_first_numeric")

        # Inheritance pattern matching
        if manual_data["inheritance"] and auto_data["inheritance"]:
            if (
                manual_data["inheritance"] in auto_data["inheritance"]
                or auto_data["inheritance"] in manual_data["inheritance"]
                or (
                    manual_data["inheritance"] == "de novo"
                    and "novo" in auto_data["inheritance"]
                )
            ):
                confidence += 0.1
                match_reasons.append("inheritance_match")

        details.update(
            {
                "reasons": match_reasons,
                "manual_gene": manual_data["gene"],
                "auto_gene": auto_data["gene"],
                "manual_case_id": manual_case_id["original"],
                "auto_case_id": auto_case_id["original"],
            }
        )

        if confidence >= self.min_confidence:
            return MatchResult(
                manual_index=manual_idx,
                auto_index=auto_idx,
                confidence=confidence,
                match_type="CASE_MATCH",
                details=details,
            )

        return None


__all__ = [
    "PublicationMatcher",
    "CaseMatcher",
]
