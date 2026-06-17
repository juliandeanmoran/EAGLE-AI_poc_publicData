from typing import Tuple, Any, Union, Dict, List
from .models import MatchResult
from .matching import PublicationMatcher, CaseMatcher
from .publication import PublicationParser
from .processing import DataProcessor

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

class ScoreComparator:
    """Handles score comparisons and difference calculations."""

    @staticmethod
    def calculate_differences(
        manual_score: Union[float, None, str], auto_score: Union[float, None]
    ) -> Dict[str, Any]:
        """Calculate comprehensive score differences."""
        # Handle manual_score string/None conversion
        if manual_score is None:
            processed_manual_score = None
        elif isinstance(manual_score, str):
            manual_score_str = manual_score.strip().upper()
            if manual_score_str in ["NOT SCORED", "NOT_SCORED", "N/A", "NA", ""]:
                processed_manual_score = None
            else:
                try:
                    processed_manual_score = float(manual_score)
                except (ValueError, TypeError):
                    processed_manual_score = None
        elif pd.isna(manual_score):
            processed_manual_score = None
        else:
            try:
                processed_manual_score = float(manual_score)
            except (ValueError, TypeError):
                processed_manual_score = None

        # Handle cases where manual score is not available
        if processed_manual_score is None:
            return {
                "difference": np.nan,
                "absolute_difference": np.nan,
                "percentage_difference": np.nan,
                "scores_match": False,
                "match_type": "MANUAL_NOT_SCORED",
            }

        # Handle auto_score conversion
        if auto_score is None:
            auto_score = 0.0
        else:
            try:
                auto_score = float(auto_score)
            except (ValueError, TypeError):
                auto_score = 0.0

        difference = auto_score - processed_manual_score
        absolute_difference = abs(difference)

        # Calculate percentage difference
        if processed_manual_score != 0:
            percentage_difference = (difference / processed_manual_score) * 100
        else:
            percentage_difference = float("inf") if auto_score != 0 else 0

        # Determine match type
        if abs(difference) < 0.001:
            scores_match = True
            match_type = "EXACT_MATCH"
        elif abs(difference) < 0.5:
            scores_match = False
            match_type = "CLOSE_MATCH"
        else:
            scores_match = False
            match_type = "SIGNIFICANT_DIFFERENCE"

        return {
            "difference": difference,
            "absolute_difference": absolute_difference,
            "percentage_difference": percentage_difference,
            "scores_match": scores_match,
            "match_type": match_type,
        }


class ScoreVerifier:
    """Main class for improved score verification."""

    def __init__(
        self,
        require_gene_match: bool = True,
        publication_match_threshold: float = 0.8,
        case_match_threshold: float = 0.4,
    ):
        self.require_gene_match = require_gene_match
        self.pub_matcher = PublicationMatcher(
            min_title_confidence=publication_match_threshold
        )
        self.case_matcher = CaseMatcher(
            require_gene_match=require_gene_match, min_confidence=case_match_threshold
        )
        self.score_comparator = ScoreComparator()

    def verify_scores(
        self, manual_df: pd.DataFrame, auto_extractions: Any
    ) -> pd.DataFrame:
        """Main verification method with improved matching."""
        # Validate input
        required_columns = ["publication", "pmid", "final_score", "gene", "id"]
        missing_columns = [
            col for col in required_columns if col not in manual_df.columns
        ]
        if missing_columns:
            raise ValueError(f"Manual dataset missing columns: {missing_columns}")

        # Standardize auto extractions
        auto_extractions_standardized = DataProcessor.standardize_auto_extractions(
            auto_extractions
        )

        results = []

        for pub_idx, auto_data in enumerate(auto_extractions_standardized):
            # Find publication matches
            pub_matches = self.pub_matcher.find_matches(manual_df, auto_data)

            # Extract cases from auto data
            auto_cases = DataProcessor.extract_cases_from_auto_data(auto_data)

            if not pub_matches:
                # Only add unmatched if gene matching not required
                if not self.require_gene_match:
                    self._add_unmatched_publication_results(
                        results, pub_idx, auto_data, auto_cases
                    )
                continue

            # Use best publication match
            best_pub_match = pub_matches[0]
            self._process_publication_match(
                results, manual_df, best_pub_match, auto_data, auto_cases, pub_idx
            )

        return pd.DataFrame(results)

    def _add_unmatched_publication_results(
        self, results: List, pub_idx: int, auto_data: Dict, auto_cases: List[Dict]
    ):
        """Add results for unmatched publications."""
        for case_idx, case_data in enumerate(auto_cases):
            results.append(
                {
                    "publication_index": pub_idx,
                    "manual_dataset_publication": "NO MATCH FOUND",
                    "auto_extraction_dataset": f"{auto_data.get('author', 'Unknown')} - {auto_data.get('title', 'Unknown')}",
                    "case_index": case_idx,
                    "manual_case_id": "NO MATCH",
                    "automatic_case_id": case_data.get("case_id", "Unknown"),
                    "manual_gene": "",
                    "automatic_gene": case_data.get("gene_symbol", ""),
                    "manual_score": np.nan,
                    "automatic_score": case_data.get("final_score", 0),
                    **self.score_comparator.calculate_differences(
                        None, case_data.get("final_score", 0)
                    ),
                    "case_confidence": 0.0,
                    "publication_match_type": "NONE",
                }
            )

    def _process_publication_match(
        self,
        results: List,
        manual_df: pd.DataFrame,
        pub_match: MatchResult,
        auto_data: Dict,
        auto_cases: List[Dict],
        pub_idx: int,
    ):
        """Process a matched publication."""
        # Get all manual rows for this publication
        manual_row = manual_df.iloc[pub_match.manual_index]
        manual_pmid = PublicationParser.extract_pmid(manual_row["pmid"])

        if manual_pmid:
            manual_pub_rows = manual_df[
                manual_df["pmid"].apply(
                    lambda x: PublicationParser.extract_pmid(str(x))
                )
                == manual_pmid
            ]
        else:
            manual_pub_rows = manual_df[
                manual_df["publication"] == manual_row["publication"]
            ]

        # Find case matches
        case_matches = self.case_matcher.find_matches(
            manual_pub_rows.to_dict("records"), auto_cases
        )

        if case_matches:
            self._add_matched_case_results(
                results,
                manual_pub_rows,
                auto_cases,
                case_matches,
                auto_data,
                pub_match,
                pub_idx,
            )
        elif not self.require_gene_match:
            self._add_publication_only_matches(
                results, manual_pub_rows, auto_cases, auto_data, pub_match, pub_idx
            )

    def _add_matched_case_results(
        self,
        results: List,
        manual_pub_rows: pd.DataFrame,
        auto_cases: List[Dict],
        case_matches: List[MatchResult],
        auto_data: Dict,
        pub_match: MatchResult,
        pub_idx: int,
    ):
        """Add results for matched cases."""
        for case_match in case_matches:
            manual_row = manual_pub_rows.iloc[case_match.manual_index]
            auto_case = auto_cases[case_match.auto_index]

            manual_score = manual_row.get("final_score")
            auto_score = auto_case.get("final_score", 0)

            # Skip "NOT SCORED" cases entirely
            if pd.isna(manual_score) or str(manual_score).strip().upper() in [
                "NOT SCORED",
                "NOT_SCORED",
                "N/A",
                "NA",
            ]:
                continue

            score_diff = self.score_comparator.calculate_differences(
                manual_score, auto_score
            )

            results.append(
                {
                    "publication_index": pub_idx,
                    "manual_dataset_publication": manual_row.get("publication", ""),
                    "auto_extraction_dataset": f"{auto_data.get('author', 'Unknown')} - {auto_data.get('title', 'Unknown')}",
                    "case_index": case_match.auto_index,
                    "manual_case_id": manual_row.get("id", "Unknown"),
                    "automatic_case_id": auto_case.get("case_id", "Unknown"),
                    "manual_gene": manual_row.get("gene", ""),
                    "automatic_gene": auto_case.get("gene_symbol", ""),
                    "manual_score": manual_score,
                    "automatic_score": auto_score,
                    **score_diff,
                    "case_confidence": case_match.confidence,
                    "publication_match_type": pub_match.match_type,
                }
            )

    def _add_publication_only_matches(
        self,
        results: List,
        manual_pub_rows: pd.DataFrame,
        auto_cases: List[Dict],
        auto_data: Dict,
        pub_match: MatchResult,
        pub_idx: int,
    ):
        """Add results for publication-only matches (legacy behavior)."""
        for case_idx, case_data in enumerate(auto_cases):
            if not manual_pub_rows.empty:
                manual_row = manual_pub_rows.iloc[0]  # Use first row as representative
                manual_score = manual_row.get("final_score")

                if pd.isna(manual_score) or str(manual_score).strip().upper() in [
                    "NOT SCORED",
                    "NOT_SCORED",
                    "N/A",
                    "NA",
                ]:
                    continue  # Skip NOT SCORED cases entirely

                auto_score = case_data.get("final_score", 0)
                score_diff = self.score_comparator.calculate_differences(
                    manual_score, auto_score
                )

                results.append(
                    {
                        "publication_index": pub_idx,
                        "manual_dataset_publication": manual_row.get("publication", ""),
                        "auto_extraction_dataset": f"{auto_data.get('author', 'Unknown')} - {auto_data.get('title', 'Unknown')}",
                        "case_index": case_idx,
                        "manual_case_id": "PUBLICATION_MATCH_ONLY",
                        "automatic_case_id": case_data.get("case_id", "Unknown"),
                        "manual_gene": manual_row.get("gene", ""),
                        "automatic_gene": case_data.get("gene_symbol", ""),
                        "manual_score": manual_score,
                        "automatic_score": auto_score,
                        **score_diff,
                        "case_confidence": 0.0,
                        "publication_match_type": pub_match.match_type,
                    }
                )
