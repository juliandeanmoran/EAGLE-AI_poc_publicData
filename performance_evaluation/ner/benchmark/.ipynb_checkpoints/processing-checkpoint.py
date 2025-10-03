import pandas as pd
from typing import Any, Dict, List
import re
import warnings

warnings.filterwarnings("ignore")


class DataProcessor:
    """Handles data extraction and processing."""

    @staticmethod
    def extract_enum_value(value: Any) -> Any:
        """Extract actual value from enum representations."""
        if isinstance(value, str):
            enum_pattern = r'<[^>]+:\s*[\'"]([^\'"]+)[\'"]>'
            match = re.match(enum_pattern, str(value))
            if match:
                return match.group(1)
        elif hasattr(value, "value"):
            return value.value
        return value

    @staticmethod
    def standardize_auto_extractions(auto_extractions: Any) -> List[Dict]:
        """Standardize auto_extractions to consistent format."""
        if isinstance(auto_extractions, list):
            return (
                auto_extractions
                if auto_extractions and isinstance(auto_extractions[0], dict)
                else []
            )
        elif isinstance(auto_extractions, dict):
            first_key = (
                next(iter(auto_extractions.keys())) if auto_extractions else None
            )
            if first_key and isinstance(auto_extractions[first_key], dict):
                # Dictionary of papers
                papers_list = []
                for key, paper_data in auto_extractions.items():
                    if isinstance(paper_data, dict):
                        if "paper_id" not in paper_data:
                            paper_data["paper_id"] = key
                        papers_list.append(paper_data)
                return papers_list
            else:
                return [auto_extractions]
        return []

    @staticmethod
    def extract_cases_from_auto_data(auto_data: Dict) -> List[Dict]:
        """Extract cases with improved structure handling."""
        cases = []

        # Try different possible locations for cases
        possible_keys = ["cases", "individual_case_scores", "case_scores"]

        for key in possible_keys:
            if key in auto_data:
                cases_data = auto_data[key]
                if (
                    isinstance(cases_data, dict)
                    and "individual_case_scores" in cases_data
                ):
                    cases = cases_data["individual_case_scores"]
                    break
                elif isinstance(cases_data, list):
                    cases = cases_data
                    break

        # Clean enum values
        cleaned_cases = []
        for case in cases if isinstance(cases, list) else []:
            if isinstance(case, dict):
                cleaned_case = {}
                for key, value in case.items():
                    if key == "variant_info" and isinstance(value, dict):
                        cleaned_variant_info = {
                            vkey: DataProcessor.extract_enum_value(vvalue)
                            for vkey, vvalue in value.items()
                        }
                        cleaned_case[key] = cleaned_variant_info
                    else:
                        cleaned_case[key] = DataProcessor.extract_enum_value(value)
                cleaned_cases.append(cleaned_case)

        return cleaned_cases
