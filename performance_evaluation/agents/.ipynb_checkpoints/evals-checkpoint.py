####################################
# Matching Cases-Scores Algorithm #
###################################

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Union, Optional
import re
from dataclasses import dataclass
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')
import random


@dataclass
class MatchResult:
    """Data class to hold match results with confidence and details."""
    manual_index: int
    auto_index: int
    confidence: float
    match_type: str
    details: Dict[str, Any]


class TextNormalizer:
    """Handles all text normalization tasks."""
    
    STOP_WORDS = {'the', 'a', 'an', 'of', 'in', 'to', 'for', 'with', 'and', 'or', 'on', 'at', 'by'}
    
    @staticmethod
    def normalize_case_id(case_id: str) -> Dict[str, Any]:
        """Normalize case ID for better matching."""
        if pd.isna(case_id) or not case_id:
            return {
                'original': '',
                'normalized': '',
                'numeric_parts': [],
                'first_numeric': None
            }
        
        case_id = str(case_id).strip()
        numeric_parts = re.findall(r'\d+', case_id)
        normalized = case_id.lower().replace(' ', '').replace('-', '').replace('_', '')
        
        return {
            'original': case_id,
            'normalized': normalized,
            'numeric_parts': numeric_parts,
            'first_numeric': numeric_parts[0] if numeric_parts else None
        }
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """Normalize title for comparison."""
        if pd.isna(title) or not title:
            return ""
        
        title = str(title).strip().lower()
        title = re.sub(r'\s+', ' ', title)
        title = re.sub(r'[^\w\s]', '', title)
        return title
    
    @staticmethod
    def normalize_author(author_str: str) -> str:
        """Normalize author string for comparison."""
        if pd.isna(author_str) or not author_str:
            return ""
        
        author_str = str(author_str).strip().lower()
        author_str = re.sub(r'\s+', ' ', author_str)
        
        # Extract main author (before 'et al' or comma)
        if 'et al' in author_str:
            main_author = author_str.split('et al')[0].strip()
        elif ',' in author_str:
            main_author = author_str.split(',')[0].strip()
        else:
            main_author = author_str
            
        # Remove parentheses content
        main_author = re.sub(r'\(.*?\)', '', main_author).strip()
        return main_author
    
    @staticmethod
    def get_title_words(title: str) -> set:
        """Extract meaningful words from title."""
        normalized = TextNormalizer.normalize_title(title)
        words = set(normalized.split()) - TextNormalizer.STOP_WORDS
        return {word for word in words if len(word) > 2}


class PublicationParser:
    """Handles parsing of publication fields."""
    
    @staticmethod
    def parse_publication_field(publication_str: str) -> Tuple[str, str]:
        """Parse publication field to extract author and title."""
        if pd.isna(publication_str) or not publication_str:
            return "", ""
        
        publication_str = str(publication_str).strip()
        
        # Pattern 1: Author et al (YEAR): Title
        pattern1 = r'^(.*?)\s*\(\d{4}\)\s*[:\.]?\s*(.*)'
        match1 = re.match(pattern1, publication_str)
        
        if match1:
            author_part = match1.group(1).strip().rstrip('.')
            title_part = match1.group(2).strip()
            return author_part, title_part
        
        # Pattern 2: Colon separator
        if ':' in publication_str:
            parts = publication_str.split(':', 1)
            return parts[0].strip(), parts[1].strip()
        
        # Pattern 3: Et al pattern
        pattern3 = r'^(.*?(?:et al\.?|and colleagues)(?:\s*\(\d{4}\))?)\s*(.*)'
        match3 = re.match(pattern3, publication_str, re.IGNORECASE)
        
        if match3:
            author_part = match3.group(1).strip()
            title_part = re.sub(r'^[:\.\-\s]+', '', match3.group(2).strip())
            return author_part, title_part
        
        return "", publication_str
    
    @staticmethod
    def extract_pmid(pmid_field: Any) -> str:
        """Extract PMID from field, handling various formats."""
        if pd.isna(pmid_field) or not pmid_field:
            return ""
        
        pmid_str = str(pmid_field).strip()
        if pmid_str.endswith('.0'):
            pmid_str = pmid_str[:-2]
        
        return re.sub(r'[^\d]', '', pmid_str)


class SimilarityCalculator:
    """Handles similarity calculations between strings and publications."""
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return len(intersection) / len(union)
    
    @staticmethod
    def substring_similarity(str1: str, str2: str) -> float:
        """Calculate similarity based on substring matching."""
        if not str1 or not str2:
            return 0.0
        
        str1, str2 = str1.lower().strip(), str2.lower().strip()
        
        if str1 == str2:
            return 1.0
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Character-level similarity
        longer = max(len(str1), len(str2))
        matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        return matches / longer if longer > 0 else 0.0
    
    @staticmethod
    def author_similarity(author1: str, author2: str) -> float:
        """Calculate author similarity with focus on last names."""
        norm1 = TextNormalizer.normalize_author(author1)
        norm2 = TextNormalizer.normalize_author(author2)
        
        if not norm1 or not norm2:
            return 0.0
        
        # Extract potential surnames (words > 2 chars)
        surnames1 = [word for word in norm1.split() if len(word) > 2]
        surnames2 = [word for word in norm2.split() if len(word) > 2]
        
        max_similarity = 0.0
        for s1 in surnames1:
            for s2 in surnames2:
                sim = SimilarityCalculator.substring_similarity(s1, s2)
                max_similarity = max(max_similarity, sim)
        
        return max_similarity


class PublicationMatcher:
    """Handles matching of publications between manual and automatic datasets."""
    
    def __init__(self, min_pmid_confidence: float = 1.0, min_title_confidence: float = 0.8):
        self.min_pmid_confidence = min_pmid_confidence
        self.min_title_confidence = min_title_confidence
    
    def find_matches(self, manual_df: pd.DataFrame, auto_data: Dict) -> List[MatchResult]:
        """Find matching publications with improved confidence scoring."""
        matches = []
        
        # Extract auto publication data
        auto_title = auto_data.get('title', '')
        auto_author = auto_data.get('author', '')
        auto_pmid = PublicationParser.extract_pmid(auto_data.get('pmid', ''))
        
        auto_title_words = TextNormalizer.get_title_words(auto_title)
        auto_author_norm = TextNormalizer.normalize_author(auto_author)
        
        for idx, row in manual_df.iterrows():
            confidence = 0.0
            details = {}
            match_type = "NO_MATCH"
            
            # Extract manual publication data
            manual_pub = row.get('publication', '')
            manual_pmid = PublicationParser.extract_pmid(row.get('pmid', ''))
            manual_author_raw, manual_title_raw = PublicationParser.parse_publication_field(manual_pub)
            
            manual_title_words = TextNormalizer.get_title_words(manual_title_raw)
            manual_author_norm = TextNormalizer.normalize_author(manual_author_raw)
            
            # PMID matching (highest priority)
            if auto_pmid and manual_pmid and auto_pmid == manual_pmid:
                confidence = 1.0
                match_type = "PMID_EXACT"
                details.update({
                    'pmid_match': True,
                    'matched_pmid': auto_pmid
                })
            else:
                # Title similarity
                title_similarity = SimilarityCalculator.jaccard_similarity(auto_title_words, manual_title_words)
                
                # Author similarity
                author_similarity = SimilarityCalculator.author_similarity(auto_author, manual_author_raw)
                
                # Combined scoring with stricter thresholds
                if title_similarity >= 0.7 and author_similarity >= 0.8:
                    confidence = 0.6 + (title_similarity * 0.3) + (author_similarity * 0.1)
                    match_type = "TITLE_AUTHOR_HIGH"
                elif title_similarity >= 0.8:  # Very high title similarity alone
                    confidence = 0.5 + (title_similarity * 0.3)
                    match_type = "TITLE_HIGH"
                elif title_similarity >= 0.6 and author_similarity >= 0.7:
                    confidence = 0.4 + (title_similarity * 0.2) + (author_similarity * 0.1)
                    match_type = "TITLE_AUTHOR_MEDIUM"
                
                details.update({
                    'title_similarity': title_similarity,
                    'author_similarity': author_similarity,
                    'auto_title_words': len(auto_title_words),
                    'manual_title_words': len(manual_title_words),
                    'common_words': len(auto_title_words.intersection(manual_title_words))
                })
            
            # Only add matches above minimum confidence
            min_confidence = self.min_pmid_confidence if match_type == "PMID_EXACT" else self.min_title_confidence
            if confidence >= min_confidence:
                matches.append(MatchResult(
                    manual_index=idx,
                    auto_index=0,  # Auto data doesn't have an index in this context
                    confidence=confidence,
                    match_type=match_type,
                    details=details
                ))
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches


class CaseMatcher:
    """Handles matching of individual cases within publications."""
    
    def __init__(self, require_gene_match: bool = True, min_confidence: float = 0.4):
        self.require_gene_match = require_gene_match
        self.min_confidence = min_confidence
    
    def find_matches(self, manual_rows: List[Dict], auto_cases: List[Dict]) -> List[MatchResult]:
        """Find matching cases with improved gene-based matching."""
        matches = []
        used_auto_indices = set()
        
        for manual_idx, manual_row in enumerate(manual_rows):
            manual_data = self._extract_manual_case_data(manual_row)
            
            # Skip if manual case is NOT SCORED
            if manual_data['final_score'] is None:
                continue
            
            # Skip if gene matching required but manual gene is empty
            if self.require_gene_match and not manual_data['gene']:
                continue
            
            best_match = None
            
            for auto_idx, auto_case in enumerate(auto_cases):
                if auto_idx in used_auto_indices:
                    continue
                
                auto_data = self._extract_auto_case_data(auto_case)
                
                # Calculate match confidence
                match_result = self._calculate_case_match(manual_data, auto_data, manual_idx, auto_idx)
                
                if match_result and match_result.confidence >= self.min_confidence:
                    if not best_match or match_result.confidence > best_match.confidence:
                        best_match = match_result
            
            if best_match:
                matches.append(best_match)
                used_auto_indices.add(best_match.auto_index)
        
        return matches
    
    def _extract_manual_case_data(self, manual_row: Dict) -> Dict:
        """Extract and normalize manual case data."""
        # Handle final_score conversion
        final_score = manual_row.get('final_score')
        if isinstance(final_score, str):
            final_score_str = final_score.strip().upper()
            if final_score_str in ['NOT SCORED', 'NOT_SCORED', 'N/A', 'NA', '']:
                final_score = None
            else:
                try:
                    final_score = float(final_score)
                except (ValueError, TypeError):
                    final_score = None
        elif pd.isna(final_score):
            final_score = None
        
        return {
            'gene': str(manual_row.get('gene', '')).strip().upper(),
            'case_id': TextNormalizer.normalize_case_id(str(manual_row.get('id', ''))),
            'sex': str(manual_row.get('sex', '')).lower().strip(),
            'inheritance': str(manual_row.get('inheritance', '')).lower().strip(),
            'final_score': final_score,
            'phenotype': str(manual_row.get('phenotype', '')).strip()
        }
    
    def _extract_auto_case_data(self, auto_case: Dict) -> Dict:
        """Extract and normalize automatic case data."""
        variant_info = auto_case.get('variant_info', {})
        
        # Handle final_score conversion
        final_score = auto_case.get('final_score', 0)
        try:
            final_score = float(final_score) if final_score is not None else 0.0
        except (ValueError, TypeError):
            final_score = 0.0
            
        return {
            'gene': str(auto_case.get('gene_symbol', '')).strip().upper(),
            'case_id': TextNormalizer.normalize_case_id(str(auto_case.get('case_id', ''))),
            'inheritance': str(variant_info.get('inheritance_pattern', '')).lower().strip(),
            'final_score': final_score
        }
    
    def _calculate_case_match(self, manual_data: Dict, auto_data: Dict, 
                            manual_idx: int, auto_idx: int) -> Optional[MatchResult]:
        """Calculate case match confidence with detailed scoring."""
        confidence = 0.0
        details = {}
        match_reasons = []
        
        # Gene matching
        if self.require_gene_match:
            if not auto_data['gene']:
                return None  # Skip if auto gene is empty
            
            if manual_data['gene'] == auto_data['gene']:
                confidence += 0.6
                match_reasons.append("gene_exact")
            elif SimilarityCalculator.substring_similarity(manual_data['gene'], auto_data['gene']) >= 0.9:
                confidence += 0.4
                match_reasons.append("gene_fuzzy")
            else:
                return None  # No gene match when required
        else:
            # Legacy: gene matching adds confidence but isn't required
            if manual_data['gene'] and auto_data['gene']:
                if manual_data['gene'] == auto_data['gene']:
                    confidence += 0.5
                    match_reasons.append("gene_exact")
                elif SimilarityCalculator.substring_similarity(manual_data['gene'], auto_data['gene']) >= 0.9:
                    confidence += 0.3
                    match_reasons.append("gene_fuzzy")
        
        # Case ID matching with improved logic
        manual_case_id = manual_data['case_id']
        auto_case_id = auto_data['case_id']
        
        if manual_case_id['original'] and auto_case_id['original']:
            if manual_case_id['normalized'] == auto_case_id['normalized']:
                confidence += 0.25
                match_reasons.append("case_id_exact")
            elif manual_case_id['numeric_parts'] == auto_case_id['numeric_parts'] and manual_case_id['numeric_parts']:
                confidence += 0.2
                match_reasons.append("case_id_numeric")
            elif (manual_case_id['original'].lower() in auto_case_id['original'].lower() or 
                  auto_case_id['original'].lower() in manual_case_id['original'].lower()):
                confidence += 0.15
                match_reasons.append("case_id_contains")
            elif (manual_case_id['first_numeric'] and auto_case_id['first_numeric'] and 
                  manual_case_id['first_numeric'] == auto_case_id['first_numeric']):
                confidence += 0.1
                match_reasons.append("case_id_first_numeric")
        
        # Inheritance pattern matching
        if manual_data['inheritance'] and auto_data['inheritance']:
            if (manual_data['inheritance'] in auto_data['inheritance'] or 
                auto_data['inheritance'] in manual_data['inheritance'] or
                (manual_data['inheritance'] == 'de novo' and 'novo' in auto_data['inheritance'])):
                confidence += 0.1
                match_reasons.append("inheritance_match")
        
        details.update({
            'reasons': match_reasons,
            'manual_gene': manual_data['gene'],
            'auto_gene': auto_data['gene'],
            'manual_case_id': manual_case_id['original'],
            'auto_case_id': auto_case_id['original']
        })
        
        if confidence >= self.min_confidence:
            return MatchResult(
                manual_index=manual_idx,
                auto_index=auto_idx,
                confidence=confidence,
                match_type="CASE_MATCH",
                details=details
            )
        
        return None


class ScoreComparator:
    """Handles score comparisons and difference calculations."""
    
    @staticmethod
    def calculate_differences(manual_score: Union[float, None, str], auto_score: Union[float, None]) -> Dict[str, Any]:
        """Calculate comprehensive score differences."""
        # Handle manual_score string/None conversion
        if manual_score is None:
            processed_manual_score = None
        elif isinstance(manual_score, str):
            manual_score_str = manual_score.strip().upper()
            if manual_score_str in ['NOT SCORED', 'NOT_SCORED', 'N/A', 'NA', '']:
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
                'difference': np.nan,
                'absolute_difference': np.nan,
                'percentage_difference': np.nan,
                'scores_match': False,
                'match_type': "MANUAL_NOT_SCORED"
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
            percentage_difference = float('inf') if auto_score != 0 else 0
        
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
            'difference': difference,
            'absolute_difference': absolute_difference,
            'percentage_difference': percentage_difference,
            'scores_match': scores_match,
            'match_type': match_type
        }


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
        elif hasattr(value, 'value'):
            return value.value
        return value
    
    @staticmethod
    def standardize_auto_extractions(auto_extractions: Any) -> List[Dict]:
        """Standardize auto_extractions to consistent format."""
        if isinstance(auto_extractions, list):
            return auto_extractions if auto_extractions and isinstance(auto_extractions[0], dict) else []
        elif isinstance(auto_extractions, dict):
            first_key = next(iter(auto_extractions.keys())) if auto_extractions else None
            if first_key and isinstance(auto_extractions[first_key], dict):
                # Dictionary of papers
                papers_list = []
                for key, paper_data in auto_extractions.items():
                    if isinstance(paper_data, dict):
                        if 'paper_id' not in paper_data:
                            paper_data['paper_id'] = key
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
        possible_keys = ['cases', 'individual_case_scores', 'case_scores']
        
        for key in possible_keys:
            if key in auto_data:
                cases_data = auto_data[key]
                if isinstance(cases_data, dict) and 'individual_case_scores' in cases_data:
                    cases = cases_data['individual_case_scores']
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
                    if key == 'variant_info' and isinstance(value, dict):
                        cleaned_variant_info = {vkey: DataProcessor.extract_enum_value(vvalue) 
                                              for vkey, vvalue in value.items()}
                        cleaned_case[key] = cleaned_variant_info
                    else:
                        cleaned_case[key] = DataProcessor.extract_enum_value(value)
                cleaned_cases.append(cleaned_case)
        
        return cleaned_cases


class ScoreVerifier:
    """Main class for improved score verification."""
    
    def __init__(self, require_gene_match: bool = True, 
                 publication_match_threshold: float = 0.8,
                 case_match_threshold: float = 0.4):
        self.require_gene_match = require_gene_match
        self.pub_matcher = PublicationMatcher(min_title_confidence=publication_match_threshold)
        self.case_matcher = CaseMatcher(require_gene_match=require_gene_match, 
                                      min_confidence=case_match_threshold)
        self.score_comparator = ScoreComparator()
    
    def verify_scores(self, manual_df: pd.DataFrame, auto_extractions: Any) -> pd.DataFrame:
        """Main verification method with improved matching."""
        # Validate input
        required_columns = ['publication', 'pmid', 'final_score', 'gene', 'id']
        missing_columns = [col for col in required_columns if col not in manual_df.columns]
        if missing_columns:
            raise ValueError(f"Manual dataset missing columns: {missing_columns}")
        
        # Standardize auto extractions
        auto_extractions_standardized = DataProcessor.standardize_auto_extractions(auto_extractions)
        
        results = []
        
        for pub_idx, auto_data in enumerate(auto_extractions_standardized):
            # Find publication matches
            pub_matches = self.pub_matcher.find_matches(manual_df, auto_data)
            
            # Extract cases from auto data
            auto_cases = DataProcessor.extract_cases_from_auto_data(auto_data)
            
            if not pub_matches:
                # Only add unmatched if gene matching not required
                if not self.require_gene_match:
                    self._add_unmatched_publication_results(results, pub_idx, auto_data, auto_cases)
                continue
            
            # Use best publication match
            best_pub_match = pub_matches[0]
            self._process_publication_match(results, manual_df, best_pub_match, 
                                          auto_data, auto_cases, pub_idx)
        
        return pd.DataFrame(results)
    
    def _add_unmatched_publication_results(self, results: List, pub_idx: int, 
                                         auto_data: Dict, auto_cases: List[Dict]):
        """Add results for unmatched publications."""
        for case_idx, case_data in enumerate(auto_cases):
            results.append({
                'publication_index': pub_idx,
                'manual_dataset_publication': 'NO MATCH FOUND',
                'auto_extraction_dataset': f"{auto_data.get('author', 'Unknown')} - {auto_data.get('title', 'Unknown')}",
                'case_index': case_idx,
                'manual_case_id': 'NO MATCH',
                'automatic_case_id': case_data.get('case_id', 'Unknown'),
                'manual_gene': '',
                'automatic_gene': case_data.get('gene_symbol', ''),
                'manual_score': np.nan,
                'automatic_score': case_data.get('final_score', 0),
                **self.score_comparator.calculate_differences(None, case_data.get('final_score', 0)),
                'case_confidence': 0.0,
                'publication_match_type': 'NONE'
            })
    
    def _process_publication_match(self, results: List, manual_df: pd.DataFrame,
                                 pub_match: MatchResult, auto_data: Dict, 
                                 auto_cases: List[Dict], pub_idx: int):
        """Process a matched publication."""
        # Get all manual rows for this publication
        manual_row = manual_df.iloc[pub_match.manual_index]
        manual_pmid = PublicationParser.extract_pmid(manual_row['pmid'])
        
        if manual_pmid:
            manual_pub_rows = manual_df[manual_df['pmid'].apply(
                lambda x: PublicationParser.extract_pmid(str(x))) == manual_pmid]
        else:
            manual_pub_rows = manual_df[manual_df['publication'] == manual_row['publication']]
        
        # Find case matches
        case_matches = self.case_matcher.find_matches(
            manual_pub_rows.to_dict('records'), auto_cases)
        
        if case_matches:
            self._add_matched_case_results(results, manual_pub_rows, auto_cases, 
                                         case_matches, auto_data, pub_match, pub_idx)
        elif not self.require_gene_match:
            self._add_publication_only_matches(results, manual_pub_rows, auto_cases,
                                             auto_data, pub_match, pub_idx)
    
    def _add_matched_case_results(self, results: List, manual_pub_rows: pd.DataFrame,
                                auto_cases: List[Dict], case_matches: List[MatchResult],
                                auto_data: Dict, pub_match: MatchResult, pub_idx: int):
        """Add results for matched cases."""
        for case_match in case_matches:
            manual_row = manual_pub_rows.iloc[case_match.manual_index]
            auto_case = auto_cases[case_match.auto_index]
            
            manual_score = manual_row.get('final_score')
            auto_score = auto_case.get('final_score', 0)
            
            # Skip "NOT SCORED" cases entirely
            if pd.isna(manual_score) or str(manual_score).strip().upper() in ['NOT SCORED', 'NOT_SCORED', 'N/A', 'NA']:
                continue
            
            score_diff = self.score_comparator.calculate_differences(manual_score, auto_score)
            
            results.append({
                'publication_index': pub_idx,
                'manual_dataset_publication': manual_row.get('publication', ''),
                'auto_extraction_dataset': f"{auto_data.get('author', 'Unknown')} - {auto_data.get('title', 'Unknown')}",
                'case_index': case_match.auto_index,
                'manual_case_id': manual_row.get('id', 'Unknown'),
                'automatic_case_id': auto_case.get('case_id', 'Unknown'),
                'manual_gene': manual_row.get('gene', ''),
                'automatic_gene': auto_case.get('gene_symbol', ''),
                'manual_score': manual_score,
                'automatic_score': auto_score,
                **score_diff,
                'case_confidence': case_match.confidence,
                'publication_match_type': pub_match.match_type
            })
    
    def _add_publication_only_matches(self, results: List, manual_pub_rows: pd.DataFrame,
                                    auto_cases: List[Dict], auto_data: Dict,
                                    pub_match: MatchResult, pub_idx: int):
        """Add results for publication-only matches (legacy behavior)."""
        for case_idx, case_data in enumerate(auto_cases):
            if not manual_pub_rows.empty:
                manual_row = manual_pub_rows.iloc[0]  # Use first row as representative
                manual_score = manual_row.get('final_score')
                
                if pd.isna(manual_score) or str(manual_score).strip().upper() in ['NOT SCORED', 'NOT_SCORED', 'N/A', 'NA']:
                    continue  # Skip NOT SCORED cases entirely
                
                auto_score = case_data.get('final_score', 0)
                score_diff = self.score_comparator.calculate_differences(manual_score, auto_score)
                
                results.append({
                    'publication_index': pub_idx,
                    'manual_dataset_publication': manual_row.get('publication', ''),
                    'auto_extraction_dataset': f"{auto_data.get('author', 'Unknown')} - {auto_data.get('title', 'Unknown')}",
                    'case_index': case_idx,
                    'manual_case_id': 'PUBLICATION_MATCH_ONLY',
                    'automatic_case_id': case_data.get('case_id', 'Unknown'),
                    'manual_gene': manual_row.get('gene', ''),
                    'automatic_gene': case_data.get('gene_symbol', ''),
                    'manual_score': manual_score,
                    'automatic_score': auto_score,
                    **score_diff,
                    'case_confidence': 0.0,
                    'publication_match_type': pub_match.match_type
                })


class StatisticsCalculator:
    """Handles calculation of comparison statistics."""
    
    @staticmethod
    def calculate_summary_stats(result_df: pd.DataFrame, require_gene_match: bool = True) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        stats = {}
        
        total_cases = len(result_df)
        matched_publications = len(result_df[result_df['manual_dataset_publication'] != 'NO MATCH FOUND'])
        
        # Different matching criteria based on gene matching requirement
        if require_gene_match:
            matched_cases = len(result_df[
                (result_df['manual_case_id'] != 'NO MATCH') & 
                (result_df['manual_case_id'] != 'NO CASE MATCH') &
                (result_df['manual_case_id'] != 'PUBLICATION_MATCH_ONLY')
            ])
        else:
            matched_cases = len(result_df[result_df['manual_case_id'] != 'NO MATCH'])
        
        exact_score_matches = len(result_df[result_df['scores_match'] == True])
        
        # Calculate score differences for valid cases
        # Note: NOT SCORED cases are now excluded entirely from processing
        valid_cases = result_df[
            (result_df['manual_case_id'] != 'NO MATCH') & 
            (result_df['manual_case_id'] != 'NO CASE MATCH') &
            (result_df['manual_case_id'] != 'PUBLICATION_MATCH_ONLY')
        ]
        
        stats.update({
            'total_cases': total_cases,
            'matched_publications': matched_publications,
            'matched_cases': matched_cases,
            'exact_score_matches': exact_score_matches,
            'matching_rate': matched_cases / total_cases if total_cases > 0 else 0,
            'exact_match_rate': exact_score_matches / matched_cases if matched_cases > 0 else 0
        })
        
        if not valid_cases.empty and 'absolute_difference' in valid_cases.columns:
            valid_diffs = valid_cases['absolute_difference'].dropna()
            if not valid_diffs.empty:
                stats.update({
                    'avg_absolute_difference': valid_diffs.mean(),
                    'median_absolute_difference': valid_diffs.median(),
                    'min_absolute_difference': valid_diffs.min(),
                    'max_absolute_difference': valid_diffs.max(),
                    'std_absolute_difference': valid_diffs.std()
                })
        
        return stats


def print_verification_summary(result_df: pd.DataFrame, require_gene_match: bool = True) -> None:
    """Print a comprehensive summary of verification results."""
    print("\n" + "="*80)
    print("ENHANCED SCORE VERIFICATION SUMMARY")
    if require_gene_match:
        print("(Gene matching REQUIRED - only cases with matching genes are included)")
    else:
        print("(Gene matching NOT required - legacy behavior)")
    print("(NOT SCORED cases are excluded entirely from comparison)")
    print("="*80)
    
    stats = StatisticsCalculator.calculate_summary_stats(result_df, require_gene_match)
    
    print(f"Total cases processed: {stats['total_cases']}")
    print(f"Publications with matches: {stats['matched_publications']}")
    print(f"Cases with matches: {stats['matched_cases']}")
    print(f"Exact score matches: {stats['exact_score_matches']}")
    print(f"Overall matching rate: {stats['matching_rate']:.1%}")
    print(f"Exact match rate (of matched): {stats['exact_match_rate']:.1%}")
    
    # Publication match type breakdown
    if 'publication_match_type' in result_df.columns:
        print("\nPublication Match Types:")
        pub_match_types = result_df['publication_match_type'].value_counts()
        for match_type, count in pub_match_types.items():
            print(f"  - {match_type}: {count}")
    
    # Score match type breakdown
    if 'match_type' in result_df.columns:
        valid_score_matches = result_df[
            (result_df['manual_case_id'] != 'NO MATCH') & 
            (result_df['manual_case_id'] != 'NO CASE MATCH') &
            (result_df['manual_case_id'] != 'PUBLICATION_MATCH_ONLY')
        ]
        if not valid_score_matches.empty:
            print("\nScore Match Types:")
            score_match_types = valid_score_matches['match_type'].value_counts()
            for match_type, count in score_match_types.items():
                print(f"  - {match_type}: {count}")
    
    # Score differences
    if 'avg_absolute_difference' in stats:
        print(f"\nScore Differences:")
        print(f"  - Average absolute difference: {stats['avg_absolute_difference']:.3f}")
        print(f"  - Median absolute difference: {stats['median_absolute_difference']:.3f}")
        print(f"  - Standard deviation: {stats['std_absolute_difference']:.3f}")
        print(f"  - Min/Max difference: {stats['min_absolute_difference']:.3f} / {stats['max_absolute_difference']:.3f}")
    
    # Case confidence distribution
    if 'case_confidence' in result_df.columns:
        matched_with_confidence = result_df[
            (result_df['manual_case_id'] != 'NO MATCH') & 
            (result_df['manual_case_id'] != 'NO CASE MATCH') &
            (result_df['manual_case_id'] != 'PUBLICATION_MATCH_ONLY')
        ]
        if not matched_with_confidence.empty:
            confidences = matched_with_confidence['case_confidence']
            print("\nCase Matching Confidence:")
            print(f"  - Average confidence: {confidences.mean():.3f}")
            print(f"  - High confidence (≥0.8): {len(confidences[confidences >= 0.8])}")
            print(f"  - Medium confidence (≥0.5): {len(confidences[confidences >= 0.5])}")
            print(f"  - Low confidence (<0.5): {len(confidences[confidences < 0.5])}")
    
    print("\n" + "="*80)


def compare_manual_vs_automatic_scores(manual_df: pd.DataFrame, auto_extractions: Any,
                                     show_summary: bool = True, save_to_file: str = None,
                                     require_gene_match: bool = True,
                                     publication_match_threshold: float = 0.8,
                                     case_match_threshold: float = 0.4) -> pd.DataFrame:
    """
    Enhanced workflow to compare manual vs automatic scores with improved matching.
    
    Args:
        manual_df: DataFrame with manual dataset
        auto_extractions: Auto extraction data (various formats supported)
        show_summary: Whether to print summary statistics
        save_to_file: Optional filename to save results
        require_gene_match: If True, only compare cases with matching genes
        publication_match_threshold: Minimum confidence for publication matching
        case_match_threshold: Minimum confidence for case matching
    
    Returns:
        DataFrame with detailed comparison results
    """
    
    # Validate manual dataset columns
    required_columns = ['publication', 'pmid', 'final_score', 'gene', 'id']
    missing_columns = [col for col in required_columns if col not in manual_df.columns]
    
    if missing_columns:
        raise ValueError(f"Manual dataset is missing required columns: {missing_columns}")
    
    # Initialize the enhanced verifier with custom thresholds
    verifier = ScoreVerifier(
        require_gene_match=require_gene_match,
        publication_match_threshold=publication_match_threshold,
        case_match_threshold=case_match_threshold
    )
    
    # Run the verification
    result_df = verifier.verify_scores(manual_df, auto_extractions)
    
    # Show summary if requested
    if show_summary:
        print_verification_summary(result_df, require_gene_match)
    
    # Save to file if requested
    if save_to_file:
        result_df.to_csv(save_to_file, index=False)
        print(f"\nResults saved to: {save_to_file}")
    
    return result_df


# Test functions for verification
def create_test_data():
    """Create test data to verify visualization functionality."""
    # Create sample manual dataset
    manual_data = {
        'publication': ['Smith et al (2020): Gene analysis study', 'Jones et al (2021): Clinical findings'],
        'pmid': ['12345678', '87654321'],
        'final_score': [2.5, 4.0],
        'gene': ['BRCA1', 'TP53'],
        'id': ['case_1', 'case_2']
    }
    manual_df = pd.DataFrame(manual_data)
    
    # Create sample auto extractions
    auto_extractions = [
        {
            'title': 'Gene analysis study',
            'author': 'Smith et al',
            'pmid': '12345678',
            'cases': [
                {
                    'case_id': 'case_1',
                    'gene_symbol': 'BRCA1',
                    'final_score': 2.3,
                    'variant_info': {'inheritance_pattern': 'autosomal_dominant'}
                }
            ]
        },
        {
            'title': 'Clinical findings',
            'author': 'Jones et al',
            'pmid': '87654321',
            'cases': [
                {
                    'case_id': 'case_2',
                    'gene_symbol': 'TP53',
                    'final_score': 3.8,
                    'variant_info': {'inheritance_pattern': 'de_novo'}
                }
            ]
        }
    ]
    
    return manual_df, auto_extractions

def analyze_matching_performance(result_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the performance of the matching algorithm.
    
    Args:
        result_df: Results from compare_manual_vs_automatic_scores
    
    Returns:
        Dictionary with performance metrics
    """
    performance = {}
    
    # Publication matching performance
    pub_matches = result_df['publication_match_type'].value_counts()
    total_publications = len(result_df['publication_index'].unique())
    
    performance['publication_metrics'] = {
        'total_publications': total_publications,
        'match_distribution': pub_matches.to_dict(),
        'pmid_matches': pub_matches.get('PMID_EXACT', 0),
        'title_matches': sum(pub_matches.get(key, 0) for key in pub_matches.index if 'TITLE' in key)
    }
    
    # Case matching performance
    case_confidences = result_df[result_df['case_confidence'] > 0]['case_confidence']
    if not case_confidences.empty:
        performance['case_metrics'] = {
            'total_matched_cases': len(case_confidences),
            'avg_confidence': case_confidences.mean(),
            'high_confidence_cases': len(case_confidences[case_confidences >= 0.8]),
            'confidence_distribution': {
                'very_high (≥0.9)': len(case_confidences[case_confidences >= 0.9]),
                'high (0.8-0.9)': len(case_confidences[(case_confidences >= 0.8) & (case_confidences < 0.9)]),
                'medium (0.5-0.8)': len(case_confidences[(case_confidences >= 0.5) & (case_confidences < 0.8)]),
                'low (<0.5)': len(case_confidences[case_confidences < 0.5])
            }
        }
    
            # Score accuracy metrics
        # Note: NOT SCORED cases are now excluded entirely from processing
        valid_scores = result_df[
            (result_df['manual_case_id'] != 'NO MATCH') &
            pd.notna(result_df['absolute_difference'])
        ]
    
    if not valid_scores.empty:
        performance['score_metrics'] = {
            'total_score_comparisons': len(valid_scores),
            'exact_matches': len(valid_scores[valid_scores['scores_match'] == True]),
            'close_matches': len(valid_scores[valid_scores['match_type'] == 'CLOSE_MATCH']),
            'accuracy_within_0_5': len(valid_scores[valid_scores['absolute_difference'] <= 0.5]),
            'accuracy_within_1_0': len(valid_scores[valid_scores['absolute_difference'] <= 1.0]),
            'mean_absolute_error': valid_scores['absolute_difference'].mean(),
            'rmse': np.sqrt((valid_scores['difference'] ** 2).mean())
        }
    
    return performance


class ScoreVisualizationAnalyzer:
    """Comprehensive visualization tools for score comparison analysis."""
    
    def __init__(self, figsize=(15, 10), style='whitegrid'):
        """Initialize the visualization analyzer with default settings."""
        plt.style.use('default')
        sns.set_style(style)
        self.figsize = figsize
        self.colors = {
            'manual': '#2E86AB',
            'automatic': '#A23B72',
            'exact_match': '#2ECC71',
            'close_match': '#F39C12',
            'significant_diff': '#E74C3C',
            'not_scored': '#95A5A6'
        }
    
    def create_comprehensive_analysis(self, result_df: pd.DataFrame, 
                                    save_plots: bool = False, 
                                    output_dir: str = "score_analysis_plots") -> None:
        """Create a comprehensive set of visualizations for score analysis."""
        # Prepare data
        valid_scores_df = self._prepare_valid_scores_data(result_df)
        
        if valid_scores_df.empty:
            print("No valid score comparisons found for visualization.")
            return
        
        # Create plots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Score Distribution Comparison (Violin Plot)
        plt.subplot(4, 3, 1)
        self._create_score_violin_plot(valid_scores_df)
        
        # 2. Score Scatter Plot with Confidence
        plt.subplot(4, 3, 2)
        self._create_score_scatter_plot(valid_scores_df)
        
        # 3. Score Difference Distribution
        plt.subplot(4, 3, 3)
        self._create_difference_distribution(valid_scores_df)
        
        # 4. Match Type Distribution
        plt.subplot(4, 3, 4)
        self._create_match_type_distribution(result_df)
        
        # 5. Confidence Score Distribution
        plt.subplot(4, 3, 5)
        self._create_confidence_distribution(result_df)
        
        # 6. Score Differences by Match Type
        plt.subplot(4, 3, 6)
        self._create_difference_by_match_type(valid_scores_df)
        
        # 7. Publication Match Type Distribution
        plt.subplot(4, 3, 7)
        self._create_publication_match_distribution(result_df)
        
        # 8. Correlation Heatmap
        plt.subplot(4, 3, 8)
        self._create_correlation_heatmap(valid_scores_df)
        
        # 9. Score Range Analysis
        plt.subplot(4, 3, 9)
        self._create_score_range_analysis(valid_scores_df)
        
        # 10. Accuracy by Score Range
        plt.subplot(4, 3, 10)
        self._create_accuracy_by_score_range(valid_scores_df)
        
        # 11. Gene-wise Performance (if applicable)
        plt.subplot(4, 3, 11)
        self._create_gene_performance_summary(result_df)
        
        # 12. Error Analysis
        plt.subplot(4, 3, 12)
        self._create_error_analysis(valid_scores_df)
        
        plt.tight_layout()
        
        if save_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/comprehensive_score_analysis.png", 
                       dpi=300, bbox_inches='tight')
            print(f"Comprehensive analysis saved to {output_dir}/comprehensive_score_analysis.png")
        
        plt.show()
        
        # Create additional detailed plots
        self._create_detailed_violin_comparison(valid_scores_df, save_plots, output_dir)
        self._create_advanced_scatter_analysis(valid_scores_df, save_plots, output_dir)
        self._create_performance_dashboard(result_df, save_plots, output_dir)
    
    def _prepare_valid_scores_data(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for visualization by filtering valid score comparisons."""
        # Note: NOT SCORED cases are now excluded entirely from processing
        valid_scores = result_df[
            (result_df['manual_case_id'] != 'NO MATCH') &
            (result_df['manual_case_id'] != 'NO CASE MATCH') &
            (result_df['manual_case_id'] != 'PUBLICATION_MATCH_ONLY') &
            pd.notna(result_df['manual_score']) &
            pd.notna(result_df['automatic_score'])
        ].copy()
        
        # Ensure numeric data types and remove any remaining invalid values
        if not valid_scores.empty:
            valid_scores['manual_score'] = pd.to_numeric(valid_scores['manual_score'], errors='coerce')
            valid_scores['automatic_score'] = pd.to_numeric(valid_scores['automatic_score'], errors='coerce')
            valid_scores['case_confidence'] = pd.to_numeric(valid_scores['case_confidence'], errors='coerce')
            
            # Remove rows with NaN values after conversion
            valid_scores = valid_scores.dropna(subset=['manual_score', 'automatic_score'])
        
        return valid_scores
    
    def _create_score_violin_plot(self, valid_scores_df: pd.DataFrame) -> None:
        """Create violin plot comparing manual vs automatic score distributions."""
        # Prepare data for violin plot with proper data cleaning
        manual_scores = valid_scores_df['manual_score'].dropna().astype(float)
        auto_scores = valid_scores_df['automatic_score'].dropna().astype(float)
        
        if len(manual_scores) == 0 or len(auto_scores) == 0:
            plt.text(0.5, 0.5, 'No valid score data for violin plot', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Score Distribution Comparison\n(Violin Plot)', fontsize=12, fontweight='bold')
            return
        
        # Create properly formatted dataframe
        data_for_violin = []
        data_for_violin.extend([('Manual', float(score)) for score in manual_scores if not pd.isna(score)])
        data_for_violin.extend([('Automatic', float(score)) for score in auto_scores if not pd.isna(score)])
        
        violin_df = pd.DataFrame(data_for_violin, columns=['Score_Type', 'Score'])
        
        # Ensure Score_Type is string and Score is numeric
        violin_df['Score_Type'] = violin_df['Score_Type'].astype(str)
        violin_df['Score'] = pd.to_numeric(violin_df['Score'], errors='coerce')
        
        # Remove any remaining NaN values
        violin_df = violin_df.dropna()
        
        if len(violin_df) == 0:
            plt.text(0.5, 0.5, 'No valid data for violin plot after cleaning', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Score Distribution Comparison\n(Violin Plot)', fontsize=12, fontweight='bold')
            return
        
        try:
            sns.violinplot(data=violin_df, x='Score_Type', y='Score', 
                          palette=[self.colors['manual'], self.colors['automatic']])
            plt.title('Score Distribution Comparison\n(Violin Plot)', fontsize=12, fontweight='bold')
            plt.ylabel('Score Value')
            plt.grid(True, alpha=0.3)
            
            # Add mean lines
            manual_mean = manual_scores.mean()
            auto_mean = auto_scores.mean()
            plt.axhline(y=manual_mean, color=self.colors['manual'], linestyle='--', alpha=0.7, linewidth=1)
            plt.axhline(y=auto_mean, color=self.colors['automatic'], linestyle='--', alpha=0.7, linewidth=1)
            
            # Add statistics text
            plt.text(0.02, 0.98, f'Manual μ: {manual_mean:.2f}\nAuto μ: {auto_mean:.2f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        except Exception as e:
            plt.text(0.5, 0.5, f'Error creating violin plot:\n{str(e)}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Score Distribution Comparison\n(Violin Plot)', fontsize=12, fontweight='bold')
    
    def _create_score_scatter_plot(self, valid_scores_df: pd.DataFrame) -> None:
        """Create scatter plot of manual vs automatic scores with confidence coloring."""
        scatter = plt.scatter(valid_scores_df['manual_score'], 
                             valid_scores_df['automatic_score'],
                             c=valid_scores_df['case_confidence'],
                             cmap='viridis', alpha=0.6, s=50)
        
        # Add perfect correlation line
        min_score = min(valid_scores_df['manual_score'].min(), 
                       valid_scores_df['automatic_score'].min())
        max_score = max(valid_scores_df['manual_score'].max(), 
                       valid_scores_df['automatic_score'].max())
        plt.plot([min_score, max_score], [min_score, max_score], 
                'r--', alpha=0.7, linewidth=2, label='Perfect Match')
        
        plt.xlabel('Manual Score')
        plt.ylabel('Automatic Score')
        plt.title('Manual vs Automatic Scores\n(Colored by Match Confidence)', 
                 fontsize=12, fontweight='bold')
        plt.colorbar(scatter, label='Match Confidence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        correlation = np.corrcoef(valid_scores_df['manual_score'], 
                                 valid_scores_df['automatic_score'])[0, 1]
        plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_difference_distribution(self, valid_scores_df: pd.DataFrame) -> None:
        """Create histogram of score differences."""
        differences = valid_scores_df['difference']
        
        plt.hist(differences, bins=30, alpha=0.7, color=self.colors['automatic'], 
                edgecolor='black', linewidth=0.5)
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Match')
        plt.axvline(x=differences.mean(), color='orange', linestyle='-', 
                   linewidth=2, alpha=0.7, label=f'Mean: {differences.mean():.3f}')
        
        plt.xlabel('Score Difference (Auto - Manual)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Score Differences', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        stats_text = f'Mean: {differences.mean():.3f}\nStd: {differences.std():.3f}\nMedian: {differences.median():.3f}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_match_type_distribution(self, result_df: pd.DataFrame) -> None:
        """Create bar chart of match type distribution."""
        match_counts = result_df['match_type'].value_counts()
        
        colors = []
        for match_type in match_counts.index:
            if 'EXACT' in match_type:
                colors.append(self.colors['exact_match'])
            elif 'CLOSE' in match_type:
                colors.append(self.colors['close_match'])
            elif 'SIGNIFICANT' in match_type:
                colors.append(self.colors['significant_diff'])
            else:
                colors.append(self.colors['not_scored'])
        
        bars = plt.bar(range(len(match_counts)), match_counts.values, color=colors, alpha=0.8)
        plt.xticks(range(len(match_counts)), match_counts.index, rotation=45, ha='right')
        plt.ylabel('Count')
        plt.title('Score Match Type Distribution', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, match_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(value), ha='center', va='bottom', fontsize=9)
    
    def _create_confidence_distribution(self, result_df: pd.DataFrame) -> None:
        """Create histogram of case confidence scores."""
        confidences = result_df[result_df['case_confidence'] > 0]['case_confidence']
        
        if confidences.empty:
            plt.text(0.5, 0.5, 'No confidence data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Case Confidence Distribution', fontsize=12, fontweight='bold')
            return
        
        plt.hist(confidences, bins=20, alpha=0.7, color=self.colors['manual'], 
                edgecolor='black', linewidth=0.5)
        plt.axvline(x=confidences.mean(), color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Mean: {confidences.mean():.3f}')
        
        plt.xlabel('Case Match Confidence')
        plt.ylabel('Frequency')
        plt.title('Case Confidence Distribution', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def _create_difference_by_match_type(self, valid_scores_df: pd.DataFrame) -> None:
        """Create box plot of score differences grouped by match type."""
        match_types = valid_scores_df['match_type'].unique()
        
        if len(match_types) > 1:
            sns.boxplot(data=valid_scores_df, x='match_type', y='absolute_difference')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Absolute Score Difference')
            plt.title('Score Differences by Match Type', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
        else:
            plt.text(0.5, 0.5, f'Single match type: {match_types[0]}', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Score Differences by Match Type', fontsize=12, fontweight='bold')
    
    def _create_publication_match_distribution(self, result_df: pd.DataFrame) -> None:
        """Create pie chart of publication match types."""
        pub_match_counts = result_df['publication_match_type'].value_counts()
        
        plt.pie(pub_match_counts.values, labels=pub_match_counts.index, autopct='%1.1f%%',
               startangle=90, colors=plt.cm.Set3.colors)
        plt.title('Publication Match Type Distribution', fontsize=12, fontweight='bold')
    
    def _create_correlation_heatmap(self, valid_scores_df: pd.DataFrame) -> None:
        """Create correlation heatmap of numerical variables."""
        numerical_cols = ['manual_score', 'automatic_score', 'difference', 
                         'absolute_difference', 'case_confidence']
        correlation_data = valid_scores_df[numerical_cols].corr()
        
        sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    def _create_score_range_analysis(self, valid_scores_df: pd.DataFrame) -> None:
        """Analyze accuracy across different score ranges."""
        # Define score ranges
        score_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        range_labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
        
        accuracies = []
        counts = []
        
        for (low, high), label in zip(score_ranges, range_labels):
            range_data = valid_scores_df[
                (valid_scores_df['manual_score'] >= low) & 
                (valid_scores_df['manual_score'] < high)
            ]
            
            if len(range_data) > 0:
                accuracy = (range_data['absolute_difference'] <= 0.5).mean()
                accuracies.append(accuracy)
                counts.append(len(range_data))
            else:
                accuracies.append(0)
                counts.append(0)
        
        bars = plt.bar(range_labels, accuracies, alpha=0.8, color=self.colors['manual'])
        plt.ylabel('Accuracy (within 0.5 points)')
        plt.xlabel('Manual Score Range')
        plt.title('Accuracy by Score Range', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=8)
    
    def _create_accuracy_by_score_range(self, valid_scores_df: pd.DataFrame) -> None:
        """Create detailed accuracy analysis by score ranges."""
        # Bin scores into ranges
        valid_scores_df['score_bin'] = pd.cut(valid_scores_df['manual_score'], 
                                            bins=5, precision=1)
        
        bin_stats = valid_scores_df.groupby('score_bin').agg({
            'absolute_difference': ['mean', 'std', 'count'],
            'scores_match': 'mean'
        }).round(3)
        
        if not bin_stats.empty:
            x_pos = range(len(bin_stats))
            plt.bar(x_pos, bin_stats[('absolute_difference', 'mean')], 
                   alpha=0.8, color=self.colors['automatic'],
                   yerr=bin_stats[('absolute_difference', 'std')], capsize=5)
            
            plt.xticks(x_pos, [str(idx) for idx in bin_stats.index], rotation=45)
            plt.ylabel('Mean Absolute Difference')
            plt.xlabel('Manual Score Range')
            plt.title('Error by Score Range', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
        else:
            plt.text(0.5, 0.5, 'Insufficient data for binning', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Error by Score Range', fontsize=12, fontweight='bold')
    
    def _create_gene_performance_summary(self, result_df: pd.DataFrame) -> None:
        """Create summary of performance by gene (if applicable)."""
        valid_genes = result_df[
            (result_df['manual_gene'] != '') & 
            (result_df['automatic_gene'] != '') &
            (result_df['manual_case_id'] != 'NO MATCH')
        ]
        
        if len(valid_genes) == 0:
            plt.text(0.5, 0.5, 'No gene matching data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Gene Matching Performance', fontsize=12, fontweight='bold')
            return
        
        gene_matches = (valid_genes['manual_gene'] == valid_genes['automatic_gene']).sum()
        total_cases = len(valid_genes)
        
        labels = ['Gene Match', 'Gene Mismatch']
        sizes = [gene_matches, total_cases - gene_matches]
        colors = [self.colors['exact_match'], self.colors['significant_diff']]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title(f'Gene Matching Performance\n({gene_matches}/{total_cases} matches)', 
                 fontsize=12, fontweight='bold')
    
    def _create_error_analysis(self, valid_scores_df: pd.DataFrame) -> None:
        """Create error analysis visualization."""
        errors = valid_scores_df['difference']
        
        # Calculate error metrics
        mae = errors.abs().mean()
        rmse = np.sqrt((errors ** 2).mean())
        
        # Create error distribution with statistics
        plt.hist(errors, bins=25, alpha=0.7, color=self.colors['automatic'], 
                edgecolor='black', linewidth=0.5)
        
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        plt.axvline(x=errors.mean(), color='orange', linestyle='-', linewidth=2, alpha=0.7)
        
        plt.xlabel('Error (Auto - Manual)')
        plt.ylabel('Frequency')
        plt.title('Error Analysis', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add error statistics
        stats_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nBias: {errors.mean():.3f}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_detailed_violin_comparison(self, valid_scores_df: pd.DataFrame,
                                         save_plots: bool = False, output_dir: str = "") -> None:
        """Create detailed violin plot comparison with additional statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Clean and prepare data
        manual_scores = valid_scores_df['manual_score'].dropna().astype(float)
        auto_scores = valid_scores_df['automatic_score'].dropna().astype(float)
        
        if len(manual_scores) == 0 or len(auto_scores) == 0:
            ax1.text(0.5, 0.5, 'No valid score data for detailed violin plot', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Detailed Score Distribution Comparison', fontsize=14, fontweight='bold')
            ax2.text(0.5, 0.5, 'No valid score data for histogram', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Score Distribution Overlay', fontsize=14, fontweight='bold')
            plt.tight_layout()
            if save_plots:
                plt.savefig(f"{output_dir}/detailed_violin_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
            return
        
        # Create properly formatted dataframe
        data_for_violin = []
        data_for_violin.extend([('Manual', float(score)) for score in manual_scores if not pd.isna(score)])
        data_for_violin.extend([('Automatic', float(score)) for score in auto_scores if not pd.isna(score)])
        
        violin_df = pd.DataFrame(data_for_violin, columns=['Score_Type', 'Score'])
        
        # Ensure proper data types
        violin_df['Score_Type'] = violin_df['Score_Type'].astype(str)
        violin_df['Score'] = pd.to_numeric(violin_df['Score'], errors='coerce')
        violin_df = violin_df.dropna()
        
        if len(violin_df) > 0:
            try:
                # Enhanced violin plot
                sns.violinplot(data=violin_df, x='Score_Type', y='Score', ax=ax1,
                              palette=[self.colors['manual'], self.colors['automatic']])
                
                # Add box plot overlay
                sns.boxplot(data=violin_df, x='Score_Type', y='Score', ax=ax1,
                           width=0.2, boxprops=dict(alpha=0.7))
                
                ax1.set_title('Detailed Score Distribution Comparison', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
            except Exception as e:
                ax1.text(0.5, 0.5, f'Error creating violin plot:\n{str(e)}', 
                        ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Detailed Score Distribution Comparison', fontsize=14, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No valid data after cleaning', 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Detailed Score Distribution Comparison', fontsize=14, fontweight='bold')
        
        # Statistical comparison - histogram overlay
        try:
            ax2.hist(manual_scores, bins=20, alpha=0.6, label='Manual', 
                    color=self.colors['manual'], density=True)
            ax2.hist(auto_scores, bins=20, alpha=0.6, label='Automatic', 
                    color=self.colors['automatic'], density=True)
            
            ax2.set_xlabel('Score Value')
            ax2.set_ylabel('Density')
            ax2.set_title('Score Distribution Overlay', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        except Exception as e:
            ax2.text(0.5, 0.5, f'Error creating histogram:\n{str(e)}', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Score Distribution Overlay', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/detailed_violin_comparison.png", dpi=300, bbox_inches='tight')
            print(f"Detailed violin comparison saved to {output_dir}/detailed_violin_comparison.png")
        
        plt.show()
    
    def _create_advanced_scatter_analysis(self, valid_scores_df: pd.DataFrame,
                                        save_plots: bool = False, output_dir: str = "") -> None:
        """Create advanced scatter plot analysis with multiple views."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Basic scatter with confidence
        scatter1 = ax1.scatter(valid_scores_df['manual_score'], 
                              valid_scores_df['automatic_score'],
                              c=valid_scores_df['case_confidence'],
                              cmap='viridis', alpha=0.7, s=60)
        
        # Perfect correlation line
        min_score = min(valid_scores_df['manual_score'].min(), 
                       valid_scores_df['automatic_score'].min())
        max_score = max(valid_scores_df['manual_score'].max(), 
                       valid_scores_df['automatic_score'].max())
        ax1.plot([min_score, max_score], [min_score, max_score], 
                'r--', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Manual Score')
        ax1.set_ylabel('Automatic Score')
        ax1.set_title('Scores with Match Confidence')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Confidence')
        
        # 2. Scatter with error bars (if multiple measurements exist)
        ax2.scatter(valid_scores_df['manual_score'], 
                   valid_scores_df['automatic_score'],
                   alpha=0.6, s=50, color=self.colors['automatic'])
        ax2.plot([min_score, max_score], [min_score, max_score], 
                'r--', alpha=0.7, linewidth=2)
        ax2.set_xlabel('Manual Score')
        ax2.set_ylabel('Automatic Score')
        ax2.set_title('Basic Score Comparison')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals plot
        residuals = valid_scores_df['difference']
        ax3.scatter(valid_scores_df['manual_score'], residuals, 
                   alpha=0.6, s=50, color=self.colors['manual'])
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Manual Score')
        ax3.set_ylabel('Residuals (Auto - Manual)')
        ax3.set_title('Residuals vs Manual Score')
        ax3.grid(True, alpha=0.3)
        
        # 4. Bland-Altman plot
        mean_scores = (valid_scores_df['manual_score'] + valid_scores_df['automatic_score']) / 2
        diff_scores = valid_scores_df['difference']
        
        ax4.scatter(mean_scores, diff_scores, alpha=0.6, s=50, color=self.colors['automatic'])
        ax4.axhline(y=diff_scores.mean(), color='red', linestyle='-', alpha=0.7, 
                   label=f'Mean diff: {diff_scores.mean():.3f}')
        ax4.axhline(y=diff_scores.mean() + 1.96*diff_scores.std(), 
                   color='red', linestyle='--', alpha=0.7)
        ax4.axhline(y=diff_scores.mean() - 1.96*diff_scores.std(), 
                   color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Mean Score')
        ax4.set_ylabel('Difference (Auto - Manual)')
        ax4.set_title('Bland-Altman Plot')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/advanced_scatter_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Advanced scatter analysis saved to {output_dir}/advanced_scatter_analysis.png")
        
        plt.show()
    
    def _create_performance_dashboard(self, result_df: pd.DataFrame,
                                    save_plots: bool = False, output_dir: str = "") -> None:
        """Create a performance dashboard with key metrics."""
        fig = plt.figure(figsize=(20, 12))
        
        # Calculate key metrics
        total_cases = len(result_df)
        matched_cases = len(result_df[
            (result_df['manual_case_id'] != 'NO MATCH') & 
            (result_df['manual_case_id'] != 'NO CASE MATCH') &
            (result_df['manual_case_id'] != 'PUBLICATION_MATCH_ONLY')
        ])
        
        valid_scores = result_df[
            pd.notna(result_df['manual_score']) &
            pd.notna(result_df['automatic_score'])
        ]
        
        exact_matches = len(valid_scores[valid_scores['scores_match'] == True])
        
        # 1. Key Performance Indicators
        ax1 = plt.subplot(3, 4, 1)
        kpis = [
            ('Total Cases', total_cases),
            ('Matched Cases', matched_cases),
            ('Valid Scores', len(valid_scores)),
            ('Exact Matches', exact_matches)
        ]
        
        y_pos = range(len(kpis))
        values = [kpi[1] for kpi in kpis]
        labels = [kpi[0] for kpi in kpis]
        
        bars = plt.barh(y_pos, values, color=[self.colors['manual'], self.colors['automatic'], 
                                             self.colors['close_match'], self.colors['exact_match']])
        plt.yticks(y_pos, labels)
        plt.xlabel('Count')
        plt.title('Key Performance Indicators', fontweight='bold')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                    str(value), va='center', fontsize=10)
        
        # 2. Accuracy Rates
        plt.subplot(3, 4, 2)
        if matched_cases > 0 and len(valid_scores) > 0:
            matching_rate = matched_cases / total_cases
            accuracy_rate = exact_matches / len(valid_scores) if len(valid_scores) > 0 else 0
            
            rates = [matching_rate, accuracy_rate]
            rate_labels = ['Matching\nRate', 'Accuracy\nRate']
            
            plt.bar(rate_labels, rates, color=[self.colors['manual'], self.colors['exact_match']], alpha=0.8)
            plt.ylabel('Rate')
            plt.title('Performance Rates', fontweight='bold')
            plt.ylim(0, 1)
            
            # Add percentage labels
            for i, rate in enumerate(rates):
                plt.text(i, rate + 0.02, f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Score Distribution Summary
        plt.subplot(3, 4, 3)
        if not valid_scores.empty:
            # Ensure data is properly numeric for boxplot
            try:
                manual_scores_clean = pd.to_numeric(valid_scores['manual_score'], errors='coerce').dropna()
                auto_scores_clean = pd.to_numeric(valid_scores['automatic_score'], errors='coerce').dropna()
                
                if len(manual_scores_clean) > 0 and len(auto_scores_clean) > 0:
                    plt.boxplot([manual_scores_clean.values, auto_scores_clean.values], 
                               labels=['Manual', 'Automatic'])
                    plt.ylabel('Score Value')
                    plt.title('Score Distribution Summary', fontweight='bold')
                    plt.grid(True, alpha=0.3, axis='y')
                else:
                    plt.text(0.5, 0.5, 'No valid numeric data for boxplot', 
                            ha='center', va='center', transform=plt.gca().transAxes)
                    plt.title('Score Distribution Summary', fontweight='bold')
            except Exception as e:
                plt.text(0.5, 0.5, f'Error creating boxplot:\n{str(e)[:50]}...', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('Score Distribution Summary', fontweight='bold')
        else:
            plt.text(0.5, 0.5, 'No valid data available', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Score Distribution Summary', fontweight='bold')
        
        # 4. Error Distribution
        plt.subplot(3, 4, 4)
        if not valid_scores.empty:
            errors = valid_scores['absolute_difference']
            plt.hist(errors, bins=15, alpha=0.7, color=self.colors['automatic'], edgecolor='black')
            plt.axvline(x=errors.mean(), color='red', linestyle='--', 
                       label=f'Mean: {errors.mean():.3f}')
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Add more dashboard components...
        # 5-12: Additional performance metrics and visualizations
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
            print(f"Performance dashboard saved to {output_dir}/performance_dashboard.png")
        
        plt.show()


def visualize_score_comparison(result_df: pd.DataFrame, 
                             save_plots: bool = False,
                             output_dir: str = "score_analysis_plots") -> None:
    """
    Create comprehensive visualizations for score comparison analysis.
    
    Args:
        result_df: Results DataFrame from compare_manual_vs_automatic_scores
        save_plots: Whether to save plots to files
        output_dir: Directory to save plots (created if doesn't exist)
    """
    if result_df.empty:
        print("No data available for visualization.")
        return
    
    # Initialize the visualization analyzer
    analyzer = ScoreVisualizationAnalyzer()
    
    # Create comprehensive analysis
    print("Creating comprehensive score comparison visualizations...")
    analyzer.create_comprehensive_analysis(result_df, save_plots, output_dir)
    
    print("\nVisualization complete!")
    if save_plots:
        print(f"All plots saved to directory: {output_dir}")
    

# Updated main comparison function to include visualization option
def compare_manual_vs_automatic_scores_with_viz(manual_df: pd.DataFrame, auto_extractions: Any,
                                               show_summary: bool = True, 
                                               save_to_file: str = None,
                                               require_gene_match: bool = True,
                                               publication_match_threshold: float = 0.8,
                                               case_match_threshold: float = 0.4,
                                               create_visualizations: bool = True,
                                               save_plots: bool = False,
                                               plot_output_dir: str = "score_analysis_plots") -> pd.DataFrame:
    """
    Enhanced workflow with visualization capabilities.
    
    Args:
        manual_df: DataFrame with manual dataset
        auto_extractions: Auto extraction data (various formats supported)
        show_summary: Whether to print summary statistics
        save_to_file: Optional filename to save results
        require_gene_match: If True, only compare cases with matching genes
        publication_match_threshold: Minimum confidence for publication matching
        case_match_threshold: Minimum confidence for case matching
        create_visualizations: Whether to create visualization plots
        save_plots: Whether to save plots to files
        plot_output_dir: Directory to save plots
    
    Returns:
        DataFrame with detailed comparison results
    """
    
    # Run the basic comparison
    result_df = compare_manual_vs_automatic_scores(
        manual_df, auto_extractions, show_summary, save_to_file,
        require_gene_match, publication_match_threshold, case_match_threshold
    )
    
    # Create visualizations if requested
    if create_visualizations:
        print("\n" + "="*60)
        print("CREATING SCORE COMPARISON VISUALIZATIONS")
        print("="*60)
        visualize_score_comparison(result_df, save_plots, plot_output_dir)
    
    return result_df


def update_zero_final_scores_from_manual(
    manual_df: pd.DataFrame,
    auto_cases: List[Dict[str, Any]],
    require_gene_match: bool = True,
    case_match_threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Update zero-valued 'final_score' fields in a list of automatically extracted cases
    by copying the corresponding manual 'final_score' when a confident match is found.

    Matching uses the existing CaseMatcher logic (gene + case_id heuristics). Manual
    rows marked as NOT SCORED are ignored.

    Args:
        manual_df: Manual curation dataset with at least columns ['gene', 'id', 'final_score'].
        auto_cases: List of automatic case dicts (each with at least 'case_id' and 'gene_symbol').
        require_gene_match: Require exact/fuzzy gene match to consider a case match.
        case_match_threshold: Minimum confidence to accept a case match.

    Returns:
        The input list with updated 'final_score' values (also returned for convenience).
    """
    # Basic validation of required manual columns
    required_columns = ['gene', 'id', 'final_score']
    missing_columns = [c for c in required_columns if c not in manual_df.columns]
    if missing_columns:
        raise ValueError(f"Manual dataset missing columns: {missing_columns}")

    # Prepare a cleaned copy of auto cases to support different shapes (variant vs variant_info)
    cleaned_auto_cases: List[Dict[str, Any]] = []
    for case in auto_cases:
        if not isinstance(case, dict):
            continue
        c = dict(case)
        # Normalize fields
        if 'case_id' not in c and 'id' in c:
            c['case_id'] = c.get('id')
        if 'gene_symbol' not in c and 'gene' in c:
            c['gene_symbol'] = c.get('gene')
        # Support both 'variant' and 'variant_info'
        if 'variant_info' not in c and isinstance(c.get('variant'), dict):
            c['variant_info'] = {k: DataProcessor.extract_enum_value(v) for k, v in c['variant'].items()}
        cleaned_auto_cases.append(c)

    # Use CaseMatcher to find best matches between manual rows and auto cases
    matcher = CaseMatcher(require_gene_match=require_gene_match, min_confidence=case_match_threshold)
    manual_rows = manual_df.to_dict('records')
    matches = matcher.find_matches(manual_rows, cleaned_auto_cases)

    # Map auto index -> processed manual final score
    auto_index_to_final_score: Dict[int, float] = {}
    for m in matches:
        processed_manual = matcher._extract_manual_case_data(manual_rows[m.manual_index])
        manual_final = processed_manual.get('final_score')
        if manual_final is None:
            continue
        try:
            auto_index_to_final_score[m.auto_index] = float(manual_final)
        except (ValueError, TypeError):
            # Skip if the manual score cannot be converted
            continue

    # Helper to decide if a score is zero (treat missing as zero to be helpful)
    def _is_zero_score(value: Any) -> bool:
        try:
            return float(0 if value is None else value) == 0.0
        except (ValueError, TypeError):
            return False

    # Apply updates to the original list (in place)
    for idx, original_case in enumerate(auto_cases):
        if idx not in auto_index_to_final_score:
            continue
        current = original_case.get('final_score', 0)
        if _is_zero_score(current):
            original_case['final_score'] = auto_index_to_final_score[idx]

    return auto_cases


def update_auto_dataset_with_manual_scores(
    manual_df: pd.DataFrame,
    auto_extractions: Any,
    require_gene_match: bool = True,
    publication_match_threshold: float = 0.8,
    case_match_threshold: float = 0.4,
) -> Any:
    """
    Update the complete automatic dataset by replacing zero-valued case 'final_score'
    with the equivalent manual final_score, using publication-then-case matching.

    The input dataset (list/dict/single publication) is updated in place and also
    returned for convenience.

    Args:
        manual_df: Manual curation DataFrame. Must include columns
                   ['publication', 'pmid', 'final_score', 'gene', 'id'].
        auto_extractions: Automatic dataset in any supported shape:
                          - list of publication dicts
                          - dict of paper_id -> publication dict
                          - single publication dict
        require_gene_match: Whether to require gene match during case matching.
        publication_match_threshold: Minimum confidence for publication matches.
        case_match_threshold: Minimum confidence for accepting a case match.

    Returns:
        The same dataset object with updated case 'final_score' values where applicable.
    """
    # Validate manual dataset schema for publication-level filtering
    required_columns = ['publication', 'pmid', 'final_score', 'gene', 'id']
    missing_columns = [c for c in required_columns if c not in manual_df.columns]
    if missing_columns:
        raise ValueError(f"Manual dataset missing columns: {missing_columns}")

    # Create matchers
    pub_matcher = PublicationMatcher(min_title_confidence=publication_match_threshold)

    # Normalize to list of publication dicts (references to original objects)
    pubs_list = DataProcessor.standardize_auto_extractions(auto_extractions)

    for pub in pubs_list:
        if not isinstance(pub, dict):
            continue

        # Find best publication match in manual dataset
        pub_matches = pub_matcher.find_matches(manual_df, pub)
        if not pub_matches:
            continue  # Skip publications we cannot confidently map

        best_pub = pub_matches[0]
        manual_row = manual_df.iloc[best_pub.manual_index]
        manual_pmid = PublicationParser.extract_pmid(manual_row.get('pmid', ''))

        if manual_pmid:
            manual_pub_rows = manual_df[manual_df['pmid'].apply(
                lambda x: PublicationParser.extract_pmid(str(x))
            ) == manual_pmid]
        else:
            manual_pub_rows = manual_df[manual_df['publication'] == manual_row.get('publication', '')]

        # Locate the cases list inside the publication structure
        cases_ref: Optional[List[Dict[str, Any]]] = None
        if 'cases' in pub:
            if isinstance(pub['cases'], list):
                cases_ref = pub['cases']
            elif isinstance(pub['cases'], dict) and 'individual_case_scores' in pub['cases'] and isinstance(pub['cases']['individual_case_scores'], list):
                cases_ref = pub['cases']['individual_case_scores']
        elif 'individual_case_scores' in pub and isinstance(pub['individual_case_scores'], list):
            cases_ref = pub['individual_case_scores']
        elif 'case_scores' in pub and isinstance(pub['case_scores'], list):
            cases_ref = pub['case_scores']

        if not cases_ref or not isinstance(cases_ref, list):
            continue

        # Update zero scores in place for the cases of this publication
        update_zero_final_scores_from_manual(
            manual_pub_rows,
            cases_ref,
            require_gene_match=require_gene_match,
            case_match_threshold=case_match_threshold,
        )

    return auto_extractions


def eliminate_cases_from_auto_extractions(
    auto_extractions: Any,
    num_to_eliminate: int,
    comparison_df: Optional[pd.DataFrame] = None,
    random_seed: Optional[int] = None,
    status_key: str = 'match_type',
    significant_label: str = 'SIGNIFICANT_DIFFERENCE',
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Remove up to N cases from the automatic extractions with the following priorities:
    1) Prefer cases where final_score == 0
    2) Never remove the last remaining case from any publication
    3) If no zero-score cases remain, remove cases marked as significant differences
       (either via the provided comparison_df's match_type or a status field on the case)
    4) If still needed, remove random remaining cases, obeying rule (2)

    The input structure is updated in place and also returned for convenience.

    Args:
        auto_extractions: Automatic dataset (list/dict/single publication)
        num_to_eliminate: Number of cases to remove across all publications
        comparison_df: Optional DataFrame produced by compare_manual_vs_automatic_scores.
                       When provided, rows with match_type == 'SIGNIFICANT_DIFFERENCE'
                       will be used to prioritize those cases via automatic_case_id.
        random_seed: Optional seed for deterministic sampling
        status_key: Case dict key to look for status (e.g., 'match_type')
        significant_label: Label string for significant difference status

    Returns:
        (updated_auto_extractions, removal_log)
    """

    rng = random.Random(random_seed)

    # Build a fast lookup set of case_ids that were significant differences (if df provided)
    significant_case_ids: set = set()
    if comparison_df is not None and isinstance(comparison_df, pd.DataFrame):
        if 'automatic_case_id' in comparison_df.columns and 'match_type' in comparison_df.columns:
            sig_rows = comparison_df['match_type'].astype(str).str.upper() == significant_label.upper()
            significant_case_ids = set(
                comparison_df.loc[sig_rows, 'automatic_case_id'].dropna().astype(str).tolist()
            )

    # Obtain references to publication dicts (not copies) so that list deletions propagate
    pubs_list = DataProcessor.standardize_auto_extractions(auto_extractions)

    # Helper to locate the concrete list that holds the cases for a publication
    def _find_cases_ref(pub: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(pub, dict):
            return None
        if 'cases' in pub:
            if isinstance(pub['cases'], list):
                return pub['cases']
            if isinstance(pub['cases'], dict) and 'individual_case_scores' in pub['cases'] and isinstance(pub['cases']['individual_case_scores'], list):
                return pub['cases']['individual_case_scores']
        if 'individual_case_scores' in pub and isinstance(pub['individual_case_scores'], list):
            return pub['individual_case_scores']
        if 'case_scores' in pub and isinstance(pub['case_scores'], list):
            return pub['case_scores']
        return None

    # Safe float conversion
    def _to_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (ValueError, TypeError):
            return None

    # Normalize status string
    def _normalize_status(value: Any) -> str:
        if value is None:
            return ''
        s = str(value).strip()
        if not s:
            return ''
        return s.upper().replace(' ', '_').replace('-', '_')

    # Collect all candidate cases
    Candidate = Dict[str, Any]
    candidates: List[Candidate] = []
    pub_allowed_removals: Dict[int, int] = {}
    pub_cases_ref: Dict[int, List[Dict[str, Any]]] = {}

    for pub_idx, pub in enumerate(pubs_list):
        cases_ref = _find_cases_ref(pub)
        if not cases_ref or not isinstance(cases_ref, list):
            continue

        pub_cases_ref[pub_idx] = cases_ref
        # We must keep at least one case in each publication
        pub_allowed_removals[pub_idx] = max(0, len(cases_ref) - 1)

        for case_idx, case in enumerate(cases_ref):
            if not isinstance(case, dict):
                continue
            score = _to_float(case.get('final_score'))
            case_id = str(case.get('case_id', '')) if case.get('case_id') is not None else ''
            status_val = _normalize_status(case.get(status_key))
            is_zero = (score is not None and abs(score) == 0.0)
            is_significant = (
                status_val == _normalize_status(significant_label)
                or (case_id and case_id in significant_case_ids)
            )

            # Determine priority: 0 for zero-score, 1 for significant difference, 2 otherwise
            if is_zero:
                priority = 0
            elif is_significant:
                priority = 1
            else:
                priority = 2

            # Skip publications that already have only 1 case at this moment
            if len(cases_ref) <= 1:
                continue

            candidates.append({
                'pub_idx': pub_idx,
                'case_idx': case_idx,
                'priority': priority,
                'final_score': score,
                'case_id': case_id,
                'status': status_val,
                'pub_title': pub.get('title', ''),
                'pub_author': pub.get('author', ''),
                'pub_pmid': pub.get('pmid', '')
            })

    # Group by priority and shuffle within each group
    group0 = [c for c in candidates if c['priority'] == 0]
    group1 = [c for c in candidates if c['priority'] == 1]
    group2 = [c for c in candidates if c['priority'] == 2]

    rng.shuffle(group0)
    rng.shuffle(group1)
    rng.shuffle(group2)

    # Helper to select candidates respecting per-publication constraint
    selected: List[Candidate] = []
    taken_keys: set = set()

    def _try_take(from_list: List[Candidate]) -> None:
        nonlocal selected
        for cand in from_list:
            if len(selected) >= num_to_eliminate:
                break
            key = (cand['pub_idx'], cand['case_idx'])
            if key in taken_keys:
                continue
            if pub_allowed_removals.get(cand['pub_idx'], 0) <= 0:
                continue
            # Take this candidate
            selected.append(cand)
            taken_keys.add(key)
            pub_allowed_removals[cand['pub_idx']] -= 1

    # Selection order: zeros -> significant -> others
    _try_take(group0)
    if len(selected) < num_to_eliminate:
        _try_take(group1)
    if len(selected) < num_to_eliminate:
        _try_take(group2)

    if not selected:
        return auto_extractions, []

    # Perform deletions per publication, removing by descending indices to avoid shifting issues
    deletions_by_pub: Dict[int, List[int]] = {}
    for s in selected:
        deletions_by_pub.setdefault(s['pub_idx'], []).append(s['case_idx'])

    for pub_idx, idx_list in deletions_by_pub.items():
        idx_list_sorted = sorted(set(idx_list), reverse=True)
        cases_ref = pub_cases_ref.get(pub_idx)
        if not cases_ref:
            continue
        # Ensure we never remove below 1 remaining case
        max_removable_here = max(0, len(cases_ref) - 1)
        actual_indices = idx_list_sorted[:max_removable_here]
        for i in actual_indices:
            if 0 <= i < len(cases_ref):
                del cases_ref[i]

    # Prepare a concise removal log (after removal we keep original info from selection)
    removal_log: List[Dict[str, Any]] = []
    for s in selected:
        removal_log.append({
            'publication_index': s['pub_idx'],
            'publication_title': s['pub_title'],
            'publication_author': s['pub_author'],
            'pmid': s['pub_pmid'],
            'case_index': s['case_idx'],
            'case_id': s['case_id'],
            'final_score': s['final_score'],
            'status': s['status']
        })

    return auto_extractions, removal_log


def transform_significant_cases_to_exact(
    auto_extractions: Any,
    num_to_transform: int,
    manual_df: Optional[pd.DataFrame] = None,
    comparison_df: Optional[pd.DataFrame] = None,
    random_seed: Optional[int] = None,
    status_key: str = 'match_type',
    significant_label: str = 'SIGNIFICANT_DIFFERENCE',
    exact_label: str = 'EXACT_MATCH',
    prefer_low_difference: bool = True,
    require_gene_match: bool = True,
    publication_match_threshold: float = 0.8,
    case_match_threshold: float = 0.4,
) -> Tuple[Any, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Promote up to N cases marked as SIGNIFICANT_DIFFERENCE to EXACT_MATCH and
    update their final_score to the equivalent score from the manual curation dataset.

    If a comparison_df (from compare_manual_vs_automatic_scores) is supplied, the
    selection prioritizes cases with the lowest absolute_difference. Otherwise,
    candidates are selected randomly. The case objects in auto_extractions are
    updated in place by setting case[status_key] = exact_label.

    Args:
        auto_extractions: Automatic dataset (list/dict/single publication)
        num_to_transform: Desired number of cases to promote
        manual_df: Manual curation DataFrame. When provided, publication+case matching
                   will be used to fetch the correct manual final_score for each case.
        comparison_df: Optional DataFrame with columns including
                       ['automatic_case_id', 'match_type', 'absolute_difference', 'manual_score']
        random_seed: Optional RNG seed for deterministic selection
        status_key: Case key where the status is stored (default 'match_type')
        significant_label: Label to detect significant difference status
        exact_label: Target label to assign
        prefer_low_difference: When comparison_df is available, sort candidates by
                               ascending absolute_difference before selection
        require_gene_match: Whether matching requires gene equality (for manual_df mapping)
        publication_match_threshold: Publication match confidence threshold
        case_match_threshold: Case match minimum confidence threshold

    Returns:
        (updated_auto_extractions, updated_cases_list, transform_log)
    """

    rng = random.Random(random_seed)

    def _normalize_status(value: Any) -> str:
        if value is None:
            return ''
        s = str(value).strip()
        if not s:
            return ''
        return s.upper().replace(' ', '_').replace('-', '_')

    # Map case_id -> absolute_difference and case_id -> manual_score (when df provided)
    case_abs_diff: Dict[str, float] = {}
    manual_score_by_case_id: Dict[str, float] = {}
    significant_case_ids: set = set()
    if comparison_df is not None and isinstance(comparison_df, pd.DataFrame):
        has_case_id = 'automatic_case_id' in comparison_df.columns
        has_match_type = 'match_type' in comparison_df.columns
        if has_case_id and has_match_type:
            sig_mask = comparison_df['match_type'].astype(str).str.upper() == _normalize_status(significant_label)
            if 'absolute_difference' in comparison_df.columns:
                # Coerce to numeric for sorting
                absdiff = pd.to_numeric(comparison_df['absolute_difference'], errors='coerce')
            else:
                absdiff = pd.Series([np.nan] * len(comparison_df))
            df_tmp = comparison_df.assign(_absdiff=absdiff)
            df_sig = df_tmp.loc[sig_mask, ['automatic_case_id', '_absdiff']].dropna(subset=['automatic_case_id'])
            for cid, ad in zip(df_sig['automatic_case_id'].astype(str), df_sig['_absdiff']):
                significant_case_ids.add(cid)
                try:
                    case_abs_diff[str(cid)] = float(ad) if not pd.isna(ad) else float('inf')
                except (ValueError, TypeError):
                    case_abs_diff[str(cid)] = float('inf')

            # Build manual score map from comparison_df if available
            if 'manual_score' in comparison_df.columns:
                # Prefer the row with the lowest absolute_difference per case id
                df_ms = comparison_df.copy()
                df_ms['automatic_case_id'] = df_ms['automatic_case_id'].astype(str)
                if 'absolute_difference' in df_ms.columns and prefer_low_difference:
                    df_ms['_absdiff'] = pd.to_numeric(df_ms['absolute_difference'], errors='coerce')
                    df_ms = df_ms.sort_values(by=['automatic_case_id', '_absdiff'], na_position='last')
                else:
                    df_ms['_ord'] = range(len(df_ms))
                    df_ms = df_ms.sort_values(by=['automatic_case_id', '_ord'])
                # Take first occurrence per case id
                df_ms_first = df_ms.drop_duplicates(subset=['automatic_case_id'], keep='first')
                for cid, ms in zip(df_ms_first['automatic_case_id'], df_ms_first['manual_score']):
                    try:
                        manual_score_by_case_id[str(cid)] = float(ms)
                    except (ValueError, TypeError):
                        continue

    # Access publications and their cases by reference
    pubs_list = DataProcessor.standardize_auto_extractions(auto_extractions)

    def _find_cases_ref(pub: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(pub, dict):
            return None
        if 'cases' in pub:
            if isinstance(pub['cases'], list):
                return pub['cases']
            if isinstance(pub['cases'], dict) and 'individual_case_scores' in pub['cases'] and isinstance(pub['cases']['individual_case_scores'], list):
                return pub['cases']['individual_case_scores']
        if 'individual_case_scores' in pub and isinstance(pub['individual_case_scores'], list):
            return pub['individual_case_scores']
        if 'case_scores' in pub and isinstance(pub['case_scores'], list):
            return pub['case_scores']
        return None

    # Collect candidate references
    Candidate = Dict[str, Any]
    candidates: List[Candidate] = []
    for pub_idx, pub in enumerate(pubs_list):
        cases_ref = _find_cases_ref(pub)
        if not cases_ref:
            continue
        for case_idx, case in enumerate(cases_ref):
            if not isinstance(case, dict):
                continue
            current_status = _normalize_status(case.get(status_key))
            case_id = str(case.get('case_id', '')) if case.get('case_id') is not None else ''
            # A candidate if marked significant either on case or via comparison df
            is_significant = (
                current_status == _normalize_status(significant_label)
                or (case_id and case_id in significant_case_ids)
            )
            if not is_significant:
                continue
            candidates.append({
                'pub_idx': pub_idx,
                'case_idx': case_idx,
                'case_id': case_id,
                'absdiff': case_abs_diff.get(case_id, float('inf')),
                'prev_status': current_status,
                'pub_title': pub.get('title', ''),
                'pub_author': pub.get('author', ''),
                'pmid': pub.get('pmid', ''),
            })

    if not candidates:
        # Still return the flattened cases list for convenience
        pubs_list_all = DataProcessor.standardize_auto_extractions(auto_extractions)
        def _find_cases_ref_ro(pub: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
            if not isinstance(pub, dict):
                return None
            if 'cases' in pub:
                if isinstance(pub['cases'], list):
                    return pub['cases']
                if isinstance(pub['cases'], dict) and 'individual_case_scores' in pub['cases'] and isinstance(pub['cases']['individual_case_scores'], list):
                    return pub['cases']['individual_case_scores']
            if 'individual_case_scores' in pub and isinstance(pub['individual_case_scores'], list):
                return pub['individual_case_scores']
            if 'case_scores' in pub and isinstance(pub['case_scores'], list):
                return pub['case_scores']
            return None
        all_cases: List[Dict[str, Any]] = []
        for p in pubs_list_all:
            cref = _find_cases_ref_ro(p)
            if isinstance(cref, list):
                all_cases.extend(cref)
        return auto_extractions, all_cases, []

    # Prioritize by smallest absolute difference if requested and available; otherwise shuffle
    if prefer_low_difference and any(np.isfinite(c['absdiff']) for c in candidates):
        candidates.sort(key=lambda c: (not np.isfinite(c['absdiff']), c['absdiff']))
    else:
        rng.shuffle(candidates)

    # Apply the promotion in place
    transform_log: List[Dict[str, Any]] = []
    # To avoid repeated lookups, cache cases list per pub
    pub_cases_cache: Dict[int, List[Dict[str, Any]]] = {}

    # Build mapping (pub_idx, case_idx) -> manual_final_score using manual_df + matchers
    manual_score_by_pub_case: Dict[Tuple[int, int], float] = {}
    if manual_df is not None and isinstance(manual_df, pd.DataFrame) and not manual_df.empty:
        pub_matcher = PublicationMatcher(min_title_confidence=publication_match_threshold)
        case_matcher = CaseMatcher(require_gene_match=require_gene_match, min_confidence=case_match_threshold)

        for pub_idx, pub in enumerate(pubs_list):
            cases_ref = _find_cases_ref(pub)
            if not cases_ref:
                continue
            # Find best publication match
            try:
                pub_matches = pub_matcher.find_matches(manual_df, pub)
            except Exception:
                pub_matches = []
            if not pub_matches:
                continue
            best_pub = pub_matches[0]
            manual_row = manual_df.iloc[best_pub.manual_index]
            manual_pmid = PublicationParser.extract_pmid(manual_row.get('pmid', ''))
            if manual_pmid:
                manual_pub_rows = manual_df[manual_df['pmid'].apply(
                    lambda x: PublicationParser.extract_pmid(str(x))
                ) == manual_pmid]
            else:
                manual_pub_rows = manual_df[manual_df['publication'] == manual_row.get('publication', '')]

            # Match cases within this publication
            try:
                matches = case_matcher.find_matches(manual_pub_rows.to_dict('records'), cases_ref)
            except Exception:
                matches = []
            for m in matches:
                # Extract processed manual final score
                processed = case_matcher._extract_manual_case_data(manual_pub_rows.to_dict('records')[m.manual_index])
                mf = processed.get('final_score')
                if mf is None:
                    continue
                try:
                    manual_score_by_pub_case[(pub_idx, m.auto_index)] = float(mf)
                except (ValueError, TypeError):
                    continue

    transformed_count = 0
    for item in candidates:
        if transformed_count >= max(0, num_to_transform):
            break
        pub_idx = item['pub_idx']
        case_idx = item['case_idx']
        if pub_idx not in pub_cases_cache:
            pub_cases_cache[pub_idx] = _find_cases_ref(pubs_list[pub_idx]) or []
        cases_ref = pub_cases_cache[pub_idx]
        if not cases_ref or not (0 <= case_idx < len(cases_ref)):
            continue
        # Resolve manual score via pub+case map first, fallback to df-based case_id map
        manual_score_val: Optional[float] = manual_score_by_pub_case.get((pub_idx, case_idx))
        if manual_score_val is None and item['case_id']:
            manual_score_val = manual_score_by_case_id.get(item['case_id'])
        if manual_score_val is None:
            # Cannot update final_score without a manual score; skip this candidate
            continue
        case_ref = cases_ref[case_idx]
        prev = case_ref.get(status_key)
        case_ref[status_key] = exact_label
        case_ref['final_score'] = manual_score_val
        transform_log.append({
            'publication_index': pub_idx,
            'publication_title': pubs_list[pub_idx].get('title', ''),
            'publication_author': pubs_list[pub_idx].get('author', ''),
            'pmid': pubs_list[pub_idx].get('pmid', ''),
            'case_index': case_idx,
            'case_id': item['case_id'],
            'previous_status': prev,
            'new_status': exact_label,
            'absolute_difference': item['absdiff'] if np.isfinite(item['absdiff']) else None,
            'manual_score_used': manual_score_val,
        })
        transformed_count += 1

    # Build the flattened up-to-date list of all cases across publications
    updated_cases: List[Dict[str, Any]] = []
    for pub in pubs_list:
        cref = _find_cases_ref(pub)
        if isinstance(cref, list):
            updated_cases.extend(cref)

    return auto_extractions, updated_cases, transform_log


def convert_auto_cases_to_target_schema(
    cases: Any,
    title: Optional[str] = None,
    author: Optional[str] = None,
) -> Any:
    """
    Convert a list of automatically extracted cases (current_schema) to the
    target schema that resembles an NER-style extraction.

    Args:
        cases: Can be one of:
            - List of publication dicts each having keys like 'title', 'author', 'cases'
            - A single publication dict with key 'cases'
            - A flat list of case dicts (current_schema cases)
        title: Optional publication title (used when a flat list of cases is provided).
        author: Optional publication author (used when a flat list of cases is provided).

    Returns:
        If input is a list of publications: returns a list of converted publication dicts.
        Otherwise, returns a single converted publication dict with keys 'title', 'author', and 'cases',
        where cases are mapped into the target schema fields: ReportedCaseDetails,
        PhenotypingMethodsAndNotes, and ReportedVariantInformation.
    """

    # Auto-detect input shape
    if isinstance(cases, list):
        # If the list appears to be publications (objects with 'cases'), convert each publication
        if any(isinstance(item, dict) and 'cases' in item for item in cases):
            outputs: List[Dict[str, Any]] = []
            for pub in cases:
                if not isinstance(pub, dict):
                    continue
                pub_cases = pub.get('cases') or []
                outputs.append(
                    convert_auto_cases_to_target_schema(
                        pub_cases,
                        title=str(pub.get('title') or ''),
                        author=str(pub.get('author') or ''),
                    )
                )
            return outputs

    # Single publication dict with nested cases
    if isinstance(cases, dict) and 'cases' in cases:
        return convert_auto_cases_to_target_schema(
            cases.get('cases') or [],
            title=str(cases.get('title') or ''),
            author=str(cases.get('author') or ''),
        )

    def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
        value = d.get(key)
        return value if value is not None else default

    def _normalize_sex(value: Any) -> str:
        if value is None:
            return "Unknown"
        s = str(value).strip().lower()
        if s in {"m", "male"}:
            return "Male"
        if s in {"f", "female"}:
            return "Female"
        return s.capitalize() if s else "Unknown"

    def _format_age(value: Any) -> str:
        if value is None:
            return "Unknown"
        s = str(value).strip()
        return s if s else "Unknown"

    def _compose_phenotype(phenotypic: Dict[str, Any], description: Optional[str]) -> str:
        parts: List[str] = []
        phenos = _get(phenotypic, 'phenotypes')
        comorbid = _get(phenotypic, 'comorbidities')
        if phenos and isinstance(phenos, str):
            parts.append(phenos)
        if comorbid and isinstance(comorbid, str):
            parts.append(comorbid)
        if not parts and description:
            parts.append(str(description))
        return "; ".join([p for p in parts if p]) or "Unknown"

    def _compose_onset(phenotypic: Dict[str, Any], description: Optional[str]) -> str:
        milestones = _get(phenotypic, 'developmental_milestones')
        if milestones:
            return str(milestones)
        if description:
            return str(description)
        return "Unknown"

    def _compose_asd_note(phenotypic: Dict[str, Any]) -> str:
        tools = _get(phenotypic, 'assessment_tools_used')
        criteria = _get(phenotypic, 'diagnostic_criteria_used')
        if tools and criteria:
            return f"Diagnosed/assessed using {tools}; criteria: {criteria}"
        if tools:
            return f"Diagnosed/assessed using {tools}"
        if criteria:
            return f"Criteria: {criteria}"
        return "Unknown"

    def _compose_cognition(phenotypic: Dict[str, Any]) -> str:
        cog = _get(phenotypic, 'cognitive_assessment_results')
        if cog:
            return str(cog)
        comment = _get(phenotypic, 'cognitive_ability_cautionary_comment')
        return str(comment) if comment else "Unknown"

    def _inheritance_pretty(value: Any) -> str:
        if not value:
            return "Unknown"
        s = str(value).strip().lower().replace('_', ' ')
        if s == 'de novo' or s == 'de novo ':
            return 'De novo'
        return s.capitalize()

    def _compose_genotyping_method(variant: Dict[str, Any]) -> str:
        method = _get(variant, 'sequencing_method')
        details = _get(variant, 'sequencing_method_details')
        if method and details:
            return f"{method} ({details})"
        if method:
            return str(method)
        if details:
            return str(details)
        return "Unknown"

    def _compose_reference_sequence(variant: Dict[str, Any]) -> str:
        ref = _get(variant, 'reference_genome')
        txs = _get(variant, 'transcripts', [])
        if ref and txs and isinstance(txs, list) and txs:
            return f"{ref}; {txs[0]}"
        if ref:
            return str(ref)
        if isinstance(txs, list) and txs:
            return str(txs[0])
        return "Unknown"

    def _compose_reported_variant(variant: Dict[str, Any]) -> str:
        cdna = _get(variant, 'cdna_hgvs')
        prot = _get(variant, 'protein_hgvs')
        var = _get(variant, 'variant')
        txs = _get(variant, 'transcripts', [])
        tx = txs[0] if isinstance(txs, list) and txs else None
        if cdna and prot and tx:
            return f"{cdna} ({prot})"
        if cdna and prot:
            return f"{cdna} ({prot})"
        if cdna:
            return str(cdna)
        if var and prot:
            return f"{var} ({prot})"
        if var:
            return str(var)
        return "Unknown"

    def _compose_genomic_coords(variant: Dict[str, Any]) -> str:
        chrom = _get(variant, 'chromosome', '')
        start = _get(variant, 'start_position', None)
        end = _get(variant, 'end_position', None)
        try:
            if chrom and start is not None and end is not None and int(start) > 0 and int(end) > 0:
                return f"Chr{chrom}: {int(start):,}-{int(end):,}"
        except (ValueError, TypeError):
            pass
        return "Unknown"

    def _compose_impact(variant: Dict[str, Any]) -> str:
        impact = _get(variant, 'impact')
        if impact:
            return str(impact)
        desc = _get(variant, 'variant_description')
        return str(desc) if desc else "Unknown"

    def _compose_variant_details(variant: Dict[str, Any]) -> str:
        region = _get(variant, 'region')
        splice_info = _get(variant, 'splice_info')
        vt = _get(variant, 'variant_type')
        details: List[str] = []
        if region:
            details.append(str(region))
        if splice_info:
            details.append(str(splice_info))
        if vt:
            details.append(str(vt))
        return "; ".join(details) if details else "Unknown"

    def _compose_evidence_type(variant: Dict[str, Any]) -> str:
        category = str(_get(variant, 'category', '') or '').lower()
        inherit = str(_get(variant, 'inheritance_pattern', '') or '').lower()
        is_denovo = 'de' in inherit and 'novo' in inherit
        if 'cnv' in category or 'copy' in category or 'deletion' in category or 'duplication' in category:
            return 'De novo CNV' if is_denovo else 'CNV'
        if 'splice' in category:
            return 'De novo splice variant' if is_denovo else 'Splice variant'
        if 'single' in category or 'snv' in category or 'nucleotide' in category or 'missense' in category:
            return 'De novo SNV' if is_denovo else 'SNV'
        if 'indel' in category or 'insertion' in category or 'deletion' in category:
            return 'De novo indel' if is_denovo else 'Indel'
        return 'De novo' if is_denovo else 'Genetic variant'

    def _compose_functional_outcome(variant: Dict[str, Any]) -> str:
        func = _get(variant, 'functional_data')
        if func:
            return str(func)
        linked = _get(variant, 'linkage_to_asd')
        if isinstance(linked, bool):
            return 'Linked to ASD in study' if linked else 'Not linked to ASD'
        return 'Unknown'

    def _intellectual_disability_status(phenotypic: Dict[str, Any]) -> str:
        text = " ".join(str(_get(phenotypic, k, '')) for k in ['phenotypes', 'comorbidities']).lower()
        if any(token in text for token in ['intellectual disability', 'developmental delay', 'idd']):
            return 'Present'
        return 'Unknown'

    converted_cases: List[Dict[str, Any]] = []
    # When a flat list of publications was passed at the top, we only reach here for a single pub
    # or a flat list of cases. Here, 'cases' should be a flat list of case dicts.
    for case in (cases or []):
        if not isinstance(case, dict):
            continue
        phenotypic = _get(case, 'phenotypic_evidence', {}) or {}
        variant = _get(case, 'variant', {}) or {}

        gene = _get(case, 'gene_symbol', 'Unknown')
        case_id = _get(case, 'case_id', 'Unknown')
        sex = _normalize_sex(_get(case, 'sex'))
        age = _format_age(_get(case, 'age'))
        severity = _get(phenotypic, 'ia_category', 'Unknown')
        onset = _compose_onset(phenotypic, _get(case, 'description'))
        phenotype = _compose_phenotype(phenotypic, _get(case, 'description'))
        family_history = _get(case, 'family_history', 'Unknown')

        asd_note = _compose_asd_note(phenotypic)
        cognition = _compose_cognition(phenotypic)

        genotyping_method = _compose_genotyping_method(variant)
        reference_sequence = _compose_reference_sequence(variant)
        reported_variant = _compose_reported_variant(variant)
        genomic_coords = _compose_genomic_coords(variant)
        impact = _compose_impact(variant)
        inheritance = _inheritance_pretty(_get(variant, 'inheritance_pattern'))
        functional_outcome = _compose_functional_outcome(variant)
        previous_research = "Linked to ASD in this study" if bool(_get(variant, 'linkage_to_asd')) else "Unknown"
        variant_details = _compose_variant_details(variant)
        note = _get(case, 'notes', None)
        evidence_type = _compose_evidence_type(variant)
        id_status = _intellectual_disability_status(phenotypic)

        converted_cases.append({
            "ReportedCaseDetails": {
                "Gene": gene if gene else "Unknown",
                "ID": case_id if case_id else "Unknown",
                "Sex": sex,
                "Age": age,
                "Severity": severity if severity else "Unknown",
                "Onset": onset,
                "Phenotype": phenotype,
                "FamilyHistory": family_history if family_history else "Unknown",
            },
            "PhenotypingMethodsAndNotes": {
                "ASD": asd_note,
                "Cognition": cognition,
            },
            "ReportedVariantInformation": {
                "GenotypingMethod": genotyping_method,
                "ReferenceSequence": reference_sequence,
                "ReportedVariant": reported_variant,
                "GenomicCoords": genomic_coords,
                "Impact": impact,
                "Inheritance": inheritance,
                "FunctionalStudyOutcome": functional_outcome,
                "PreviousResearch": previous_research,
                "VariantDetails": variant_details,
                "Note": note if note else "Unknown",
                "EvidenceType": evidence_type,
                "IntellectualDisability": id_status,
            }
        })

    return {
        "title": title or "",
        "author": author or "",
        "cases": converted_cases,
    }


def convert_target_cases_to_current_schema(
    data: Any,
    default_title: Optional[str] = None,
    default_author: Optional[str] = None,
) -> Any:
    """
    Convert cases from the NER-like target schema back to the current schema.

    Input can be:
      - List of publications (each with 'title', 'author', 'cases' where cases follow target schema)
      - Single publication dict with 'cases' (target schema)
      - Flat list of target-schema case items

    Returns:
      - If input is a list of publications: returns a list of current-schema publications
      - If input is a single publication: returns one current-schema publication
      - If input is a flat list of cases: returns a flat list of current-schema cases
    """

    def _s(val: Any, default: str = "") -> str:
        return str(val).strip() if val is not None else default

    def _lower(val: Any) -> str:
        s = _s(val)
        return s.lower() if s else ""

    def _sex_to_current(val: Any) -> str:
        s = _lower(val)
        if s in {"male", "m"}:
            return "male"
        if s in {"female", "f"}:
            return "female"
        return "unknown" if s else ""

    def _inheritance_to_current(val: Any) -> str:
        s = _lower(val).replace(" ", "_")
        mapping = {
            "de_novo": "de_novo",
            "denovo": "de_novo",
            "maternal": "maternal",
            "paternal": "paternal",
            "inherited": "inherited",
        }
        return mapping.get(s, s)

    def _parse_genotyping_method(val: Any) -> Tuple[Optional[str], Optional[str]]:
        text = _s(val)
        if not text:
            return None, None
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", text)
        if m:
            return (m.group(1) or None, m.group(2) or None)
        return text, None

    def _parse_reference_sequence(val: Any) -> Tuple[Optional[str], Optional[List[str]]]:
        text = _s(val)
        if not text:
            return None, None
        parts = [p.strip() for p in text.split(';') if p.strip()]
        ref = None
        tx: List[str] = []
        for p in parts:
            if p.upper().startswith("GRCH") or p.lower().startswith("hg"):
                ref = p
            elif re.search(r"\b[NM]P?_[0-9.]+", p):
                tx.append(p)
        if not parts:
            if re.search(r"\b[NM]P?_[0-9.]+", text):
                tx.append(text)
            else:
                ref = text
        return (ref if ref else None), (tx if tx else None)

    def _parse_reported_variant(val: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        # returns (cdna_hgvs, protein_hgvs, variant_raw)
        text = _s(val)
        if not text:
            return None, None, None
        m = re.search(r"(?:(?:[NMR]M?_[0-9.]+):)?(c\.[^\s()]+)(?:\s*\(([^)]+)\))?", text)
        if m:
            cdna = m.group(1)
            prot = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            return cdna, prot, text
        if text.startswith('c.'):
            return text, None, text
        return None, None, text

    def _parse_coords(val: Any) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        text = _s(val)
        if not text:
            return None, None, None
        m = re.search(r"Chr\s*([A-Za-z0-9]+)\s*:\s*([0-9,]+)\s*[-–]\s*([0-9,]+)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"Chr\s*([A-Za-z0-9]+)\s*:\s*([0-9,]+)", text, re.IGNORECASE)
        if m:
            chrom = m.group(1)
            try:
                start = int(m.group(2).replace(',', '')) if m.lastindex and m.lastindex >= 2 else None
            except Exception:
                start = None
            try:
                end = int(m.group(3).replace(',', '')) if m.lastindex and m.lastindex >= 3 else None
            except Exception:
                end = None
            return chrom, start, end
        return None, None, None

    def _category_from_evidence_type(val: Any) -> Optional[str]:
        s = _lower(val)
        if not s:
            return None
        if 'cnv' in s or 'copy' in s:
            return 'copy_number_variant'
        if 'splice' in s:
            return 'splice_variant'
        if 'indel' in s:
            return 'indel'
        if 'snv' in s or 'single' in s or 'nucleotide' in s:
            return 'single_nucleotide_polymorphism'
        return None

    def _bool_linkage_from_text(val: Any) -> Optional[bool]:
        s = _lower(val)
        if not s:
            return None
        if 'asd' in s:
            return True
        return None

    def _compose_description(onset: str, phenotype: str, severity: str) -> str:
        parts: List[str] = []
        if onset:
            parts.append(f"Onset: {onset}")
        if phenotype:
            parts.append(f"Phenotype: {phenotype}")
        if severity and severity.lower() != 'unknown':
            parts.append(f"Severity: {severity}")
        return "; ".join(parts)

    def _convert_cases_list(cases_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for item in (cases_list or []):
            if not isinstance(item, dict):
                continue
            rcd = item.get('ReportedCaseDetails', {}) or {}
            pmn = item.get('PhenotypingMethodsAndNotes', {}) or {}
            rvi = item.get('ReportedVariantInformation', {}) or {}

            gene = _s(rcd.get('Gene'), '')
            case_id = _s(rcd.get('ID'), '')
            sex = _sex_to_current(rcd.get('Sex'))
            age = _s(rcd.get('Age'), '')
            severity = _s(rcd.get('Severity'), '')
            onset = _s(rcd.get('Onset'), '')
            phenotype = _s(rcd.get('Phenotype'), '')
            fam_hist = _s(rcd.get('FamilyHistory'), '')

            asd_text = _s(pmn.get('ASD'), '')
            cognition = _s(pmn.get('Cognition'), '')

            genotyping = _s(rvi.get('GenotypingMethod'), '')
            refseq = _s(rvi.get('ReferenceSequence'), '')
            reported_variant = _s(rvi.get('ReportedVariant'), '')
            coords = _s(rvi.get('GenomicCoords'), '')
            impact = _s(rvi.get('Impact'), '')
            inheritance = _s(rvi.get('Inheritance'), '')
            func_outcome = _s(rvi.get('FunctionalStudyOutcome'), '')
            prev_research = _s(rvi.get('PreviousResearch'), '')
            variant_details = _s(rvi.get('VariantDetails'), '')
            note = _s(rvi.get('Note'), '')
            evidence_type = _s(rvi.get('EvidenceType'), '')
            id_status = _s(rvi.get('IntellectualDisability'), '')

            seq_method, seq_details = _parse_genotyping_method(genotyping)
            ref_genome, txs = _parse_reference_sequence(refseq)
            cdna_hgvs, protein_hgvs, variant_raw = _parse_reported_variant(reported_variant)
            chrom, start_pos, end_pos = _parse_coords(coords)

            current_case: Dict[str, Any] = {
                'case_id': case_id or 'Unknown',
                'gene_symbol': gene or 'Unknown',
                'description': _compose_description(onset, phenotype, severity),
                'cohort_name': None,
                'age': age or None,
                'sex': sex or None,
                'ethnicity': None,
                'family_history': fam_hist or None,
                'notes': note or None,
                'quotes': None,
                'phenotypic_evidence': {
                    'age_at_diagnosis': None,
                    'core_asd_symptoms': None,
                    'diagnostic_criteria_used': None,
                    'assessment_tools_used': asd_text or None,
                    'phenotype_source_indicator_a': False,
                    'phenotype_source_indicator_b': False,
                    'phenotype_source_indicator_c': False,
                    'phenotype_source_indicator_d': False,
                    'phenotype_source_indicator_e': False,
                    'phenotype_source_indicator_f': False,
                    'phenotype_confidence': None,
                    'cognitive_assessment_results': None,
                    'ia_category': severity or None,
                    'cognitive_ability_cautionary_comment': cognition or None,
                    'developmental_milestones': onset or None,
                    'comorbidities': ('intellectual disability' if id_status.lower() == 'present' else None),
                    'phenotypes': phenotype or None,
                },
                'variant': {
                    'sequencing_method': seq_method,
                    'sequencing_method_details': seq_details,
                    'reference_genome': ref_genome,
                    'chromosome': chrom or '',
                    'start_position': int(start_pos) if isinstance(start_pos, int) else 0,
                    'end_position': int(end_pos) if isinstance(end_pos, int) else 0,
                    'reference_allele': '',
                    'alternate_allele': '',
                    'region': variant_details if (variant_details and 'splice' not in variant_details.lower()) else (
                        'splice acceptor' if 'acceptor' in variant_details.lower() else (
                            'splice donor' if 'donor' in variant_details.lower() else None
                        )
                    ) if variant_details else None,
                    'transcripts': txs or None,
                    'variant': cdna_hgvs or variant_raw or None,
                    'genomic_hgvs': None,
                    'cdna_hgvs': cdna_hgvs or None,
                    'protein_hgvs': protein_hgvs or None,
                    'variant_description': variant_details or None,
                    'rsid': None,
                    'variant_type': None,
                    'category': _category_from_evidence_type(evidence_type) or None,
                    'impact': impact or None,
                    'splice_info': (variant_details if variant_details and 'splice' in variant_details.lower() else None),
                    'splice_type': None,
                    'inheritance_pattern': _inheritance_to_current(inheritance) or None,
                    'segregation_data': None,
                    'zygosity': None,
                    'population_frequency': None,
                    'in_gnomad': None,
                    'gnomad_frequency': None,
                    'quality_score': None,
                    'functional_data': func_outcome or None,
                    'linkage_to_asd': _bool_linkage_from_text(prev_research),
                    'clinical_significance': None,
                    'variant_status': None,
                    'phenotype_confidence': None,
                },
                'final_score': 0.0,
                'initial_genetic_score': 0.0,
                'score_rationale': '',
                'functional_data_adjustment': 0,
                'phenotype_confidence_adjustment': 0,
            }

            out.append(current_case)
        return out

    # If it's a list, determine whether it's a list of publications or a flat list of cases
    if isinstance(data, list):
        if any(isinstance(item, dict) and 'cases' in item for item in data):
            publications_out: List[Dict[str, Any]] = []
            for pub in data:
                if not isinstance(pub, dict):
                    continue
                pub_cases = pub.get('cases') or []
                publications_out.append({
                    'title': _s(pub.get('title'), default_title or ''),
                    'author': _s(pub.get('author'), default_author or ''),
                    'pmid': _s(pub.get('pmid'), ''),
                    'cases': _convert_cases_list(pub_cases),
                })
            return publications_out
        # Flat list of target cases
        return _convert_cases_list(data)

    # Single publication dict
    if isinstance(data, dict) and 'cases' in data:
        return {
            'title': _s(data.get('title'), default_title or ''),
            'author': _s(data.get('author'), default_author or ''),
            'pmid': _s(data.get('pmid'), ''),
            'cases': _convert_cases_list(data.get('cases') or []),
        }

    # Unknown shape; return empty compatible structure
    return {
        'title': _s(default_title, ''),
        'author': _s(default_author, ''),
        'pmid': '',
        'cases': [],
    }


# ==============================================================================
# F1 Evaluation Utilities (phenotype, variants, and case matching)
# ==============================================================================

import ast  # Used for parsing list-like strings safely


class PhenotypeConfidenceParser:
    """Parse and normalize phenotype confidence labels per EAGLE guidelines.

    Mapping rules derived from the EAGLE guideline source indicators A-F:
    - High confidence: any of A, B, or C is True
    - Medium confidence: D is True (and none of A/B/C)
    - Low confidence: E or F is True (and none of A/B/C/D)
    If indicators are unavailable, will attempt to use string field 'phenotype_confidence'.
    """

    @staticmethod
    def _canonicalize(label: str) -> str:
        if not label:
            return 'unknown'
        s = str(label).strip().lower()
        if s.startswith('high'):
            return 'high'
        if s.startswith('medium'):
            return 'medium'
        if s.startswith('low'):
            return 'low'
        return 'unknown'

    @staticmethod
    def parse_from_auto(phenotypic_evidence: Optional[Dict[str, Any]]) -> str:
        if not isinstance(phenotypic_evidence, dict):
            return 'unknown'
        # Prefer indicators if present
        a = bool(phenotypic_evidence.get('phenotype_source_indicator_a'))
        b = bool(phenotypic_evidence.get('phenotype_source_indicator_b'))
        c = bool(phenotypic_evidence.get('phenotype_source_indicator_c'))
        d = bool(phenotypic_evidence.get('phenotype_source_indicator_d'))
        e = bool(phenotypic_evidence.get('phenotype_source_indicator_e'))
        f = bool(phenotypic_evidence.get('phenotype_source_indicator_f'))

        if a or b or c:
            return 'high'
        if d:
            return 'medium'
        if e or f:
            return 'low'

        # Fallback: direct field
        return PhenotypeConfidenceParser._canonicalize(
            str(phenotypic_evidence.get('phenotype_confidence') or '')
        )

    @staticmethod
    def parse_from_manual(value: Any) -> str:
        return PhenotypeConfidenceParser._canonicalize(str(value or ''))


class VariantNormalizer:
    """Normalize genomic, cDNA (coding), and protein HGVS-like strings for comparison.

    These normalizers are intentionally forgiving and target consistent string
    comparison rather than full HGVS compliance.
    """

    @staticmethod
    def _clean_list_like(value: Any) -> List[str]:
        """Parse value that may be a list, list-like string, or comma-separated string."""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value if v is not None]
        s = str(value).strip()
        if not s:
            return []
        # Try literal_eval for strings like "['c.1', 'c.2']"
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(v) for v in parsed if v is not None]
        except Exception:
            pass
        # Fallback: split by comma
        return [t.strip() for t in s.split(',') if t.strip()]

    @staticmethod
    def _strip_wrappers(text: str) -> str:
        return (text or '').replace('[', '').replace(']', '').replace('(', '').replace(')', '')

    @staticmethod
    def normalize_genomic(hgvs: str) -> str:
        if not hgvs:
            return ''
        s = str(hgvs)
        s = VariantNormalizer._strip_wrappers(s)
        s = s.replace(' ', '')
        s = s.replace('Chr', 'chr').replace('CHR', 'chr')
        s_lower = s.lower()

        # Extract g.-prefixed substring if present
        pos = s_lower.find('g.')
        if pos >= 0:
            core = s[pos:]
        else:
            core = s
        # Drop reference genome annotations like (hg19) or :hg19
        core = re.sub(r'\(hg\d+\)|hg\d+|\(grch\d+\)|grch\d+', '', core, flags=re.IGNORECASE)
        # Keep only a subset of allowed characters; uppercase nucleotides
        core = core.replace('chr', '')
        core = core.replace(':', '')
        # Standardize case: keep 'g.' lowercase and nucleotides upper
        if core.lower().startswith('g.'):
            prefix = 'g.'
            rest = core[2:]
        else:
            prefix = ''
            rest = core
        # Uppercase letters in rest, keep punctuation
        rest = re.sub(r'[a-z]', lambda m: m.group(0).upper(), rest)
        return f"{prefix}{rest}"

    @staticmethod
    def normalize_cdna(cdna: str) -> str:
        if not cdna:
            return ''
        s = str(cdna)
        s = VariantNormalizer._strip_wrappers(s)
        s = s.replace(' ', '')
        if not s.lower().startswith('c.'):
            # Try to extract c. segment
            m = re.search(r'(c\.[^;\s,]+)', s, flags=re.IGNORECASE)
            if m:
                s = m.group(1)
        # Enforce c. prefix and uppercase rest
        if s.lower().startswith('c.'):
            prefix = 'c.'
            rest = s[2:]
        else:
            prefix = ''
            rest = s
        rest = re.sub(r'[a-z]', lambda m: m.group(0).upper(), rest)
        return f"{prefix}{rest}"

    @staticmethod
    def normalize_protein(protein: str) -> str:
        if not protein:
            return ''
        s = str(protein)
        s = VariantNormalizer._strip_wrappers(s)
        s = s.replace(' ', '')
        # Extract p. segment if present
        if not s.lower().startswith('p.'):
            m = re.search(r'(p\.[^;\s,]+)', s, flags=re.IGNORECASE)
            if m:
                s = m.group(1)
        # Enforce uppercase variant portion
        if s.lower().startswith('p.'):
            prefix = 'p.'
            rest = s[2:]
        else:
            prefix = ''
            rest = s
        rest = re.sub(r'[a-z]', lambda m: m.group(0).upper(), rest)
        return f"{prefix}{rest}"

    @staticmethod
    def extract_manual_variants(manual_row: Dict[str, Any]) -> Dict[str, set]:
        """Return sets of normalized variants from a manual dataframe row.

        Looks for several common column names for genomic, cDNA, and protein variants.
        """
        # Candidate column names
        genomic_cols = ['genomic_variants', 'genomic_variant', 'genomic_hgvs', 'genomic']
        coding_cols = ['coding_variants', 'coding_variant', 'cdna_hgvs', 'coding_hgvs']
        protein_cols = ['protein_variants', 'protein_variant', 'protein_hgvs']

        def _first_present(cols: List[str]) -> Optional[str]:
            for c in cols:
                if c in manual_row and manual_row.get(c) not in (None, ''):
                    return c
            return None

        g_col = _first_present(genomic_cols)
        c_col = _first_present(coding_cols)
        p_col = _first_present(protein_cols)

        genomic_values = VariantNormalizer._clean_list_like(manual_row.get(g_col)) if g_col else []
        coding_values = VariantNormalizer._clean_list_like(manual_row.get(c_col)) if c_col else []
        protein_values = VariantNormalizer._clean_list_like(manual_row.get(p_col)) if p_col else []

        # Fallback: attempt to parse from free-text 'description' or similar
        if not genomic_values:
            for free_col in ['description', 'notes', 'variant_notes']:
                free_text = manual_row.get(free_col)
                if free_text:
                    # Extract g. segments
                    genomic_values.extend(re.findall(r'(g\.[^\s\]\)\;\,\|]+)', str(free_text)))
                    if genomic_values:
                        break

        genomic_set = {VariantNormalizer.normalize_genomic(v) for v in genomic_values if VariantNormalizer.normalize_genomic(v)}
        coding_set = {VariantNormalizer.normalize_cdna(v) for v in coding_values if VariantNormalizer.normalize_cdna(v)}
        protein_set = {VariantNormalizer.normalize_protein(v) for v in protein_values if VariantNormalizer.normalize_protein(v)}

        return {
            'genomic': genomic_set,
            'coding': coding_set,
            'protein': protein_set,
        }

    @staticmethod
    def extract_auto_variants(auto_case: Dict[str, Any]) -> Dict[str, set]:
        variant = auto_case.get('variant', {}) if isinstance(auto_case, dict) else {}
        if not isinstance(variant, dict):
            variant = {}
        genomic_values: List[str] = []
        coding_values: List[str] = []
        protein_values: List[str] = []

        # Preferred explicit fields
        if variant.get('genomic_hgvs'):
            genomic_values.append(str(variant.get('genomic_hgvs')))
        if variant.get('cdna_hgvs'):
            coding_values.append(str(variant.get('cdna_hgvs')))
        if variant.get('protein_hgvs'):
            protein_values.append(str(variant.get('protein_hgvs')))

        # Fallback: generic 'variant' field may contain c. or g.
        vfield = variant.get('variant')
        if vfield:
            s = str(vfield)
            if 'c.' in s or s.lower().startswith('c.'):
                coding_values.append(s)
            if 'g.' in s or s.lower().startswith('g.'):
                genomic_values.append(s)

        genomic_set = {VariantNormalizer.normalize_genomic(v) for v in genomic_values if VariantNormalizer.normalize_genomic(v)}
        coding_set = {VariantNormalizer.normalize_cdna(v) for v in coding_values if VariantNormalizer.normalize_cdna(v)}
        protein_set = {VariantNormalizer.normalize_protein(v) for v in protein_values if VariantNormalizer.normalize_protein(v)}

        return {
            'genomic': genomic_set,
            'coding': coding_set,
            'protein': protein_set,
        }


class F1Calculator:
    """Helper for precision/recall/F1 computations (micro and macro)."""

    @staticmethod
    def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}

    @staticmethod
    def multiclass_prf(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
        counts = {lbl: {'tp': 0, 'fp': 0, 'fn': 0} for lbl in labels}
        for t, p in zip(y_true, y_pred):
            if t == p and t in labels:
                counts[t]['tp'] += 1
            else:
                if p in labels:
                    counts[p]['fp'] += 1
                if t in labels:
                    counts[t]['fn'] += 1
        per_label = {}
        macro_f1s = []
        total_tp = total_fp = total_fn = 0
        for lbl in labels:
            c = counts[lbl]
            metrics = F1Calculator.prf_from_counts(c['tp'], c['fp'], c['fn'])
            per_label[lbl] = metrics
            macro_f1s.append(metrics['f1'])
            total_tp += c['tp']; total_fp += c['fp']; total_fn += c['fn']
        macro = sum(macro_f1s)/len(macro_f1s) if macro_f1s else 0.0
        micro = F1Calculator.prf_from_counts(total_tp, total_fp, total_fn)
        return {'per_label': per_label, 'macro_f1': macro, 'micro': micro}


class F1Evaluator:
    """Compute F1 metrics between manual and automatic curations and report failures."""

    def __init__(self,
                 require_gene_match: bool = True,
                 publication_match_threshold: float = 0.8,
                 case_match_min_confidence: float = 0.3):
        self.require_gene_match = require_gene_match
        self.publication_match_threshold = publication_match_threshold
        self.case_match_min_confidence = case_match_min_confidence
        self.pub_matcher = PublicationMatcher(min_title_confidence=publication_match_threshold)
        # We'll reuse CaseMatcher's internal scoring but not its find_matches filtering
        self.case_matcher = CaseMatcher(require_gene_match=require_gene_match,
                                        min_confidence=case_match_min_confidence)

    def _manual_pub_key(self, row: Dict[str, Any]) -> str:
        pmid = PublicationParser.extract_pmid(row.get('pmid', ''))
        if pmid:
            return f"pmid:{pmid}"
        return f"pub:{str(row.get('publication') or '')}"

    def _collect_manual_pub_rows(self, manual_df: pd.DataFrame, manual_index: int) -> pd.DataFrame:
        manual_row = manual_df.iloc[manual_index]
        manual_pmid = PublicationParser.extract_pmid(manual_row.get('pmid', ''))
        if manual_pmid:
            return manual_df[manual_df['pmid'].apply(lambda x: PublicationParser.extract_pmid(str(x))) == manual_pmid]
        return manual_df[manual_df['publication'] == manual_row.get('publication', '')]

    def _extract_manual_case_core(self, manual_row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'gene': str(manual_row.get('gene', '')).strip().upper(),
            'case_id': TextNormalizer.normalize_case_id(str(manual_row.get('id', ''))),
            'inheritance': str(manual_row.get('inheritance', '')).lower().strip(),
        }

    def _extract_auto_case_core(self, auto_case: Dict[str, Any]) -> Dict[str, Any]:
        variant_info = auto_case.get('variant_info', {}) if isinstance(auto_case, dict) else {}
        return {
            'gene': str(auto_case.get('gene_symbol', '')).strip().upper(),
            'case_id': TextNormalizer.normalize_case_id(str(auto_case.get('case_id', ''))),
            'inheritance': str(variant_info.get('inheritance_pattern', '')).lower().strip(),
        }

    def _score_case_pair(self, manual_row: Dict[str, Any], auto_case: Dict[str, Any]) -> float:
        # Reuse CaseMatcher scoring to leverage robust heuristics
        manual_detailed = self.case_matcher._extract_manual_case_data(manual_row)
        auto_detailed = self.case_matcher._extract_auto_case_data(auto_case)
        result = self.case_matcher._calculate_case_match(manual_detailed, auto_detailed, 0, 0)
        return result.confidence if result else 0.0

    def _greedy_match_cases(self, manual_rows: List[Dict[str, Any]], auto_cases: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """Return (matches, unmatched_manual_indices, unmatched_auto_indices) with greedy best-score matching.

        - matches: list of (manual_idx, auto_idx, score) indices relative to provided lists
        """
        scores: List[Tuple[float, int, int]] = []  # (score, m_idx, a_idx)
        for m_idx, m_row in enumerate(manual_rows):
            for a_idx, a_case in enumerate(auto_cases):
                score = self._score_case_pair(m_row, a_case)
                if score >= self.case_match_min_confidence:
                    scores.append((score, m_idx, a_idx))
        # Sort descending by score, then greedily take non-conflicting pairs
        scores.sort(key=lambda t: t[0], reverse=True)
        used_m: set = set()
        used_a: set = set()
        pairs: List[Tuple[int, int, float]] = []
        for score, m_idx, a_idx in scores:
            if m_idx in used_m or a_idx in used_a:
                continue
            used_m.add(m_idx)
            used_a.add(a_idx)
            pairs.append((m_idx, a_idx, score))
        unmatched_m = [i for i in range(len(manual_rows)) if i not in used_m]
        unmatched_a = [i for i in range(len(auto_cases)) if i not in used_a]
        return pairs, unmatched_m, unmatched_a

    def _get_auto_cases_ref(self, pub: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(pub, dict):
            return []
        if 'cases' in pub:
            if isinstance(pub['cases'], list):
                return pub['cases']
            if isinstance(pub['cases'], dict) and 'individual_case_scores' in pub['cases'] and isinstance(pub['cases']['individual_case_scores'], list):
                return pub['cases']['individual_case_scores']
        if 'individual_case_scores' in pub and isinstance(pub['individual_case_scores'], list):
            return pub['individual_case_scores']
        if 'case_scores' in pub and isinstance(pub['case_scores'], list):
            return pub['case_scores']
        return []

    def evaluate(self, manual_df: pd.DataFrame, auto_extractions: Any) -> Dict[str, Any]:
        # Validate minimal manual columns
        required_cols = ['publication', 'pmid', 'gene', 'id']
        missing = [c for c in required_cols if c not in manual_df.columns]
        if missing:
            raise ValueError(f"Manual dataset missing columns for F1 evaluation: {missing}")

        pubs_list = DataProcessor.standardize_auto_extractions(auto_extractions)

        # Track global counts
        case_tp = case_fp = case_fn = 0
        phenotype_true: List[str] = []
        phenotype_pred: List[str] = []
        variant_tp = {'genomic': 0, 'coding': 0, 'protein': 0, 'all': 0}
        variant_fp = {'genomic': 0, 'coding': 0, 'protein': 0, 'all': 0}
        variant_fn = {'genomic': 0, 'coding': 0, 'protein': 0, 'all': 0}

        # Failure logs
        failures: Dict[str, List[Dict[str, Any]]] = {
            'unmatched_publications_auto': [],
            'unmatched_publications_manual': [],
            'unmatched_cases_auto': [],
            'unmatched_cases_manual': [],
            'phenotype_mismatches': [],
            'variant_mismatches': [],
        }

        matched_manual_pub_indices: set = set()

        for pub_idx, pub in enumerate(pubs_list):
            try:
                pub_matches = self.pub_matcher.find_matches(manual_df, pub)
            except Exception:
                pub_matches = []
            if not pub_matches:
                # Entire publication unmatched → all cases are false positives
                auto_cases = self._get_auto_cases_ref(pub)
                case_fp += len(auto_cases)
                for a_idx, case in enumerate(auto_cases):
                    failures['unmatched_cases_auto'].append({
                        'publication_index': pub_idx,
                        'auto_case_index': a_idx,
                        'case_id': case.get('case_id', ''),
                        'gene': case.get('gene_symbol', ''),
                        'reason': 'Publication not matched in manual dataset'
                    })
                failures['unmatched_publications_auto'].append({
                    'publication_index': pub_idx,
                    'title': pub.get('title', ''),
                    'author': pub.get('author', ''),
                    'pmid': pub.get('pmid', ''),
                })
                continue

            best_pub = pub_matches[0]
            matched_manual_pub_indices.add(best_pub.manual_index)
            manual_pub_rows = self._collect_manual_pub_rows(manual_df, best_pub.manual_index)
            auto_cases = self._get_auto_cases_ref(pub)

            # Perform greedy matching of cases inside this publication
            matches, unmatched_m, unmatched_a = self._greedy_match_cases(manual_pub_rows.to_dict('records'), auto_cases)

            # Case-level TP/FP/FN
            case_tp += len(matches)
            case_fn += len(unmatched_m)
            case_fp += len(unmatched_a)

            # Log unmatched cases
            for m_local_idx in unmatched_m:
                mrow = manual_pub_rows.iloc[m_local_idx]
                failures['unmatched_cases_manual'].append({
                    'publication_index': pub_idx,
                    'manual_case_id': mrow.get('id', ''),
                    'gene': mrow.get('gene', ''),
                    'reason': 'No matching automatic case found'
                })
            for a_local_idx in unmatched_a:
                acase = auto_cases[a_local_idx]
                failures['unmatched_cases_auto'].append({
                    'publication_index': pub_idx,
                    'auto_case_index': a_local_idx,
                    'case_id': acase.get('case_id', ''),
                    'gene': acase.get('gene_symbol', ''),
                    'reason': 'No matching manual case found'
                })

            # Phenotype and variants for matched pairs
            for m_local_idx, a_local_idx, score in matches:
                mrow = manual_pub_rows.iloc[m_local_idx]
                acase = auto_cases[a_local_idx]

                # Phenotype confidence
                manual_label = PhenotypeConfidenceParser.parse_from_manual(mrow.get('phenotype_quality', ''))
                auto_label = PhenotypeConfidenceParser.parse_from_auto(acase.get('phenotypic_evidence'))
                if manual_label != 'unknown' or auto_label != 'unknown':
                    phenotype_true.append(manual_label)
                    phenotype_pred.append(auto_label)
                    if manual_label != auto_label:
                        failures['phenotype_mismatches'].append({
                            'publication_index': pub_idx,
                            'manual_case_id': mrow.get('id', ''),
                            'auto_case_id': acase.get('case_id', ''),
                            'manual_label': manual_label,
                            'auto_label': auto_label,
                        })

                # Variants
                mv = VariantNormalizer.extract_manual_variants(mrow.to_dict())
                av = VariantNormalizer.extract_auto_variants(acase)

                for key in ['genomic', 'coding', 'protein']:
                    inter = mv[key].intersection(av[key])
                    only_m = mv[key] - av[key]
                    only_a = av[key] - mv[key]
                    variant_tp[key] += len(inter)
                    variant_fn[key] += len(only_m)
                    variant_fp[key] += len(only_a)
                    if only_m or only_a:
                        failures['variant_mismatches'].append({
                            'publication_index': pub_idx,
                            'manual_case_id': mrow.get('id', ''),
                            'auto_case_id': acase.get('case_id', ''),
                            'type': key,
                            'missing_in_auto': sorted(list(only_m)),
                            'extra_in_auto': sorted(list(only_a)),
                        })

                # Combined variant tokens for overall metric
                m_all = mv['genomic'].union(mv['coding']).union(mv['protein'])
                a_all = av['genomic'].union(av['coding']).union(av['protein'])
                inter_all = m_all.intersection(a_all)
                variant_tp['all'] += len(inter_all)
                variant_fn['all'] += len(m_all - a_all)
                variant_fp['all'] += len(a_all - m_all)

        # Manual publications never matched by any auto publication
        if len(matched_manual_pub_indices) < len(manual_df):
            # Identify unmatched publications by key
            matched_keys = set()
            for idx in matched_manual_pub_indices:
                try:
                    matched_keys.add(self._manual_pub_key(manual_df.iloc[idx].to_dict()))
                except Exception:
                    continue
            # Consider all unique pubs
            manual_df['_pub_key_tmp'] = manual_df.apply(lambda r: self._manual_pub_key(r), axis=1)
            unmatched_pub_keys = set(manual_df['_pub_key_tmp'].unique()) - matched_keys
            for key in unmatched_pub_keys:
                sub = manual_df[manual_df['_pub_key_tmp'] == key]
                # All their cases are false negatives
                case_fn += len(sub)
                failures['unmatched_publications_manual'].append({
                    'publication_key': key,
                    'num_cases': len(sub),
                })
            # Drop temp column
            manual_df.drop(columns=['_pub_key_tmp'], inplace=True)

        # Build metrics
        cases_metrics = F1Calculator.prf_from_counts(case_tp, case_fp, case_fn)

        # Phenotype metrics (multiclass)
        phenotype_metrics: Dict[str, Any]
        if phenotype_true and phenotype_pred:
            labels = ['high', 'medium', 'low']
            phenotype_metrics = F1Calculator.multiclass_prf(phenotype_true, phenotype_pred, labels)
        else:
            phenotype_metrics = {'per_label': {}, 'macro_f1': 0.0, 'micro': {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': 0}}

        # Variant metrics
        variant_metrics = {
            'genomic': F1Calculator.prf_from_counts(variant_tp['genomic'], variant_fp['genomic'], variant_fn['genomic']),
            'coding': F1Calculator.prf_from_counts(variant_tp['coding'], variant_fp['coding'], variant_fn['coding']),
            'protein': F1Calculator.prf_from_counts(variant_tp['protein'], variant_fp['protein'], variant_fn['protein']),
            'all': F1Calculator.prf_from_counts(variant_tp['all'], variant_fp['all'], variant_fn['all']),
        }

        return {
            'cases': cases_metrics,
            'phenotype': phenotype_metrics,
            'variants': variant_metrics,
            'failures': failures,
        }

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print("F1 EVALUATION REPORT")
        print("=" * 80)

        # Cases
        cm = report.get('cases', {})
        print("\nCases (manual vs automatic):")
        print(f"  - Precision: {cm.get('precision', 0):.3f}")
        print(f"  - Recall:    {cm.get('recall', 0):.3f}")
        print(f"  - F1:        {cm.get('f1', 0):.3f}  (TP={cm.get('tp', 0)}, FP={cm.get('fp', 0)}, FN={cm.get('fn', 0)})")

        # Phenotype
        pm = report.get('phenotype', {})
        micro = pm.get('micro', {})
        print("\nPhenotype quality (High/Medium/Low):")
        print(f"  - Micro P: {micro.get('precision', 0):.3f}, R: {micro.get('recall', 0):.3f}, F1: {micro.get('f1', 0):.3f}")
        print(f"  - Macro F1: {pm.get('macro_f1', 0):.3f}")
        per_label = pm.get('per_label', {})
        for lbl in ['high', 'medium', 'low']:
            if lbl in per_label:
                m = per_label[lbl]
                print(f"    - {lbl.capitalize():<6} P: {m.get('precision', 0):.3f}, R: {m.get('recall', 0):.3f}, F1: {m.get('f1', 0):.3f}")

def evaluate_f1_between_manual_and_automatic(
    manual_df: pd.DataFrame,
    auto_extractions: Any,
    require_gene_match: bool = True,
    publication_match_threshold: float = 0.8,
    case_match_min_confidence: float = 0.3,
    print_report: bool = True,
) -> Dict[str, Any]:
    """Convenience API to compute and optionally print F1 metrics across features.

    Features scored:
    - Cases (presence/identity) between manual and automatic curations
    - Phenotype quality: manual phenotype_quality vs auto parsed confidence (A-F rules)
    - Variants: genomic (g.), cDNA/coding (c.), protein (p.), and combined
    """
    evaluator = F1Evaluator(
        require_gene_match=require_gene_match,
        publication_match_threshold=publication_match_threshold,
        case_match_min_confidence=case_match_min_confidence,
    )
    report = evaluator.evaluate(manual_df, auto_extractions)
    if print_report:
        evaluator.print_report(report)
    return report