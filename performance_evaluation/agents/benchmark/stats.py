import pandas as pd
from typing import Dict, Any

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