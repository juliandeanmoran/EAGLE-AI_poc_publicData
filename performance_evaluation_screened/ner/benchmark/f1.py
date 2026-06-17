import pandas as pd
import warnings

from typing import Dict, Any, List, Tuple
from .parsers import PhenotypeConfidenceParser, VariantNormalizer
from .matching import PublicationMatcher, CaseMatcher
from .publication import PublicationParser
from .text import TextNormalizer
from .processing import DataProcessor

warnings.filterwarnings("ignore")


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
                 publication_match_threshold: float = 0.6,
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