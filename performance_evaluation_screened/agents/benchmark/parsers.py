import ast
from typing import Dict, Any, Optional, List
import re
import warnings

warnings.filterwarnings("ignore")


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
            return "unknown"
        s = str(label).strip().lower()
        if s.startswith("high"):
            return "high"
        if s.startswith("medium"):
            return "medium"
        if s.startswith("low"):
            return "low"
        return "unknown"

    @staticmethod
    def parse_from_auto(phenotypic_evidence: Optional[Dict[str, Any]]) -> str:
        if not isinstance(phenotypic_evidence, dict):
            return "unknown"
        # Prefer indicators if present
        a = bool(phenotypic_evidence.get("phenotype_source_indicator_a"))
        b = bool(phenotypic_evidence.get("phenotype_source_indicator_b"))
        c = bool(phenotypic_evidence.get("phenotype_source_indicator_c"))
        d = bool(phenotypic_evidence.get("phenotype_source_indicator_d"))
        e = bool(phenotypic_evidence.get("phenotype_source_indicator_e"))
        f = bool(phenotypic_evidence.get("phenotype_source_indicator_f"))

        if a or b or c:
            return "high"
        if d:
            return "medium"
        if e or f:
            return "low"

        # Fallback: direct field
        return PhenotypeConfidenceParser._canonicalize(
            str(phenotypic_evidence.get("phenotype_confidence") or "")
        )

    @staticmethod
    def parse_from_manual(value: Any) -> str:
        return PhenotypeConfidenceParser._canonicalize(str(value or ""))


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
        return [t.strip() for t in s.split(",") if t.strip()]

    @staticmethod
    def _strip_wrappers(text: str) -> str:
        return (
            (text or "")
            .replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
        )

    @staticmethod
    def normalize_genomic(hgvs: str) -> str:
        if not hgvs:
            return ""
        s = str(hgvs)
        s = VariantNormalizer._strip_wrappers(s)
        s = s.replace(" ", "")
        s = s.replace("Chr", "chr").replace("CHR", "chr")
        s_lower = s.lower()

        # Extract g.-prefixed substring if present
        pos = s_lower.find("g.")
        if pos >= 0:
            core = s[pos:]
        else:
            core = s
        # Drop reference genome annotations like (hg19) or :hg19
        core = re.sub(
            r"\(hg\d+\)|hg\d+|\(grch\d+\)|grch\d+", "", core, flags=re.IGNORECASE
        )
        # Keep only a subset of allowed characters; uppercase nucleotides
        core = core.replace("chr", "")
        core = core.replace(":", "")
        # Standardize case: keep 'g.' lowercase and nucleotides upper
        if core.lower().startswith("g."):
            prefix = "g."
            rest = core[2:]
        else:
            prefix = ""
            rest = core
        # Uppercase letters in rest, keep punctuation
        rest = re.sub(r"[a-z]", lambda m: m.group(0).upper(), rest)
        return f"{prefix}{rest}"

    @staticmethod
    def normalize_cdna(cdna: str) -> str:
        if not cdna:
            return ""
        s = str(cdna)
        s = VariantNormalizer._strip_wrappers(s)
        s = s.replace(" ", "")
        if not s.lower().startswith("c."):
            # Try to extract c. segment
            m = re.search(r"(c\.[^;\s,]+)", s, flags=re.IGNORECASE)
            if m:
                s = m.group(1)
        # Enforce c. prefix and uppercase rest
        if s.lower().startswith("c."):
            prefix = "c."
            rest = s[2:]
        else:
            prefix = ""
            rest = s
        rest = re.sub(r"[a-z]", lambda m: m.group(0).upper(), rest)
        return f"{prefix}{rest}"

    @staticmethod
    def normalize_protein(protein: str) -> str:
        if not protein:
            return ""
        s = str(protein)
        s = VariantNormalizer._strip_wrappers(s)
        s = s.replace(" ", "")
        # Extract p. segment if present
        if not s.lower().startswith("p."):
            m = re.search(r"(p\.[^;\s,]+)", s, flags=re.IGNORECASE)
            if m:
                s = m.group(1)
        # Enforce uppercase variant portion
        if s.lower().startswith("p."):
            prefix = "p."
            rest = s[2:]
        else:
            prefix = ""
            rest = s
        rest = re.sub(r"[a-z]", lambda m: m.group(0).upper(), rest)
        return f"{prefix}{rest}"

    @staticmethod
    def extract_manual_variants(manual_row: Dict[str, Any]) -> Dict[str, set]:
        """Return sets of normalized variants from a manual dataframe row.

        Looks for several common column names for genomic, cDNA, and protein variants.
        """
        # Candidate column names
        genomic_cols = [
            "genomic_variants",
            "genomic_variant",
            "genomic_hgvs",
            "genomic",
        ]
        coding_cols = ["coding_variants", "coding_variant", "cdna_hgvs", "coding_hgvs"]
        protein_cols = ["protein_variants", "protein_variant", "protein_hgvs"]

        def _first_present(cols: List[str]) -> Optional[str]:
            for c in cols:
                if c in manual_row and manual_row.get(c) not in (None, ""):
                    return c
            return None

        g_col = _first_present(genomic_cols)
        c_col = _first_present(coding_cols)
        p_col = _first_present(protein_cols)

        genomic_values = (
            VariantNormalizer._clean_list_like(manual_row.get(g_col)) if g_col else []
        )
        coding_values = (
            VariantNormalizer._clean_list_like(manual_row.get(c_col)) if c_col else []
        )
        protein_values = (
            VariantNormalizer._clean_list_like(manual_row.get(p_col)) if p_col else []
        )

        # Fallback: attempt to parse from free-text 'description' or similar
        if not genomic_values:
            for free_col in ["description", "notes", "variant_notes"]:
                free_text = manual_row.get(free_col)
                if free_text:
                    # Extract g. segments
                    genomic_values.extend(
                        re.findall(r"(g\.[^\s\]\)\;\,\|]+)", str(free_text))
                    )
                    if genomic_values:
                        break

        genomic_set = {
            VariantNormalizer.normalize_genomic(v)
            for v in genomic_values
            if VariantNormalizer.normalize_genomic(v)
        }
        coding_set = {
            VariantNormalizer.normalize_cdna(v)
            for v in coding_values
            if VariantNormalizer.normalize_cdna(v)
        }
        protein_set = {
            VariantNormalizer.normalize_protein(v)
            for v in protein_values
            if VariantNormalizer.normalize_protein(v)
        }

        return {
            "genomic": genomic_set,
            "coding": coding_set,
            "protein": protein_set,
        }

    @staticmethod
    def extract_auto_variants(auto_case: Dict[str, Any]) -> Dict[str, set]:
        variant = auto_case.get("variant", {}) if isinstance(auto_case, dict) else {}
        if not isinstance(variant, dict):
            variant = {}
        genomic_values: List[str] = []
        coding_values: List[str] = []
        protein_values: List[str] = []

        # Preferred explicit fields
        if variant.get("genomic_hgvs"):
            genomic_values.append(str(variant.get("genomic_hgvs")))
        if variant.get("cdna_hgvs"):
            coding_values.append(str(variant.get("cdna_hgvs")))
        if variant.get("protein_hgvs"):
            protein_values.append(str(variant.get("protein_hgvs")))

        # Fallback: generic 'variant' field may contain c. or g.
        vfield = variant.get("variant")
        if vfield:
            s = str(vfield)
            if "c." in s or s.lower().startswith("c."):
                coding_values.append(s)
            if "g." in s or s.lower().startswith("g."):
                genomic_values.append(s)

        genomic_set = {
            VariantNormalizer.normalize_genomic(v)
            for v in genomic_values
            if VariantNormalizer.normalize_genomic(v)
        }
        coding_set = {
            VariantNormalizer.normalize_cdna(v)
            for v in coding_values
            if VariantNormalizer.normalize_cdna(v)
        }
        protein_set = {
            VariantNormalizer.normalize_protein(v)
            for v in protein_values
            if VariantNormalizer.normalize_protein(v)
        }

        return {
            "genomic": genomic_set,
            "coding": coding_set,
            "protein": protein_set,
        }
