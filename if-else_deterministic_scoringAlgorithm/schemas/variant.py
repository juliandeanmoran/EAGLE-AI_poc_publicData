from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class VariantStatusEnum(str, Enum):
    RARE = "rare"
    COMMON = "common"


class VariantType(str, Enum):
    MISSENSE = "Missense"
    NONSENSE = "Nonsense"
    SILENT = "Silent/Synonymous"
    FRAMESHIFT = "Frameshift"
    INFRAME = "Inframe Insertion/Deletion"
    SPLICE_SITE = "Splice Site"
    STOP_GAIN = "Stop Gain"
    START_LOSS = "Start Loss"
    COPY_NUMBER_VARIATION = "Copy Number Variation"
    DELETION = "Deletion"
    DUPLICATION = "Duplication"
    AMPLIFICATION = "Amplification"
    INVERSION = "Inversion"
    TRANSLOCATION = "Translocation"
    REPEAT_EXPANSION = "Repeat Expansion"
    REGULATORY = "Regulatory Variant"
    UTR5 = "5' UTR Variant"
    UTR3 = "3' UTR Variant"
    INTERGENIC = "Intergenic Variant"
    NON_CODING_RNA = "Non-Coding RNA Variant"
    OTHER = "Other"


class VariantCategory(str, Enum):
    SNP = "Single Nucleotide Polymorphism"
    INDEL = "Insertion/Deletion"
    CNV = "Copy Number Variation"
    SV = "Structural Variant"
    OTHER = "Other"


class ClinicalSignificance(str, Enum):
    BENIGN = "Benign"
    LIKELY_BENIGN = "Likely Benign"
    UNCERTAIN = "Uncertain Significance"
    LIKELY_PATHOGENIC = "Likely Pathogenic"
    PATHOGENIC = "Pathogenic"


# Base schema with common fields
class Variant(BaseModel):
    gene_id: int = Field(
        description="Unique identifier for the gene associated with this variant"
    )
    genotyping_method: Optional[str] = Field(
        description="Technical approach used to identify the variant (e.g., whole genome sequencing, exome sequencing, targeted panel)"
    )
    reference_genome: Optional[str] = Field(
        description="Version of the human reference genome used for variant calling and annotation (e.g., GRCh37/hg19, GRCh38/hg38)"
    )
    chromosome: str = Field(
        description="Chromosome on which the variant is located (e.g., 1-22, X, Y, MT)"
    )
    start_position: int = Field(
        description="Starting genomic coordinate of the variant on the chromosome"
    )
    end_position: int = Field(
        description="Ending genomic coordinate of the variant on the chromosome"
    )
    reference_allele: str = Field(
        description="Nucleotide sequence in the reference genome at this position"
    )
    alternate_allele: str = Field(
        description="Alternative nucleotide sequence identified in the sample"
    )
    region: Optional[str] = Field(
        description="Genomic region containing the variant (e.g., exon, intron, promoter, UTR)"
    )
    transcripts: Optional[Dict[str, Optional[List[str]]]] = Field(
        description="List of transcript identifiers affected by this variant"
    )
    variant: str = Field(
        description="Concise representation of the variant in standard format (e.g., chr1:g.1234A>G)"
    )
    genomic_hgvs: Optional[str] = Field(
        description="Variant described using genomic HGVS nomenclature (e.g., NC_000001.11:g.1234A>G)"
    )
    cdna_hgvs: Optional[str] = Field(
        description="Variant described using cDNA HGVS nomenclature (e.g., NM_001234.5:c.123A>G)"
    )
    protein_hgvs: Optional[str] = Field(
        description="Variant described using protein HGVS nomenclature (e.g., NP_001234.5:p.Lys41Arg)"
    )
    variant_description: Optional[str] = Field(
        description="Detailed textual description of the variant including clinical context and functional implications"
    )
    rsid: Optional[str] = Field(
        description="Reference SNP ID number from dbSNP database if available (e.g., rs12345)"
    )
    variant_type: Optional[VariantType] = Field(
        description="The molecular classification of the genetic variant (e.g., missense, nonsense, frameshift)"
    )
    category: Optional[VariantCategory] = Field(
        description="Broader classification of the variant type (SNP, INDEL, CNV, etc.)"
    )
    zygosity: Optional[str] = Field(
        description="Genotype state of the variant (e.g., heterozygous, homozygous, hemizygous)"
    )
    # Stored as JSON in DB, expecting dict structure
    population_frequency: Optional[Dict[str, float]] = Field(
        description="Allele frequencies of the variant across different population databases (e.g., gnomAD, 1000 Genomes, ExAC)"
    )
    impact: Optional[str] = Field(
        description="Predicted functional consequence of the variant on protein function (e.g., loss-of-function, gain-of-function)"
    )
    inheritance_pattern: Optional[str] = Field(
        description="Mode of transmission of the variant in the family (e.g., de novo, autosomal dominant, autosomal recessive, X-linked)"
    )
    segregation_data: Optional[str] = Field(
        description="Information about how the variant segregates with the phenotype in family members"
    )
    sift_score: Optional[float] = Field(
        description="SIFT algorithm score predicting the effect of amino acid substitution on protein function (0-1, lower scores indicate damaging)"
    )
    polyphen_score: Optional[float] = Field(
        description="PolyPhen-2 algorithm score predicting the impact of amino acid substitution on protein structure and function (0-1, higher scores indicate damaging)"
    )
    cadd_score: Optional[float] = Field(
        description="Combined Annotation Dependent Depletion (CADD) score for predicting deleteriousness of genetic variants (higher scores indicate more deleterious)"
    )
    # Stored as JSON in DB, expecting dict structure
    other_scores: Optional[Dict[str, float]] = Field(
        description="Additional computational prediction scores for variant pathogenicity assessment (e.g., REVEL, MutationTaster, GERP++)"
    )
    linkage_to_asd: bool = Field(
        description="Indicates whether the variant has been linked to Autism Spectrum Disorder in literature or databases"
    )
    clinical_significance: Optional[ClinicalSignificance] = Field(
        description="Clinical interpretation of the variant according to ACMG guidelines (Benign, Likely Benign, Uncertain Significance, Likely Pathogenic, Pathogenic)"
    )
    variant_status: Optional[VariantStatusEnum] = Field(
        description="Classification of the variant based on population frequency (rare or common)"
    )

    class Config:
        use_enum_values = True


class VariantWithQuotes(BaseModel):
    gene_id: int = Field(
        description="Unique identifier for the gene associated with this variant"
    )
    genotyping_method: Optional[str] = Field(
        description="Technical approach used to identify the variant (e.g., whole genome sequencing, exome sequencing, targeted panel)"
    )
    reference_genome: Optional[str] = Field(
        description="Version of the human reference genome used for variant calling and annotation (e.g., GRCh37/hg19, GRCh38/hg38)"
    )
    chromosome: str = Field(
        description="Chromosome on which the variant is located (e.g., 1-22, X, Y, MT)"
    )
    start_position: int = Field(
        description="Starting genomic coordinate of the variant on the chromosome"
    )
    end_position: int = Field(
        description="Ending genomic coordinate of the variant on the chromosome"
    )
    reference_allele: str = Field(
        description="Nucleotide sequence in the reference genome at this position"
    )
    alternate_allele: str = Field(
        description="Alternative nucleotide sequence identified in the sample"
    )
    region: Optional[str] = Field(
        description="Genomic region containing the variant (e.g., exon, intron, promoter, UTR)"
    )
    transcripts: Optional[Dict[str, Optional[List[str]]]] = Field(
        description="List of transcript identifiers affected by this variant"
    )
    variant: str = Field(
        description="Concise representation of the variant in standard format (e.g., chr1:g.1234A>G)"
    )
    genomic_hgvs: Optional[str] = Field(
        description="Variant described using genomic HGVS nomenclature (e.g., NC_000001.11:g.1234A>G)"
    )
    cdna_hgvs: Optional[str] = Field(
        description="Variant described using cDNA HGVS nomenclature (e.g., NM_001234.5:c.123A>G)"
    )
    protein_hgvs: Optional[str] = Field(
        description="Variant described using protein HGVS nomenclature (e.g., NP_001234.5:p.Lys41Arg)"
    )
    variant_description: Optional[str] = Field(
        description="Detailed textual description of the variant including clinical context and functional implications"
    )
    rsid: Optional[str] = Field(
        description="Reference SNP ID number from dbSNP database if available (e.g., rs12345)"
    )
    variant_type: Optional[VariantType] = Field(
        description="The molecular classification of the genetic variant (e.g., missense, nonsense, frameshift)"
    )
    category: Optional[VariantCategory] = Field(
        description="Broader classification of the variant type (SNP, INDEL, CNV, etc.)"
    )
    zygosity: Optional[str] = Field(
        description="Genotype state of the variant (e.g., heterozygous, homozygous, hemizygous)"
    )
    # Stored as JSON in DB, expecting dict structure
    population_frequency: Optional[Dict[str, float]] = Field(
        description="Allele frequencies of the variant across different population databases (e.g., gnomAD, 1000 Genomes, ExAC)"
    )
    impact: Optional[str] = Field(
        description="Predicted functional consequence of the variant on protein function (e.g., loss-of-function, gain-of-function)"
    )
    inheritance_pattern: Optional[str] = Field(
        description="Mode of transmission of the variant in the family (e.g., de novo, autosomal dominant, autosomal recessive, X-linked)"
    )
    segregation_data: Optional[str] = Field(
        description="Information about how the variant segregates with the phenotype in family members"
    )
    sift_score: Optional[float] = Field(
        description="SIFT algorithm score predicting the effect of amino acid substitution on protein function (0-1, lower scores indicate damaging)"
    )
    polyphen_score: Optional[float] = Field(
        description="PolyPhen-2 algorithm score predicting the impact of amino acid substitution on protein structure and function (0-1, higher scores indicate damaging)"
    )
    cadd_score: Optional[float] = Field(
        description="Combined Annotation Dependent Depletion (CADD) score for predicting deleteriousness of genetic variants (higher scores indicate more deleterious)"
    )
    # Stored as JSON in DB, expecting dict structure
    other_scores: Optional[Dict[str, float]] = Field(
        description="Additional computational prediction scores for variant pathogenicity assessment (e.g., REVEL, MutationTaster, GERP++)"
    )
    linkage_to_asd: bool = Field(
        description="Indicates whether the variant has been linked to Autism Spectrum Disorder in literature or databases"
    )
    clinical_significance: Optional[ClinicalSignificance] = Field(
        description="Clinical interpretation of the variant according to ACMG guidelines (Benign, Likely Benign, Uncertain Significance, Likely Pathogenic, Pathogenic)"
    )
    variant_status: Optional[VariantStatusEnum] = Field(
        description="Classification of the variant based on population frequency (rare or common)"
    )

    quotes: Optional[List[str]] = Field(
        description="Direct quotes from the paper describing the variant, including details about its identification, characterization, inheritance pattern, and any other relevant information as presented in the original text. Must extract the complete sentence from the paper where variant details are mentioned for the specific variant being extracted."
    )

    class Config:
        use_enum_values = True
