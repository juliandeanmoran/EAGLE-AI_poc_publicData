from .text import TextNormalizer

import warnings

warnings.filterwarnings("ignore")


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
