from src.models.tier_1_similarity.binary_similarity import TanimotoEngine, JaccardEngine
import numpy as np


class SimilarityEngine:
    """
    Tier 1 — Chemical Similarity Engine.

    Thin wrapper that exposes TanimotoEngine under the existing SimilarityEngine API
    to preserve backwards compatibility with all callers (services.py, advanced_fusion.py etc).

    For phenotypic similarity, see JaccardEngine in binary_similarity.py.
    """

    @staticmethod
    def calculate_tanimoto(query_fingerprints, database_fingerprints):
        """
        Compute pairwise Tanimoto similarity for binary Morgan fingerprint vectors.

        Delegates to TanimotoEngine, which inherits the shared Sparse CSR
        intersection-over-union math from BinarySimilarityEngine.

        Modes:
            Search Mode: query_fingerprints is 1D (single query drug)
            Matrix Mode: query_fingerprints is 2D (N query drugs)
        """
        return TanimotoEngine.calculate(
            query_fingerprints=query_fingerprints,
            database_fingerprints=database_fingerprints
        )