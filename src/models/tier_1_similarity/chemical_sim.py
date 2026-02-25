import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityEngine:
    """
    Tier 1 — Chemical Similarity Engine.

    Responsibility: How similar are two drugs by their chemical structure?
    Input: Binary Morgan fingerprint vectors (2048-bit arrays of 0s and 1s).

    This class does NOT perform graph operations, network walks, or any
    continuous-vector math. Those belong in Tier 2 (rwr.py).

    Methods:
        calculate_tanimoto — Jaccard similarity on binary fingerprints
    """

    @staticmethod
    def calculate_tanimoto(target_vectors, all_vectors):
        """
        Computes Tanimoto (Jaccard) Similarity for binary fingerprint vectors.

        Modes:
            Search Mode: target_vectors is 1D (1 query drug)
                         -> Returns 1D array of scores against all_vectors
            Matrix Mode: target_vectors is 2D (N drugs)
                         -> Returns 2D (N x M) pairwise similarity matrix

        Formula: J(q, a) = |q ∩ a| / |q ∪ a|
        """
        # Convert to binary integers (0 or 1) to enable matrix multiplication
        t_mat = target_vectors.astype(bool).astype(int)
        a_mat = all_vectors.astype(bool).astype(int)

        # Detect Search Mode (single 1D query vector)
        is_1d_search = False
        if t_mat.ndim == 1:
            is_1d_search = True
            t_mat = t_mat.reshape(1, -1)  # Promote to 2D for unified math path

        # 1. Intersection via dot product: (N x F) · (F x M) = (N x M)
        #    Each cell [i][j] counts the number of matching '1' bits
        intersection = np.dot(t_mat, a_mat.T)

        # 2. Union = |A| + |B| - |A ∩ B|, computed via broadcasting
        row_sums_t = np.sum(t_mat, axis=1).reshape(-1, 1)  # Shape (N, 1)
        row_sums_a = np.sum(a_mat, axis=1).reshape(1, -1)  # Shape (1, M)
        union = row_sums_t + row_sums_a - intersection

        # Guard: if both vectors are all-zero, yield score=0 not NaN
        union[union == 0] = 1.0

        result = intersection / union

        # Return flat 1D array for Search Mode, 2D matrix for Matrix Mode
        if is_1d_search:
            return result.flatten()

        return result