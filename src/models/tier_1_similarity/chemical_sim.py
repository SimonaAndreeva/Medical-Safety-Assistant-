import numpy as np
from scipy.sparse import csr_matrix
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
        Optimized for Option 3: Sparse Matrix Algebra.
        Since Morgan fingerprints are ~95% zeroes, sparse multiplication 
        drastically accelerates the binary T=c/(a+b+c) calculation.

        Modes:
            Search Mode: target_vectors is 1D (1 query drug)
            Matrix Mode: target_vectors is 2D (N drugs)
        """
        # Ensure we are working with 2D dense arrays first to avoid shape issues
        t_dense = np.atleast_2d(target_vectors).astype(bool).astype(int)
        a_dense = np.atleast_2d(all_vectors).astype(bool).astype(int)
        
        is_1d_search = target_vectors.ndim == 1

        # Convert to Sparse Compressed Row (CSR) matrices
        t_csr = csr_matrix(t_dense)
        a_csr = csr_matrix(a_dense)

        # 1. Intersection (c): sparse dot product is instantly calculated 
        #    by only multiplying the non-zero (1) elements.
        intersection = t_csr.dot(a_csr.T).toarray()  # Shape: (N, M)

        # 2. Vector Magnitudes (a+c) and (b+c)
        #    Counting the number of 1s in each vector:
        t_sums = np.array(t_csr.sum(axis=1))  # Shape: (N, 1)
        a_sums = np.array(a_csr.sum(axis=1)).T  # Shape: (1, M)

        # 3. Union (a+b+c) = (a+c) + (b+c) - c
        union = t_sums + a_sums - intersection

        # Prevent division by zero for completely empty (zero) vectors
        union[union == 0] = 1.0

        # 4. Tanimoto = Intersection / Union
        result = intersection / union

        if is_1d_search:
            return result.flatten()
            
        return result