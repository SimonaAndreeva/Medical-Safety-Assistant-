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
    def calculate_tanimoto(query_fingerprints, database_fingerprints):
        """
        Computes Tanimoto (Jaccard) Similarity for binary fingerprint vectors.
        Optimized for Option 3: Sparse Matrix Algebra.
        Since Morgan fingerprints are ~95% zeroes, sparse multiplication 
        drastically accelerates the binary T=c/(a+b+c) calculation.

        Modes:
            Search Mode: query_fingerprints is 1D (1 query drug)
            Matrix Mode: query_fingerprints is 2D (N drugs)
        """
        # Ensure we are working with 2D dense arrays first to avoid shape issues
        query_dense = np.atleast_2d(query_fingerprints).astype(bool).astype(int)
        database_dense = np.atleast_2d(database_fingerprints).astype(bool).astype(int)
        
        is_single_query_mode = query_fingerprints.ndim == 1

        # Convert to Sparse Compressed Row (CSR) matrices
        query_sparse = csr_matrix(query_dense)
        database_sparse = csr_matrix(database_dense)

        # 1. Intersection (c): sparse dot product is instantly calculated 
        #    by only multiplying the non-zero (1) bits.
        shared_bits_count = query_sparse.dot(database_sparse.T).toarray()  # Shape: (N, M)

        # 2. Vector Magnitudes (a+c) and (b+c)
        #    Counting the number of 1s (set bits) in each fingerprint:
        query_bits_count = np.array(query_sparse.sum(axis=1))  # Shape: (N, 1)
        database_bits_count = np.array(database_sparse.sum(axis=1)).T  # Shape: (1, M)

        # 3. Union (a+b+c) = (a+c) + (b+c) - c
        #    Total unique set bits across both fingerprints
        total_unique_bits_count = query_bits_count + database_bits_count - shared_bits_count

        # Prevent division by zero for completely empty (zero) vectors
        total_unique_bits_count[total_unique_bits_count == 0] = 1.0

        # 4. Tanimoto = Intersection / Union
        tanimoto_scores = shared_bits_count / total_unique_bits_count

        if is_single_query_mode:
            return tanimoto_scores.flatten()
            
        return tanimoto_scores