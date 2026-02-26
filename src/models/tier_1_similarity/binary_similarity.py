import numpy as np
from scipy.sparse import csr_matrix


class BinarySimilarityEngine:
    """
    Shared Base Class — Binary Similarity Math Engine.

    Implements intersection-over-union (IoU) for binary vectors using
    SciPy Sparse CSR Matrix multiplication.

    Because Morgan fingerprints and SIDER phenotypic profiles are both
    >90% zeroes, sparse multiplication skips zero-bits entirely.
    This makes intersection and union calculations run in:
        O(nnz(A) * nnz(B))  instead of  O(Features)
    
    Both Tanimoto (chemistry) and Jaccard (phenotypics) are mathematically
    identical: c / (a + b + c). Only their domain and naming differs.
    This base class provides the single shared implementation.

    Subclasses:
        TanimotoEngine — binary molecule fingerprints
        JaccardEngine  — binary phenotypic side-effect profiles
    """

    @staticmethod
    def _calculate_binary_similarity(query_vectors, database_vectors):
        """
        Core Sparse CSR binary intersection-over-union calculation.

        Args:
            query_vectors:    2D numpy array, shape (N, Features)
            database_vectors: 2D numpy array, shape (M, Features)

        Returns:
            similarity_scores: 2D numpy array, shape (N, M)
        """
        # Normalize inputs to 2D bool-to-int arrays
        query_dense    = np.atleast_2d(query_vectors).astype(bool).astype(int)
        database_dense = np.atleast_2d(database_vectors).astype(bool).astype(int)

        # Convert to Sparse Compressed Row matrices
        query_sparse    = csr_matrix(query_dense)
        database_sparse = csr_matrix(database_dense)

        # 1. Intersection (c): Only non-zero bits are multiplied
        shared_bits_count = query_sparse.dot(database_sparse.T).toarray()  # Shape: (N, M)

        # 2. Per-vector set sizes (a+c) and (b+c)
        query_bits_count    = np.array(query_sparse.sum(axis=1))     # Shape: (N, 1)
        database_bits_count = np.array(database_sparse.sum(axis=1)).T  # Shape: (1, M)

        # 3. Union (a+b+c) = (a+c) + (b+c) - c
        total_unique_bits_count = query_bits_count + database_bits_count - shared_bits_count

        # Prevent division by zero for all-zero vectors
        total_unique_bits_count[total_unique_bits_count == 0] = 1.0

        # 4. Score = Intersection / Union
        similarity_scores = shared_bits_count / total_unique_bits_count

        return similarity_scores


class TanimotoEngine(BinarySimilarityEngine):
    """
    Tier 1 — Chemical Similarity Engine.

    Domain: Binary Morgan fingerprint vectors (2048-bit arrays).
    Formula: Tanimoto = shared_bits / total_unique_bits  (c / a+b+c)

    Modes:
        Search Mode: query_fingerprints is 1D (single query drug)
        Matrix Mode: query_fingerprints is 2D (N query drugs)
    """

    @staticmethod
    def calculate(query_fingerprints, database_fingerprints):
        """
        Compute pairwise Tanimoto similarity between two sets of
        Morgan fingerprints using Sparse CSR matrix multiplication.

        Returns:
            1D array if query was 1D (Search Mode)
            2D array if query was 2D (Matrix Mode)
        """
        is_single_query_mode = np.ndim(query_fingerprints) == 1

        tanimoto_scores = BinarySimilarityEngine._calculate_binary_similarity(
            query_vectors=query_fingerprints,
            database_vectors=database_fingerprints
        )

        return tanimoto_scores.flatten() if is_single_query_mode else tanimoto_scores


class JaccardEngine(BinarySimilarityEngine):
    """
    Tier 1 — Phenotypic Similarity Engine.

    Domain: Binary SIDER side-effect profile vectors.
    Formula: Jaccard = shared_phenotypes / total_unique_phenotypes  (c / a+b+c)

    Identical math to Tanimoto, but applied to clinical phenotypic data
    rather than molecular fingerprints.
    """

    @staticmethod
    def calculate(query_phenotype_vectors, database_phenotype_vectors):
        """
        Compute pairwise Jaccard similarity between two sets of
        binary phenotypic side-effect profiles.

        Returns:
            2D similarity matrix, shape (N, M)
        """
        jaccard_scores = BinarySimilarityEngine._calculate_binary_similarity(
            query_vectors=query_phenotype_vectors,
            database_vectors=database_phenotype_vectors
        )

        return jaccard_scores
