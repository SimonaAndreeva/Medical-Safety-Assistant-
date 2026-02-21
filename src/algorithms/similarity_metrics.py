import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine:
    """
    Core algorithm for calculating drug similarities (Tanimoto & Cosine).
    Updated to support Matrix-Matrix calculations.
    """

    @staticmethod
    def calculate_tanimoto(target_vectors, all_vectors):
        """
        Computes Tanimoto (Jaccard) Similarity.
        
        Modes:
        1. Search Mode: target_vectors is 1D (1 drug) -> Returns 1D Array
        2. Matrix Mode: target_vectors is 2D (All drugs) -> Returns 2D Adjacency Matrix
        """
        # Convert to binary integers (0 or 1)
        # We use .astype(int) to allow matrix multiplication
        t_mat = target_vectors.astype(bool).astype(int)
        a_mat = all_vectors.astype(bool).astype(int)

        # Handle "Search Mode" (Single Vector)
        is_1d_search = False
        if t_mat.ndim == 1:
            is_1d_search = True
            t_mat = t_mat.reshape(1, -1) # Make it 2D for the math to work

        # 1. Intersection (Matrix Dot Product)
        # Algebra: (A . B^T) counts the matching '1's
        intersection = np.dot(t_mat, a_mat.T)
        
        # 2. Union (A + B - Intersection)
        # Count bits set to 1 in each row
        row_sums_t = np.sum(t_mat, axis=1).reshape(-1, 1) # Shape (N, 1)
        row_sums_a = np.sum(a_mat, axis=1).reshape(1, -1) # Shape (1, M)
        
        # Broadcasting adds them into a grid
        union = row_sums_t + row_sums_a - intersection
        
        # Avoid division by zero
        union[union == 0] = 1.0
        
        # 3. Calculate Tanimoto
        result = intersection / union
        
        # Return 1D array for search, 2D matrix for HIN
        if is_1d_search:
            return result.flatten()
            
        return result

    @staticmethod
    def calculate_cosine(target_vector, all_vectors):
        """
        Computes Cosine Similarity (Angle).
        Useful for continuous vectors (like Biological Embeddings).
        """
        # Scikit-learn handles 1D vs 2D automatically
        return cosine_similarity(target_vector, all_vectors).flatten()