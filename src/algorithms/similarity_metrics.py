import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SimilarityEngine:
    """
    Core algorithm for calculating drug similarities (Tanimoto & Cosine).
    """

    @staticmethod
    def calculate_tanimoto(target_vector, all_vectors):
        """
        Computes Tanimoto (Jaccard) Similarity between one target and many others.
        Formula: Intersection / Union
        """
        # Ensure inputs are boolean (True/False)
        t_bool = target_vector.astype(bool)
        all_bool = all_vectors.astype(bool)
        
        # 1. Intersection (Bits present in BOTH)
        intersection = np.sum(all_bool & t_bool, axis=1)
        
        # 2. Union (Bits present in EITHER)
        sum_target = np.sum(t_bool)
        sum_all = np.sum(all_bool, axis=1)
        
        union = sum_target + sum_all - intersection
        
        # Avoid division by zero
        union[union == 0] = 1 
        
        return intersection / union

    @staticmethod
    def calculate_cosine(target_vector, all_vectors):
        """
        Computes Cosine Similarity (Angle).
        Useful for continuous vectors (like Biological Embeddings later).
        """
        return cosine_similarity(target_vector, all_vectors).flatten()