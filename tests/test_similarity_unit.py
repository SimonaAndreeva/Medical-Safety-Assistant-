import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.tier_1_similarity.chemical_sim import SimilarityEngine

def test_identical_vectors():
    target = np.array([1, 1, 0, 0])
    all_vecs = np.array([[1, 1, 0, 0]])
    score = SimilarityEngine.calculate_tanimoto(target, all_vecs)
    assert score[0] == 1.0, f"Expected 1.0, got {score[0]}"

def test_disjoint_vectors():
    target = np.array([1, 1, 0, 0])
    all_vecs = np.array([[0, 0, 1, 1]])
    score = SimilarityEngine.calculate_tanimoto(target, all_vecs)
    assert score[0] == 0.0, f"Expected 0.0, got {score[0]}"

def test_half_overlap():
    target = np.array([1, 1, 0, 0])
    all_vecs = np.array([[1, 0, 0, 0]])
    # Intersection: 1
    # Union: 2
    # Score: 0.5
    score = SimilarityEngine.calculate_tanimoto(target, all_vecs)
    assert score[0] == 0.5, f"Expected 0.5, got {score[0]}"

def test_zero_vectors():
    target = np.array([0, 0, 0, 0])
    all_vecs = np.array([[0, 0, 0, 0]])
    score = SimilarityEngine.calculate_tanimoto(target, all_vecs)
    assert score[0] == 0.0, f"Expected 0.0 for zero vectors, got {score[0]}"

def test_matrix_mode():
    target = np.array([
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ])
    all_vecs = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    scores = SimilarityEngine.calculate_tanimoto(target, all_vecs)
    # Target 0 vs All 0: 1.0
    # Target 0 vs All 1: 0.0
    assert scores[0][0] == 1.0
    assert scores[0][1] == 0.0
    
if __name__ == "__main__":
    test_identical_vectors()
    test_disjoint_vectors()
    test_half_overlap()
    test_zero_vectors()
    test_matrix_mode()
    print("✅ All Tanimoto unit tests passed!")
