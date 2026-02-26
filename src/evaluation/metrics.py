import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score

def enrichment_factor(y_true, y_scores, top_fraction=0.01):
    """
    Enrichment Factor at top K% (EF@K%).

    Pharma standard: measures how many more true positives are in the top
    K% of ranked predictions vs random selection.

    EF = 1.0 → model is no better than random.
    EF = 10.0 → top 1% contains 10× more positives than random chance.
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    num_total = len(y_scores)
    num_top = max(1, int(num_total * top_fraction))

    # Rank predictions by score (highest first)
    top_indices = np.argsort(y_scores)[::-1][:num_top]
    top_true_positives = y_true[top_indices].sum()

    # Expected true positives if we picked randomly from the same size pool
    total_positives = y_true.sum()
    expected_random = total_positives * top_fraction

    if expected_random == 0:
        return 0.0

    return round(float(top_true_positives / expected_random), 4)


def calculate_ddi_metrics(y_true, y_scores, ef_fraction=0.01):
    """
    Core metric engine for the thesis.
    AUPR is the primary indicator of performance in imbalanced DDI tasks.
    Also computes Enrichment Factor at the specified top fraction (default: 1%).
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # 1. AUPR (Area Under Precision-Recall Curve) - THE DEFENSE STANDARD
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aupr = auc(recall, precision)
    
    # 2. AUROC (Area Under Receiver Operating Characteristic)
    auroc = roc_auc_score(y_true, y_scores)
    
    # 3. AP (Average Precision)
    ap = average_precision_score(y_true, y_scores)

    # 4. Enrichment Factor @ top K%
    ef = enrichment_factor(y_true, y_scores, top_fraction=ef_fraction)

    return {
        "AUPR": round(aupr, 4),
        "AUROC": round(auroc, 4),
        "Avg_Precision": round(ap, 4),
        f"EF@{int(ef_fraction*100)}%": ef
    }

def print_performance_comparison(baseline_metrics, advanced_metrics):
    """Formatted output for thesis documentation."""
    print("\n📊 COMPARATIVE ANALYSIS: Baseline vs. Advanced Fusion")
    print("-" * 55)
    print(f"{'Metric':<15} | {'Chemical Baseline':<20} | {'Advanced Fusion':<15}")
    print("-" * 55)
    for m in baseline_metrics.keys():
        diff = advanced_metrics[m] - baseline_metrics[m]
        symbol = "+" if diff >= 0 else ""
        print(f"{m:<15} | {baseline_metrics[m]:<20} | {advanced_metrics[m]:<15} ({symbol}{diff:.4f})")