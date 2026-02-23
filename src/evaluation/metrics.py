import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, average_precision_score

def calculate_ddi_metrics(y_true, y_scores):
    """
    Core metric engine for the thesis.
    AUPR is the primary indicator of performance in imbalanced DDI tasks.
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

    return {
        "AUPR": round(aupr, 4),
        "AUROC": round(auroc, 4),
        "Avg_Precision": round(ap, 4)
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