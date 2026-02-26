"""
Ablation Study — Similarity Model Evaluation
=============================================
Compares three conditions on the same ground-truth DDI dataset:

    Condition A: Chemical model only (TanimotoEngine on Morgan fingerprints)
    Condition B: Phenotypic model only (JaccardEngine on SIDER side-effect profiles)
    Condition C: Fused model (AdvancedFusionModel, weight_pheno from config)

Ground truth: drug_interactions table from the project database.
A pair is labelled y_true=1 if it has a known interaction, y_true=0 otherwise.

Runs across N_SEEDS different random negative-sample draws and reports
mean ± std for AUPR, AUROC, and EF@1% — the three metrics from the evaluation plan.

Usage:
    python src/evaluation/evaluate_ablation.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# --- Project root on sys.path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.config import settings
from src.models.tier_1_similarity.binary_similarity import TanimotoEngine, JaccardEngine
from src.models.tier_1_similarity.advanced_fusion import AdvancedFusionModel
from src.models.tier_2_network.ppi_network_model import PPINetworkModel
from src.evaluation.metrics import calculate_ddi_metrics

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_SEEDS = 5           # Number of random negative-sample draws for statistical validation
NEGATIVE_RATIO = 10   # Negatives per positive (industry standard: 1:10 ratio)
EF_FRACTION = 0.01    # Enrichment Factor at top 1%


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_positive_pairs(engine):
    """Load all known drug-drug interactions (DDIs) from the database as positive pairs."""
    query = """
        SELECT DISTINCT
            d1.id     AS id1,
            d1.smiles AS smiles1,
            d2.id     AS id2,
            d2.smiles AS smiles2
        FROM drug_interactions di
        JOIN drugs d1 ON d1.id = di.drug_a_id
        JOIN drugs d2 ON d2.id = di.drug_b_id
        WHERE d1.smiles IS NOT NULL
          AND d2.smiles IS NOT NULL
          AND d1.smiles != ''
          AND d2.smiles != ''
          AND d1.id != d2.id
    """
    df = pd.read_sql(query, engine)
    df['label'] = 1
    print(f"✅ Loaded {len(df):,} positive DDI pairs (ground truth).")
    return df


def load_all_drugs(engine):
    """Load all drugs with valid SMILES and DrugBank IDs for negative sampling."""
    return pd.read_sql("SELECT id, smiles FROM drugs WHERE smiles IS NOT NULL AND smiles != ''", engine)


def sample_negative_pairs(positive_df, all_drugs_df, seed):
    """
    Sample random drug pairs that are NOT in the positive set.

    Ratio controlled by NEGATIVE_RATIO constant (default 1:10).
    A fixed seed ensures reproducibility within a single run,
    while varying the seed across N_SEEDS captures sampling variance.
    """
    rng = np.random.default_rng(seed)
    num_negatives = len(positive_df) * NEGATIVE_RATIO

    # Build a set of known positive pairs for fast O(1) rejection
    positive_pairs_set = set(
        zip(positive_df['id1'], positive_df['id2'])
    ) | set(
        zip(positive_df['id2'], positive_df['id1'])  # symmetric
    )

    drug_ids = all_drugs_df['id'].values
    drug_smiles_map = dict(zip(all_drugs_df['id'], all_drugs_df['smiles']))

    negatives = []
    attempts = 0
    max_attempts = num_negatives * 20  # safety cap

    while len(negatives) < num_negatives and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(len(drug_ids), size=2, replace=False)
        id_a, id_b = drug_ids[i], drug_ids[j]

        if (id_a, id_b) not in positive_pairs_set:
            negatives.append({
                'id1': id_a, 'smiles1': drug_smiles_map[id_a],
                'id2': id_b, 'smiles2': drug_smiles_map[id_b],
                'label': 0
            })
            positive_pairs_set.add((id_a, id_b))  # prevent re-sampling

    neg_df = pd.DataFrame(negatives)
    print(f"   Sampled {len(neg_df):,} negative pairs (seed={seed}).")
    return neg_df


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

BATCH_SIZE = 512  # Process this many pairs at a time — keeps memory flat

def compute_scores_for_all_conditions(eval_df, model):
    """
    Compute pairwise similarity scores for all three conditions across every pair.

    Uses chunked batching (BATCH_SIZE pairs per matrix call) to keep memory flat.
    Without this, a 76k-pair eval set would create a 76k×76k matrix = ~43 GiB.

    All three conditions are evaluated on the SAME pairs — critical for valid ablation.
    """
    n = len(eval_df)
    chemical_scores   = np.zeros(n)
    phenotypic_scores = np.zeros(n)
    fused_scores      = np.zeros(n)
    network_scores    = np.zeros(n)

    query_smiles_all    = eval_df['smiles1'].tolist()
    database_smiles_all = eval_df['smiles2'].tolist()
    query_ids_all       = eval_df['id1'].tolist()
    database_ids_all    = eval_df['id2'].tolist()

    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)

        q_smiles  = query_smiles_all[start:end]
        d_smiles  = database_smiles_all[start:end]
        q_ids     = query_ids_all[start:end]
        d_ids     = database_ids_all[start:end]

        # --- Condition A: Chemical only (pairwise diagonal) ---
        chem_matrix = model.get_chemical_similarity(q_smiles, d_smiles)
        # get_chemical_similarity returns np.diag for matched-length lists > 1
        if hasattr(chem_matrix, '__len__') and len(np.array(chem_matrix).shape) == 2:
            chem_batch = np.diag(chem_matrix)
        else:
            chem_batch = np.array(chem_matrix).flatten()
        chemical_scores[start:end] = chem_batch

        # --- Condition B: Phenotypic only (pairwise diagonal) ---
        pheno_matrix = model.get_phenotypic_similarity(q_ids, d_ids)
        if hasattr(pheno_matrix, '__len__') and len(np.array(pheno_matrix).shape) == 2:
            pheno_batch = np.diag(pheno_matrix)
        else:
            pheno_batch = np.array(pheno_matrix).flatten()
        phenotypic_scores[start:end] = pheno_batch

        # --- Condition C: Fused ---
        drugs_query    = [{'id': i, 'smiles': s} for i, s in zip(q_ids, q_smiles)]
        drugs_database = [{'id': i, 'smiles': s} for i, s in zip(d_ids, d_smiles)]
        fused_batch    = model.predict_fusion_score(drugs_query, drugs_database)
        fused_scores[start:end] = np.array(fused_batch).flatten()

        # --- Condition D: PPI RWR Network ---
        net_matrix = model.ppi_model.get_network_similarity(q_ids, d_ids)
        if hasattr(net_matrix, '__len__') and len(np.array(net_matrix).shape) == 2:
            net_batch = np.diag(net_matrix)
        else:
            net_batch = np.array(net_matrix).flatten()
        network_scores[start:end] = net_batch

    return {
        'A: Chemical Only':    chemical_scores,
        'B: Phenotypic Only':  phenotypic_scores,
        'C: Tier 1 Fused':     fused_scores,
        'D: Tier 2 PPI RWR':   network_scores
    }


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-SEED RUN
# ─────────────────────────────────────────────────────────────────────────────

def run_single_seed(positive_df, all_drugs_df, model, seed):
    """Run the complete ablation evaluation for one random seed."""
    print(f"\n🎲 Seed {seed}...")

    negative_df = sample_negative_pairs(positive_df, all_drugs_df, seed)
    eval_df     = pd.concat([positive_df, negative_df], ignore_index=True)
    y_true      = eval_df['label'].values

    scores_by_condition = compute_scores_for_all_conditions(eval_df, model)

    seed_results = {}
    for condition_name, y_scores in scores_by_condition.items():
        metrics = calculate_ddi_metrics(y_true, y_scores, ef_fraction=EF_FRACTION)
        seed_results[condition_name] = metrics

    return seed_results


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(all_seed_results):
    """Compute mean ± std across N_SEEDS for each condition and metric."""
    conditions = list(all_seed_results[0].keys())
    metric_keys = list(all_seed_results[0][conditions[0]].keys())

    aggregated = {}
    for condition in conditions:
        aggregated[condition] = {}
        for metric in metric_keys:
            values = [run[condition][metric] for run in all_seed_results]
            aggregated[condition][metric] = {
                'mean': round(np.mean(values), 4),
                'std':  round(np.std(values), 4),
            }

    return aggregated


def print_aggregated_results(aggregated):
    """Print the final ablation study results table."""
    conditions  = list(aggregated.keys())
    metric_keys = list(aggregated[conditions[0]].keys())

    col_width = 22
    header = f"{'Condition':<25}" + "".join(f"{m:>{col_width}}" for m in metric_keys)

    print("\n")
    print("=" * 90)
    print("   ABLATION STUDY RESULTS  —  mean ± std across {:d} random seeds".format(N_SEEDS))
    print("=" * 90)
    print(header)
    print("-" * 90)

    for condition, metrics in aggregated.items():
        row = f"{condition:<25}"
        for metric in metric_keys:
            mean = metrics[metric]['mean']
            std  = metrics[metric]['std']
            cell = f"{mean:.4f} ± {std:.4f}"
            row += f"{cell:>{col_width}}"
        print(row)

    print("=" * 90)

    # Highlight the best condition per metric
    print("\n🏆 Best condition per metric:")
    for metric in metric_keys:
        best_condition = max(aggregated, key=lambda c: aggregated[c][metric]['mean'])
        best_mean = aggregated[best_condition][metric]['mean']
        print(f"   {metric:<12}: {best_condition} ({best_mean:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_study():
    print("\n" + "=" * 60)
    print("   SIMILARITY MODEL ABLATION STUDY")
    print("   Chemical vs Phenotypic vs Fused")
    print("=" * 60)

    db_engine = create_engine(settings.DB_URL)

    print("\n📦 Loading AdvancedFusionModel & PPINetworkModel...")
    model = AdvancedFusionModel()
    model.ppi_model = PPINetworkModel()  # Attach it temporarily for the eval script

    print("\n📥 Loading positive DDI pairs from database...")
    positive_df  = load_positive_pairs(db_engine)
    all_drugs_df = load_all_drugs(db_engine)

    if len(positive_df) == 0:
        print("❌ No positive pairs found in the database. Check database connection.")
        return

    all_seed_results = []
    total_start = time.perf_counter()

    for seed in range(N_SEEDS):
        seed_result = run_single_seed(positive_df, all_drugs_df, model, seed)
        all_seed_results.append(seed_result)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n⏱  Total evaluation time: {total_elapsed:.2f}s across {N_SEEDS} seeds")

    aggregated = aggregate_results(all_seed_results)
    print_aggregated_results(aggregated)


if __name__ == "__main__":
    run_ablation_study()
