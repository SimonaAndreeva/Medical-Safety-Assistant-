import os
import shutil
import re

# --- CONFIGURATION ---
DIRS_TO_CREATE = [
    "src/algorithms",
    "src/pipelines",
    "src/evaluation",
    "src/web",
    "data/processed", # Ensure this exists
]

MOVES = {
    "src/features/build_chemical.py": "src/pipelines/build_chemical_space.py",
    "src/features/build_network.py": "src/pipelines/build_ppi_network.py",
    "src/analysis/evaluate_rwr.py": "src/evaluation/metrics.py",
    "src/services/similarity.py": "src/web/services.py",
}

# --- 1. SETUP DIRECTORIES ---
print("📂 Creating scientific directory structure...")
for d in DIRS_TO_CREATE:
    os.makedirs(d, exist_ok=True)

# --- 2. INTELLIGENT SPLIT OF MATH.PY ---
# We need to split the old math.py into 'rwr.py' and 'similarity_metrics.py'
print("🧠 Splitting math.py into scientific components...")

math_path = "src/utils/math.py"
if os.path.exists(math_path):
    with open(math_path, "r") as f:
        content = f.read()

    # Create similarity_metrics.py
    sim_content = "import numpy as np\nfrom sklearn.metrics.pairwise import cosine_similarity\n\n"
    sim_content += "class SimilarityEngine:\n"
    sim_content += "    @staticmethod\n    def calculate_tanimoto" + content.split("def calculate_tanimoto")[1].split("@staticmethod")[0]
    # (Extract Cosine if present or add default)
    if "def calculate_cosine" in content:
         sim_content += "    @staticmethod\n    def calculate_cosine" + content.split("def calculate_cosine")[1].split("# ==")[0]
    
    with open("src/algorithms/similarity_metrics.py", "w") as f:
        f.write(sim_content)
    print("   -> Created src/algorithms/similarity_metrics.py")

    # Create rwr.py
    rwr_content = "import numpy as np\nfrom scipy.sparse import csr_matrix, diags\n\n"
    rwr_content += "class RWR:\n"
    # Extract build_transition_matrix
    if "def build_transition_matrix" in content:
        part = content.split("def build_transition_matrix")[1]
        # logic to grab until end or next function
        rwr_content += "    @staticmethod\n    def build_transition_matrix" + part
    
    # Quick fix for indentation if needed or manual check recommended
    with open("src/algorithms/rwr.py", "w") as f:
        f.write(rwr_content)
    print("   -> Created src/algorithms/rwr.py (Check indentation!)")

# --- 3. MOVE FILES ---
print("🚚 Moving pipeline and service files...")
for src, dst in MOVES.items():
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"   -> Moved {src} to {dst}")
    else:
        print(f"   ⚠️ Could not find {src}, skipping.")

# --- 4. UPDATE IMPORTS (The Hard Part) ---
print("electric_plug Updating code imports...")

replacements = [
    # Fix PPI Builder
    ("src/pipelines/build_ppi_network.py", "from src.utils.math import SimilarityEngine", "from src.algorithms.rwr import RWR\nfrom src.algorithms.similarity_metrics import SimilarityEngine"),
    ("src/pipelines/build_ppi_network.py", "SimilarityEngine.build_transition_matrix", "RWR.build_transition_matrix"),
    ("src/pipelines/build_ppi_network.py", "SimilarityEngine.calculate_rwr", "RWR.calculate_rwr"),
    
    # Fix Web Service
    ("src/web/services.py", "from src.utils.math import SimilarityEngine", "from src.algorithms.similarity_metrics import SimilarityEngine"),
    
    # Fix Evaluation Script
    ("src/evaluation/metrics.py", "from src.services.similarity import DrugSimilarityService", "from src.web.services import DrugSimilarityService"),
    
    # Fix Main.py
    ("main.py", "from src.services.similarity import DrugSimilarityService", "from src.web.services import DrugSimilarityService"),
]

for file_path, old, new in replacements:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            c = f.read()
        if old in c:
            c = c.replace(old, new)
            with open(file_path, "w") as f:
                f.write(c)
            print(f"   -> Updated imports in {file_path}")

print("\n✅ Refactoring Complete! Please manually check 'src/algorithms/rwr.py' indentation.")