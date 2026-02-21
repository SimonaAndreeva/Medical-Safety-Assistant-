import os
import shutil
from pathlib import Path

def restructure_project():
    project_root = Path(__file__).parent
    src_dir = project_root / "src"
    
    # 1. Define the Tiered Architecture
    new_dirs = [
        src_dir / "data" / "raw",
        src_dir / "data" / "processed",
        src_dir / "models" / "tier_1_similarity",
        src_dir / "models" / "tier_2_network",
        src_dir / "models" / "tier_3_hin",
        src_dir / "models" / "tier_4_gnn",
        src_dir / "web"
    ]
    
    print("Creating new directory structure...")
    for d in new_dirs:
        d.mkdir(parents=True, exist_ok=True)
        # Create __init__.py to make them proper Python packages
        (d / "__init__.py").touch(exist_ok=True)
        print(f"  Created: {d.relative_to(project_root)}")

    # 2. Define File Movements (Old Path -> New Path)
    # This maps your current files to their new semantic locations
    moves = {
        src_dir / "algorithms" / "similarity_metrics.py": src_dir / "models" / "tier_1_similarity" / "chemical_sim.py",
        src_dir / "algorithms" / "rwr.py": src_dir / "models" / "tier_2_network" / "rwr.py",
        src_dir / "pipelines" / "build_hin_network.py": src_dir / "models" / "tier_3_hin" / "matrix_builder.py",
        src_dir / "web" / "hin_service.py": src_dir / "models" / "tier_3_hin" / "hin_model.py"
    }

    print("\nMoving files to new architecture...")
    for old_path, new_path in moves.items():
        if old_path.exists():
            shutil.move(str(old_path), str(new_path))
            print(f"  Moved: {old_path.name} -> {new_path.relative_to(project_root)}")
        else:
            print(f"  Skip: {old_path.relative_to(project_root)} (Not found)")

    # 3. Cleanup empty directories
    dirs_to_remove = [
        src_dir / "algorithms",
        src_dir / "pipelines"
    ]
    
    print("\nCleaning up old directories...")
    for d in dirs_to_remove:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
            print(f"  Removed empty directory: {d.relative_to(project_root)}")
            
    print("\nRefactoring complete. You must now fix your import statements.")

if __name__ == "__main__":
    restructure_project()