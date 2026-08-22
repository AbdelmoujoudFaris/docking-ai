"""Quickstart: rank the bundled molecule library with a trained GNN.

No download needed -- trains a small classifier on the bundled synthetic
labels, then screens the same bundled SMILES list and prints the top hits.
Real Vina re-scoring (`dock_top_k`) needs an external receptor + AutoDock
Vina install, so it's shown here only as commented-out sample code.

    .\\.venv\\Scripts\\python.exe examples\\02_virtual_screening.py
"""

from __future__ import annotations

from docking_ai.data.sample_data import SYNTHETIC_DATA_DISCLAIMER, build_synthetic_dataset
from docking_ai.screening.screen import score_library
from docking_ai.training.train import TrainConfig, train


def main() -> None:
    print(f"[example 02] {SYNTHETIC_DATA_DISCLAIMER}")

    df = build_synthetic_dataset()
    cfg = TrainConfig(task="classification", label_col="synthetic_active", run_name="example_screening")
    result = train(df=df, cfg=cfg)
    print(f"[example 02] test metrics: {result['test_metrics']}")

    ranked = score_library(result["checkpoint_path"], df["smiles"].tolist())
    print("[example 02] top 10 ranked molecules:")
    print(ranked.head(10).to_string(index=False))

    # Real physics-based re-scoring of the top-K hits requires AutoDock Vina
    # and a prepared receptor PDBQT -- both external, not bundled. See the
    # README's "AutoDock Vina setup" section, then:
    #
    # from docking_ai.docking.vina_wrapper import DockingBox
    # from docking_ai.screening.screen import dock_top_k
    # box = DockingBox(center=(0, 0, 0), size=(20, 20, 20))
    # dock_top_k(ranked, receptor_pdbqt="path/to/receptor.pdbqt", box=box, k=5)


if __name__ == "__main__":
    main()
