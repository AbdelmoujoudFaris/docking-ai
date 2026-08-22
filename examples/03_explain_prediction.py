"""Quickstart: explain a GNN prediction for one bundled molecule.

No download needed -- trains on the bundled synthetic labels, then explains
the prediction for aspirin (one of the bundled EXAMPLE_MOLECULES) with both
attribution methods and renders a 2D atom-importance image.

    .\\.venv\\Scripts\\python.exe examples\\03_explain_prediction.py
"""

from __future__ import annotations

from docking_ai.config import EXPLAIN_DIR
from docking_ai.data.sample_data import EXAMPLE_MOLECULES, SYNTHETIC_DATA_DISCLAIMER, build_synthetic_dataset
from docking_ai.explain.gnn_explain import explain_molecule, render_atom_importance
from docking_ai.training.train import TrainConfig, train


def main() -> None:
    print(f"[example 03] {SYNTHETIC_DATA_DISCLAIMER}")

    df = build_synthetic_dataset()
    cfg = TrainConfig(task="regression", label_col="synthetic_score", run_name="example_explain")
    result = train(df=df, cfg=cfg)

    smiles = EXAMPLE_MOLECULES["aspirin"]
    for method in ("gradient", "occlusion"):
        exp = explain_molecule(result["checkpoint_path"], smiles, method=method)
        print(f"[example 03] {method} prediction: {exp['prediction']:.4f}")
        print(f"[example 03] {method} atom importances: {[round(v, 3) for v in exp['atom_importances']]}")

        out_path = EXPLAIN_DIR / f"example_aspirin_{method}.png"
        render_atom_importance(exp["smiles"], exp["atom_importances"], out_path)
        print(f"[example 03] wrote {out_path}")


if __name__ == "__main__":
    main()
