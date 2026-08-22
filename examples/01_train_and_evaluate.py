"""Quickstart: train + evaluate a GNN on the bundled example molecules.

No download needed -- `build_synthetic_dataset()` ships ~77 real molecules
with synthetic labels (see docking_ai.data.sample_data docstring). Runs on
CPU in well under a minute.

    .\\.venv\\Scripts\\python.exe examples\\01_train_and_evaluate.py
"""

from __future__ import annotations

from docking_ai.data.sample_data import SYNTHETIC_DATA_DISCLAIMER, build_synthetic_dataset
from docking_ai.evaluation.evaluate import evaluate_checkpoint
from docking_ai.training.train import TrainConfig, train


def main() -> None:
    print(f"[example 01] {SYNTHETIC_DATA_DISCLAIMER}")

    df = build_synthetic_dataset()
    print(f"[example 01] loaded {len(df)} bundled molecules")

    cfg = TrainConfig(task="regression", label_col="synthetic_score", run_name="example_regression")
    result = train(df=df, cfg=cfg)
    print(f"[example 01] test metrics: {result['test_metrics']}")
    print(f"[example 01] checkpoint: {result['checkpoint_path']}")

    report = evaluate_checkpoint(
        ckpt_path=result["checkpoint_path"],
        df=df,
        label_col=cfg.label_col,
        run_name="example_regression",
    )
    print(f"[example 01] full-dataset eval report: {report}")


if __name__ == "__main__":
    main()
