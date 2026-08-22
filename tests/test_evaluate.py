from pathlib import Path

import pytest

from docking_ai.data.sample_data import build_synthetic_dataset
from docking_ai.evaluation.evaluate import evaluate_checkpoint
from docking_ai.training.train import TrainConfig, train


@pytest.fixture(scope="module")
def tiny_checkpoint_and_df():
    df = build_synthetic_dataset()
    cfg = TrainConfig(
        task="regression",
        label_col="synthetic_score",
        hidden_dim=16,
        num_layers=2,
        max_epochs=5,
        patience=5,
        batch_size=8,
        run_name="pytest_eval",
    )
    result = train(df=df, cfg=cfg)
    return result["checkpoint_path"], df


def test_evaluate_checkpoint_writes_report_and_plot(tiny_checkpoint_and_df):
    ckpt_path, df = tiny_checkpoint_and_df
    report = evaluate_checkpoint(ckpt_path, df, label_col="synthetic_score", run_name="pytest_eval_report")
    assert "rmse" in report["metrics"]
    assert report["n_samples"] == len(df)

    from docking_ai.config import LOGS_DIR

    assert (LOGS_DIR / "pytest_eval_report_eval_report.json").exists()
    assert (LOGS_DIR / "pytest_eval_report_eval_plot.png").exists()
