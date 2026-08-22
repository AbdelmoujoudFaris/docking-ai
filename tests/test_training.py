from pathlib import Path

from docking_ai.data.sample_data import build_synthetic_dataset
from docking_ai.training.train import TrainConfig, load_model_from_checkpoint, train


def test_train_smoke_regression():
    df = build_synthetic_dataset()
    cfg = TrainConfig(
        task="regression",
        label_col="synthetic_score",
        hidden_dim=16,
        num_layers=2,
        max_epochs=8,
        patience=8,
        batch_size=8,
        run_name="pytest_regression",
    )
    result = train(df=df, cfg=cfg)

    assert Path(result["checkpoint_path"]).exists()
    assert "rmse" in result["test_metrics"]
    assert all(n > 0 for n in result["dataset_sizes"].values())

    model, ckpt = load_model_from_checkpoint(result["checkpoint_path"])
    assert ckpt["config"]["task"] == "regression"


def test_train_smoke_classification():
    df = build_synthetic_dataset()
    cfg = TrainConfig(
        task="classification",
        label_col="synthetic_active",
        hidden_dim=16,
        num_layers=2,
        max_epochs=8,
        patience=8,
        batch_size=8,
        run_name="pytest_classification",
    )
    result = train(df=df, cfg=cfg)

    assert Path(result["checkpoint_path"]).exists()
    assert "roc_auc" in result["test_metrics"]


def test_loss_decreases_over_training():
    import csv

    df = build_synthetic_dataset()
    cfg = TrainConfig(
        task="regression",
        label_col="synthetic_score",
        hidden_dim=16,
        num_layers=2,
        max_epochs=40,
        patience=40,
        batch_size=8,
        run_name="pytest_loss_curve",
    )
    result = train(df=df, cfg=cfg)

    with open(result["history_path"]) as f:
        rows = list(csv.DictReader(f))
    first_train_loss = float(rows[0]["train_loss"])
    last_train_loss = float(rows[-1]["train_loss"])
    assert last_train_loss < first_train_loss
