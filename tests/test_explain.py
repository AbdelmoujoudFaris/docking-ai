from pathlib import Path

import pytest

from docking_ai.data.sample_data import build_synthetic_dataset
from docking_ai.explain.gnn_explain import explain_molecule, render_atom_importance
from docking_ai.training.train import TrainConfig, train


@pytest.fixture(scope="module")
def tiny_checkpoint():
    df = build_synthetic_dataset()
    cfg = TrainConfig(
        task="regression",
        label_col="synthetic_score",
        hidden_dim=16,
        num_layers=2,
        max_epochs=5,
        patience=5,
        batch_size=8,
        run_name="pytest_explain",
    )
    result = train(df=df, cfg=cfg)
    return result["checkpoint_path"]


def test_gradient_explanation_shape(tiny_checkpoint):
    out = explain_molecule(tiny_checkpoint, "CC(=O)Oc1ccccc1C(=O)O", method="gradient")  # aspirin
    assert len(out["atom_importances"]) == 13  # aspirin has 13 heavy atoms
    assert isinstance(out["prediction"], float)


def test_occlusion_explanation_shape(tiny_checkpoint):
    out = explain_molecule(tiny_checkpoint, "CCO", method="occlusion")
    assert len(out["atom_importances"]) == 3


def test_render_atom_importance_writes_png(tiny_checkpoint, tmp_path):
    out = explain_molecule(tiny_checkpoint, "CCO", method="gradient")
    png_path = render_atom_importance(out["smiles"], out["atom_importances"], tmp_path / "ethanol.png")
    assert png_path.exists()
    assert png_path.stat().st_size > 0
