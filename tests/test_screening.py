import pytest

from docking_ai.data.sample_data import EXAMPLE_MOLECULES, build_synthetic_dataset
from docking_ai.docking.vina_wrapper import DockingBox, DockingUnavailableError
from docking_ai.screening.screen import dock_top_k, score_library
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
        run_name="pytest_screening",
    )
    result = train(df=df, cfg=cfg)
    return result["checkpoint_path"]


def test_score_library_ranks_descending(tiny_checkpoint):
    smiles_list = list(EXAMPLE_MOLECULES.values())[:20]
    ranked = score_library(tiny_checkpoint, smiles_list)
    scores = ranked["predicted_score"].tolist()
    assert scores == sorted(scores, reverse=True)
    assert ranked["rank"].tolist() == list(range(1, len(ranked) + 1))


def test_score_library_drops_invalid_smiles(tiny_checkpoint):
    smiles_list = ["CCO", "not a smiles!!", "c1ccccc1"]
    ranked = score_library(tiny_checkpoint, smiles_list)
    assert len(ranked) == 2


def test_dock_top_k_fails_fast_without_vina(tiny_checkpoint, tmp_path):
    import shutil

    if shutil.which("vina") is not None:
        pytest.skip("AutoDock Vina is installed on PATH in this environment")

    smiles_list = list(EXAMPLE_MOLECULES.values())[:5]
    ranked = score_library(tiny_checkpoint, smiles_list)
    box = DockingBox(0, 0, 0, 20, 20, 20)
    with pytest.raises(DockingUnavailableError):
        dock_top_k(ranked, receptor_pdbqt=str(tmp_path / "receptor.pdbqt"), box=box, k=2, run_name="pytest_dock")
