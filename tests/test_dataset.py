from docking_ai.data.dataset import MoleculeDataset, collate_molecules, get_scaffold, scaffold_split
from docking_ai.data.sample_data import build_synthetic_dataset


def test_build_synthetic_dataset_has_expected_columns():
    df = build_synthetic_dataset()
    assert len(df) > 30
    for col in ("smiles", "synthetic_score", "synthetic_active", "mol_wt", "log_p"):
        assert col in df.columns
    assert df.attrs["disclaimer"] == "SYNTHETIC DATA — SOFTWARE TESTING ONLY"


def test_molecule_dataset_drops_invalid_and_builds_tensors():
    smiles = ["CCO", "c1ccccc1", "not a smiles!!"]
    labels = [1.0, 2.0, 3.0]
    ds = MoleculeDataset(smiles, labels)
    assert len(ds) == 2  # invalid one dropped

    x, adj, label = ds[0]
    assert x.ndim == 2
    assert adj.ndim == 3
    assert label.ndim == 0


def test_collate_pads_to_batch_max():
    smiles = ["CCO", "c1ccccc1"]  # 3 atoms, 6 atoms
    labels = [1.0, 2.0]
    ds = MoleculeDataset(smiles, labels)
    batch = [ds[i] for i in range(len(ds))]
    x, adj, mask, labels_out = collate_molecules(batch)
    assert x.shape[1] == 6
    assert mask.sum(dim=1).tolist() == [3.0, 6.0]
    assert adj.shape[-1] == 6


def test_scaffold_split_no_scaffold_leakage():
    df = build_synthetic_dataset()
    smiles = df["smiles"].tolist()
    train_idx, val_idx, test_idx = scaffold_split(smiles, seed=0)

    assert len(train_idx) + len(val_idx) + len(test_idx) == len(smiles)
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).isdisjoint(test_idx)
    assert set(val_idx).isdisjoint(test_idx)

    train_scaffolds = {get_scaffold(smiles[i]) for i in train_idx}
    val_scaffolds = {get_scaffold(smiles[i]) for i in val_idx}
    test_scaffolds = {get_scaffold(smiles[i]) for i in test_idx}
    assert train_scaffolds.isdisjoint(val_scaffolds)
    assert train_scaffolds.isdisjoint(test_scaffolds)
    assert val_scaffolds.isdisjoint(test_scaffolds)
