"""PyTorch Dataset for molecular graphs + Bemis-Murcko scaffold splitting."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import Dataset

from docking_ai.data.molecules import MolGraph, NUM_BOND_TYPES, smiles_to_graph


def get_scaffold(smiles: str) -> str:
    """Bemis-Murcko generic scaffold SMILES for a molecule (empty string if none)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        scaffold = ""
    return scaffold


def scaffold_split(
    smiles_list: Sequence[str],
    frac_train: float = 0.8,
    frac_val: float = 0.1,
    frac_test: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Group molecules by Bemis-Murcko scaffold, then greedily assign whole
    scaffold groups to train/val/test so that no scaffold appears in more than
    one split (standard MoleculeNet-style scaffold split -> a harder, more
    realistic generalization test than a random split).
    """
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6

    scaffold_to_indices: dict[str, list[int]] = defaultdict(list)
    for idx, smiles in enumerate(smiles_list):
        scaffold_to_indices[get_scaffold(smiles)].append(idx)

    rng = np.random.default_rng(seed)
    groups = list(scaffold_to_indices.values())
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in order]
    groups.sort(key=len, reverse=True)  # largest scaffold groups placed first

    n_total = len(smiles_list)
    n_train_target = frac_train * n_total
    n_val_target = frac_val * n_total

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for group in groups:
        if len(train_idx) + len(group) <= n_train_target or not train_idx:
            train_idx.extend(group)
        elif len(val_idx) + len(group) <= n_val_target or not val_idx:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    return train_idx, val_idx, test_idx


class MoleculeDataset(Dataset):
    """Wraps a list of SMILES + numeric labels as molecular graph tensors.

    Invalid/unparsable SMILES are dropped at construction time (a warning
    count is printed). Use `.smiles` / `.labels` to inspect what survived.
    """

    def __init__(self, smiles_list: Sequence[str], labels: Sequence[float]):
        assert len(smiles_list) == len(labels)
        self.graphs: list[MolGraph] = []
        self.smiles: list[str] = []
        self.labels: list[float] = []

        n_dropped = 0
        for smi, label in zip(smiles_list, labels):
            graph = smiles_to_graph(smi)
            if graph is None or graph.is_empty():
                n_dropped += 1
                continue
            self.graphs.append(graph)
            self.smiles.append(smi)
            self.labels.append(float(label))

        if n_dropped:
            print(f"[MoleculeDataset] dropped {n_dropped} unparsable SMILES out of {len(smiles_list)}")

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int):
        graph = self.graphs[idx]
        x = torch.from_numpy(graph.x)
        adj = torch.from_numpy(graph.adj)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, adj, label

    def subset(self, indices: Sequence[int]) -> "MoleculeDataset":
        sub = MoleculeDataset.__new__(MoleculeDataset)
        sub.graphs = [self.graphs[i] for i in indices]
        sub.smiles = [self.smiles[i] for i in indices]
        sub.labels = [self.labels[i] for i in indices]
        return sub


def collate_molecules(batch):
    """Pad a batch of (x, adj, label) graphs to the batch's max atom count.

    Returns:
        x: (B, N_max, F) padded atom features
        adj: (B, C, N_max, N_max) padded per-bond-type adjacency
        mask: (B, N_max) 1.0 for real atoms, 0.0 for padding
        labels: (B,)
    """
    feat_dim = batch[0][0].shape[1]
    n_max = max(x.shape[0] for x, _, _ in batch)
    b = len(batch)

    x_out = torch.zeros(b, n_max, feat_dim, dtype=torch.float32)
    adj_out = torch.zeros(b, NUM_BOND_TYPES, n_max, n_max, dtype=torch.float32)
    mask_out = torch.zeros(b, n_max, dtype=torch.float32)
    labels_out = torch.zeros(b, dtype=torch.float32)

    for i, (x, adj, label) in enumerate(batch):
        n = x.shape[0]
        x_out[i, :n] = x
        adj_out[i, :, :n, :n] = adj
        mask_out[i, :n] = 1.0
        labels_out[i] = label

    return x_out, adj_out, mask_out, labels_out
