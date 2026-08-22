"""RDKit-based molecular featurization: SMILES -> atom/bond graph tensors.

The featurization is a standard hand-crafted atom/bond descriptor scheme
(similar to those used in MoleculeNet-style GNN baselines): one-hot atom
identity/degree/charge/hybridization/ring/H-count/chirality, plus a
per-bond-type (single/double/triple/aromatic) adjacency stack so the GNN can
learn distinct propagation weights per bond type without needing a sparse
message-passing library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem

# --- Atom feature vocabulary -------------------------------------------------

ATOM_SYMBOLS = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"]  # + "other"
DEGREES = [0, 1, 2, 3, 4, 5]  # last bucket also catches >5
FORMAL_CHARGES = [-2, -1, 0, 1, 2]  # + "other"
HYBRIDIZATIONS = [
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]  # + "other"
NUM_HS = [0, 1, 2, 3, 4]  # last bucket also catches >4
CHIRAL_TAGS = [
    rdchem.ChiralType.CHI_UNSPECIFIED,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]  # + "other"

ATOM_FEATURE_DIM = (
    (len(ATOM_SYMBOLS) + 1)
    + len(DEGREES)  # _bucket has no "other" slot: last bucket catches overflow
    + (len(FORMAL_CHARGES) + 1)
    + (len(HYBRIDIZATIONS) + 1)
    + 1  # aromatic
    + len(NUM_HS)  # _bucket has no "other" slot: last bucket catches overflow
    + 1  # in ring
    + (len(CHIRAL_TAGS) + 1)
)

BOND_TYPES = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]
NUM_BOND_TYPES = len(BOND_TYPES)


def _one_hot(value, choices) -> list:
    vec = [0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)  # "other" bucket
    vec[idx] = 1
    return vec


def _bucket(value: int, buckets: list) -> list:
    vec = [0] * len(buckets)
    idx = min(value, buckets[-1])
    vec[buckets.index(idx) if idx in buckets else len(buckets) - 1] = 1
    return vec


def atom_features(atom: rdchem.Atom) -> np.ndarray:
    feats = []
    feats += _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
    feats += _bucket(atom.GetDegree(), DEGREES)
    feats += _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
    feats += _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
    feats += [1 if atom.GetIsAromatic() else 0]
    feats += _bucket(atom.GetTotalNumHs(), NUM_HS)
    feats += [1 if atom.IsInRing() else 0]
    feats += _one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
    return np.array(feats, dtype=np.float32)


@dataclass
class MolGraph:
    smiles: str
    x: np.ndarray  # (N, ATOM_FEATURE_DIM)
    adj: np.ndarray  # (NUM_BOND_TYPES, N, N), binary, no self-loops
    num_atoms: int

    def is_empty(self) -> bool:
        return self.num_atoms == 0


def smiles_to_graph(smiles: str, sanitize: bool = True) -> Optional[MolGraph]:
    """Parse a SMILES string into a MolGraph. Returns None on parse failure."""
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        return None
    return mol_to_graph(mol, smiles=Chem.MolToSmiles(mol))


def mol_to_graph(mol: rdchem.Mol, smiles: Optional[str] = None) -> MolGraph:
    n = mol.GetNumAtoms()
    x = np.zeros((n, ATOM_FEATURE_DIM), dtype=np.float32)
    for atom in mol.GetAtoms():
        x[atom.GetIdx()] = atom_features(atom)

    adj = np.zeros((NUM_BOND_TYPES, n, n), dtype=np.float32)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        channel = BOND_TYPES.index(bt) if bt in BOND_TYPES else BOND_TYPES.index(rdchem.BondType.SINGLE)
        adj[channel, i, j] = 1.0
        adj[channel, j, i] = 1.0

    return MolGraph(
        smiles=smiles if smiles is not None else Chem.MolToSmiles(mol),
        x=x,
        adj=adj,
        num_atoms=n,
    )
