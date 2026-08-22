"""Quickstart: SMILES -> 3D -> docking-ready ligand PDBQT.

No download needed -- ligand preparation is pure RDKit (conformer embedding +
force-field optimization) + meeko, no external binary required. This is the
one piece of the docking pipeline that works without installing AutoDock
Vina or supplying a receptor structure; see the README's "AutoDock Vina
setup" section for what real docking additionally requires.

    .\\.venv\\Scripts\\python.exe examples\\04_ligand_prep_no_docking.py
"""

from __future__ import annotations

from docking_ai.config import DOCKING_DIR
from docking_ai.data.sample_data import EXAMPLE_MOLECULES
from docking_ai.docking.ligand_prep import LigandPrepError, smiles_to_pdbqt


def main() -> None:
    names = ["aspirin", "caffeine", "ibuprofen"]
    for name in names:
        smiles = EXAMPLE_MOLECULES[name]
        out_path = DOCKING_DIR / f"example_{name}.pdbqt"
        try:
            smiles_to_pdbqt(smiles, out_path)
            print(f"[example 04] {name}: wrote {out_path}")
        except LigandPrepError as e:
            print(f"[example 04] {name}: failed ({e})")


if __name__ == "__main__":
    main()
