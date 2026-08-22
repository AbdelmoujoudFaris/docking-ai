"""Receptor-side helpers for AutoDock Vina docking.

Deliberately NOT included: automatic generation of a receptor PDBQT from an
arbitrary raw PDB file. Correct receptor preparation (choosing protonation
states, handling missing residues/loops, deciding what to do with waters,
cofactors, and metal ions, picking a rigid vs. flexible-residue treatment)
requires structural-biology judgment calls that should not be silently
automated -- getting them wrong silently produces confidently-wrong docking
scores. Prepare your receptor PDBQT with one of:

  * ADFRsuite `prepare_receptor` (https://ccsb.scripps.edu/adfr/) -- the
    modern, actively maintained tool, or
  * MGLTools `prepare_receptor4.py` (legacy, Python 2)

and pass the resulting .pdbqt path into `vina_wrapper`. This module only
validates that file and helps derive a search-box from a reference ligand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem


class ReceptorPrepError(RuntimeError):
    pass


def validate_receptor_pdbqt(path: str | Path) -> dict:
    """Sanity-check a receptor PDBQT file. Raises ReceptorPrepError if it
    doesn't look usable; otherwise returns a small summary dict.
    """
    path = Path(path)
    if not path.exists():
        raise ReceptorPrepError(f"Receptor PDBQT not found: {path}")

    text = path.read_text(errors="ignore")
    atom_lines = [ln for ln in text.splitlines() if ln.startswith(("ATOM", "HETATM"))]
    if not atom_lines:
        raise ReceptorPrepError(f"No ATOM/HETATM records found in {path}; not a valid PDBQT")

    missing_charge = sum(1 for ln in atom_lines if len(ln) < 70)
    if missing_charge:
        raise ReceptorPrepError(
            f"{missing_charge}/{len(atom_lines)} atom records in {path} look too short to "
            "carry AutoDock partial-charge/atom-type columns -- this file was probably not "
            "run through prepare_receptor / prepare_receptor4.py"
        )

    return {"path": str(path), "n_atoms": len(atom_lines)}


def box_from_ligand(ligand_path: str | Path, padding: float = 4.0) -> dict:
    """Derive a Vina search box (center + size, in Angstrom) from a reference
    ligand structure file (.pdb/.mol/.sdf/.pdbqt-as-pdb), e.g. a
    co-crystallized ligand near the binding site of interest.

    This is the standard "box around a known/reference ligand" approach.
    `padding` is added on every side of the ligand's coordinate bounding box.
    """
    ligand_path = Path(ligand_path)
    suffix = ligand_path.suffix.lower()
    if suffix == ".sdf":
        mol = next(Chem.SDMolSupplier(str(ligand_path), removeHs=False))
    elif suffix in (".mol", ".mol2"):
        mol = Chem.MolFromMolFile(str(ligand_path), removeHs=False)
    else:
        mol = Chem.MolFromPDBFile(str(ligand_path), removeHs=False)

    if mol is None or mol.GetNumConformers() == 0:
        raise ReceptorPrepError(f"Could not read a 3D structure with coordinates from {ligand_path}")

    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    center = (mins + maxs) / 2.0
    size = (maxs - mins) + 2 * padding

    return {
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "center_z": float(center[2]),
        "size_x": float(size[0]),
        "size_y": float(size[1]),
        "size_z": float(size[2]),
    }
