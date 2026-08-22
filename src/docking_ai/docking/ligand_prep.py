"""SMILES -> 3D conformer -> AutoDock PDBQT ligand file, via RDKit + meeko.

meeko (https://github.com/forlilab/Meeko) is a pure-Python/RDKit tool that
replaces the old MGLTools `prepare_ligand4.py` Perl/Python2 script for
generating AutoDock4/Vina-compatible PDBQT files, so no external binary is
required for ligand preparation.
"""

from __future__ import annotations

from pathlib import Path

from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AllChem


class LigandPrepError(RuntimeError):
    pass


def embed_3d(mol, seed: int = 42, max_attempts: int = 5):
    """Add hydrogens, generate a 3D conformer, and optimize with MMFF (falling
    back to UFF if MMFF parameters are unavailable for this molecule).
    """
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        # retry with random coordinates as a fallback for awkward topologies
        params.useRandomCoords = True
        cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise LigandPrepError("RDKit conformer embedding failed for this molecule")

    for attempt in range(max_attempts):
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=2000)
            break
        except Exception:
            if attempt == max_attempts - 1:
                raise LigandPrepError("Force-field optimization failed") from None
    return mol


def smiles_to_pdbqt(smiles: str, out_path: str | Path, seed: int = 42) -> Path:
    """Convert a SMILES string to a docking-ready ligand PDBQT file.

    Returns the output path. Raises LigandPrepError on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise LigandPrepError(f"RDKit could not parse SMILES: {smiles!r}")

    mol_3d = embed_3d(mol, seed=seed)

    preparator = MoleculePreparation()
    setups = preparator.prepare(mol_3d)
    if not setups:
        raise LigandPrepError("meeko produced no molecule setups")

    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(setups[0])
    if not is_ok:
        raise LigandPrepError(f"meeko failed to write PDBQT: {err_msg}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(pdbqt_string)
    return out_path
