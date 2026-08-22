"""AI-driven virtual screening: rank a library of candidate molecules with a
trained GNN, optionally hand the top-K off to AutoDock Vina for physics-based
re-scoring.

This is a triage tool, not a source of truth: GNN scores here are only as
good as the (in this repo's default configuration, synthetic -- see
`docking_ai.data.sample_data`) labels the model was trained on. Docking
affinities from `run_vina` are real physics-based estimates *if* you supply a
real, properly-prepared receptor; this module never fabricates either.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from docking_ai.config import SCREENING_DIR, VINA_PATH
from docking_ai.data.dataset import MoleculeDataset, collate_molecules
from docking_ai.docking.ligand_prep import LigandPrepError, smiles_to_pdbqt
from docking_ai.docking.vina_wrapper import DockingBox, DockingUnavailableError, find_vina_executable, run_vina
from docking_ai.training.train import load_model_from_checkpoint
from torch.utils.data import DataLoader


@torch.no_grad()
def score_library(ckpt_path: str, smiles_list: list[str], device: str = "cpu", batch_size: int = 64) -> pd.DataFrame:
    """Predict a score for every molecule in `smiles_list` using a trained
    checkpoint. Returns a DataFrame sorted by descending predicted score,
    with unparsable SMILES dropped.
    """
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)

    dummy_labels = [0.0] * len(smiles_list)
    dataset = MoleculeDataset(smiles_list, dummy_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)

    scores = []
    for x, adj, mask, _ in loader:
        x, adj, mask = x.to(device), adj.to(device), mask.to(device)
        out = model(x, adj, mask)
        if model.task == "classification":
            out = torch.sigmoid(out)
        scores.extend(out.cpu().numpy().tolist())

    df = pd.DataFrame({"smiles": dataset.smiles, "predicted_score": scores})
    df = df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    return df


def dock_top_k(
    ranked_df: pd.DataFrame,
    receptor_pdbqt: str,
    box: DockingBox,
    k: int = 5,
    run_name: str = "screen",
    vina_path: str | None = None,
) -> pd.DataFrame:
    """Re-score the top-k GNN hits with real AutoDock Vina docking.

    Raises DockingUnavailableError immediately (before touching any ligand)
    if the Vina executable isn't installed, so callers can catch it and fall
    back to GNN-only ranking instead of silently proceeding.
    """
    find_vina_executable(vina_path)  # fail fast, before doing ligand prep work

    out_dir = SCREENING_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    top = ranked_df.head(k).copy()
    affinities = []
    errors = []
    for i, row in top.iterrows():
        smi = row["smiles"]
        try:
            lig_path = smiles_to_pdbqt(smi, out_dir / f"ligand_{i}.pdbqt")
            result = run_vina(
                receptor_pdbqt=receptor_pdbqt,
                ligand_pdbqt=lig_path,
                box=box,
                out_pdbqt=out_dir / f"docked_{i}.pdbqt",
                vina_path=vina_path,
            )
            affinities.append(result.best_affinity_kcal_mol)
            errors.append(None)
        except LigandPrepError as e:
            affinities.append(None)
            errors.append(f"ligand_prep_error: {e}")

    top["vina_best_affinity_kcal_mol"] = affinities
    top["docking_error"] = errors

    report_path = out_dir / "docking_report.json"
    report_path.write_text(top.to_json(orient="records", indent=2))
    print(f"[screen] wrote docking report to {report_path}")
    return top
