"""Per-atom explainability for MolecularGNN predictions.

Two independent, simple, well-understood methods (no extra dependency like
captum/shap needed):

  * gradient x input saliency  -- fast, one backward pass, standard first
    baseline for GNN/CNN attribution.
  * occlusion (leave-one-atom-out) -- slower (one forward pass per atom) but
    model-agnostic and easier to sanity-check: literally "how much does the
    prediction change if this atom is silenced".

Both return one importance score per (real, non-padding) atom, and can be
rendered onto the 2D molecule depiction for a human-readable explanation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from docking_ai.data.molecules import smiles_to_graph
from docking_ai.training.train import load_model_from_checkpoint


def gradient_atom_importance(model, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Gradient x input saliency for a single molecule (batch size 1).

    Returns an (N,) array, one score per atom (padding excluded by the caller
    slicing to num_atoms).
    """
    model.eval()
    x = x.clone().requires_grad_(True)
    out = model(x, adj, mask)
    scalar = out.sum()
    scalar.backward()
    grad = x.grad.detach()
    importance = (grad * x.detach()).sum(dim=-1).squeeze(0)  # (N,)
    return importance.numpy()


@torch.no_grad()
def occlusion_atom_importance(model, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """Leave-one-atom-out occlusion importance for a single molecule (batch size 1).

    importance[i] = baseline_prediction - prediction_with_atom_i_silenced
    A positive value means atom i was contributing positively to the score.
    """
    model.eval()
    baseline = model(x, adj, mask).item()
    n = int(mask.sum().item())
    importances = np.zeros(n, dtype=np.float32)

    for i in range(n):
        x_occ = x.clone()
        adj_occ = adj.clone()
        x_occ[0, i, :] = 0.0
        adj_occ[0, :, i, :] = 0.0
        adj_occ[0, :, :, i] = 0.0
        occluded_pred = model(x_occ, adj_occ, mask).item()
        importances[i] = baseline - occluded_pred

    return importances


def explain_molecule(ckpt_path: str, smiles: str, method: str = "gradient", device: str = "cpu") -> dict:
    """Explain a single molecule's prediction. `method` is "gradient" or "occlusion"."""
    assert method in ("gradient", "occlusion")

    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
    graph = smiles_to_graph(smiles)
    if graph is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    x = torch.from_numpy(graph.x).unsqueeze(0)
    adj = torch.from_numpy(graph.adj).unsqueeze(0)
    mask = torch.ones(1, graph.num_atoms, dtype=torch.float32)

    with torch.no_grad():
        pred = model(x, adj, mask).item()
        if model.task == "classification":
            pred = float(torch.sigmoid(torch.tensor(pred)))

    if method == "gradient":
        importances = gradient_atom_importance(model, x, adj, mask)[: graph.num_atoms]
    else:
        importances = occlusion_atom_importance(model, x, adj, mask)

    return {
        "smiles": graph.smiles,
        "prediction": pred,
        "method": method,
        "atom_importances": importances.tolist(),
    }


def render_atom_importance(smiles: str, atom_importances: list[float], out_path: str | Path, size=(500, 500)) -> Path:
    """Render the molecule with atoms colored by (signed, normalized) importance:
    red = pushes prediction up, blue = pushes prediction down.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    importances = np.array(atom_importances, dtype=np.float32)
    max_abs = max(float(np.abs(importances).max()), 1e-8)
    normalized = importances / max_abs  # in [-1, 1]

    highlight_atoms = list(range(mol.GetNumAtoms()))
    colors = {}
    for i, val in enumerate(normalized):
        val = float(val)
        if val >= 0:
            colors[i] = (1.0, 1.0 - val, 1.0 - val)  # white -> red
        else:
            colors[i] = (1.0 + val, 1.0 + val, 1.0)  # white -> blue

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol, highlightAtoms=highlight_atoms, highlightAtomColors=colors
    )
    drawer.FinishDrawing()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(drawer.GetDrawingText())
    return out_path
