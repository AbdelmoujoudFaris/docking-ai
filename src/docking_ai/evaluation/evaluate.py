"""Evaluate a trained checkpoint on an arbitrary (smiles, label) DataFrame and
write a JSON metrics report + a predicted-vs-actual (or ROC) plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from docking_ai.config import LOGS_DIR
from docking_ai.data.dataset import MoleculeDataset, collate_molecules
from docking_ai.training.metrics import classification_metrics, regression_metrics
from docking_ai.training.train import load_model_from_checkpoint


@torch.no_grad()
def predict(model, dataset: MoleculeDataset, device: str = "cpu", batch_size: int = 32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_molecules)
    all_true, all_pred = [], []
    model.eval()
    for x, adj, mask, labels in loader:
        x, adj, mask = x.to(device), adj.to(device), mask.to(device)
        out = model(x, adj, mask)
        if model.task == "classification":
            out = torch.sigmoid(out)
        all_true.extend(labels.numpy().tolist())
        all_pred.extend(out.cpu().numpy().tolist())
    return all_true, all_pred


def evaluate_checkpoint(
    ckpt_path: str,
    df,
    label_col: str,
    run_name: str = "eval",
    device: str = "cpu",
) -> dict:
    model, ckpt = load_model_from_checkpoint(ckpt_path, device=device)
    dataset = MoleculeDataset(df["smiles"].tolist(), df[label_col].tolist())

    y_true, y_pred = predict(model, dataset, device=device)

    if model.task == "classification":
        metrics = classification_metrics(y_true, y_pred)
    else:
        metrics = regression_metrics(y_true, y_pred)

    report = {
        "checkpoint": str(ckpt_path),
        "n_samples": len(dataset),
        "metrics": metrics,
    }
    report_path = LOGS_DIR / f"{run_name}_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[evaluate] wrote report to {report_path}")

    _plot(model.task, y_true, y_pred, LOGS_DIR / f"{run_name}_eval_plot.png")
    return report


def _plot(task: str, y_true, y_pred, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    if task == "classification":
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax.plot(fpr, tpr, label="ROC curve")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
    else:
        ax.scatter(y_true, y_pred, alpha=0.7)
        lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
        ax.plot(lims, lims, linestyle="--", color="gray")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Predicted vs. true")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] saved plot to {out_path}")
