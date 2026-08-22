"""Training loop: scaffold-split molecular dataset -> MolecularGNN.

Run directly for a smoke-test training run on the bundled synthetic example
dataset:

    python -m docking_ai.training.train
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from docking_ai.config import LOGS_DIR, MODELS_DIR, RANDOM_SEED
from docking_ai.data.dataset import MoleculeDataset, collate_molecules, scaffold_split
from docking_ai.data.molecules import ATOM_FEATURE_DIM, NUM_BOND_TYPES
from docking_ai.data.sample_data import build_synthetic_dataset
from docking_ai.models.gnn import MolecularGNN
from docking_ai.training.metrics import classification_metrics, regression_metrics


@dataclass
class TrainConfig:
    task: str = "regression"  # "regression" or "classification"
    label_col: str = "synthetic_score"
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 16
    max_epochs: int = 200
    patience: int = 25
    seed: int = RANDOM_SEED
    device: str = "cpu"
    run_name: str = "gnn_run"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(df, cfg: TrainConfig):
    smiles = df["smiles"].tolist()
    labels = df[cfg.label_col].tolist()
    dataset = MoleculeDataset(smiles, labels)

    train_idx, val_idx, test_idx = scaffold_split(dataset.smiles, seed=cfg.seed)
    train_ds = dataset.subset(train_idx)
    val_ds = dataset.subset(val_idx)
    test_ds = dataset.subset(test_idx)

    loaders = {
        "train": DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_molecules),
        "val": DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_molecules),
        "test": DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_molecules),
    }
    sizes = {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)}
    return dataset, loaders, sizes


def build_model(cfg: TrainConfig) -> MolecularGNN:
    output_dim = 1
    return MolecularGNN(
        atom_feature_dim=ATOM_FEATURE_DIM,
        num_relations=NUM_BOND_TYPES,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        output_dim=output_dim,
        dropout=cfg.dropout,
        task=cfg.task,
    )


def run_epoch(model, loader, optimizer, loss_fn, device, train: bool):
    model.train(train)
    total_loss, n = 0.0, 0
    all_true, all_pred = [], []
    for x, adj, mask, labels in loader:
        x, adj, mask, labels = x.to(device), adj.to(device), mask.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            out = model(x, adj, mask)
            loss = loss_fn(out, labels)
            if train:
                loss.backward()
                optimizer.step()
        bsz = x.size(0)
        total_loss += loss.item() * bsz
        n += bsz
        all_true.extend(labels.detach().cpu().numpy().tolist())
        if model.task == "classification":
            probs = torch.sigmoid(out)
            all_pred.extend(probs.detach().cpu().numpy().tolist())
        else:
            all_pred.extend(out.detach().cpu().numpy().tolist())
    avg_loss = total_loss / max(n, 1)
    return avg_loss, all_true, all_pred


def train(df=None, cfg: TrainConfig | None = None) -> dict:
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    if df is None:
        df = build_synthetic_dataset(seed=cfg.seed)

    dataset, loaders, sizes = build_dataloaders(df, cfg)
    print(f"[train] dataset sizes: {sizes}")

    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss() if cfg.task == "classification" else nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    log_path = LOGS_DIR / f"{cfg.run_name}_history.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])

        for epoch in range(1, cfg.max_epochs + 1):
            train_loss, _, _ = run_epoch(model, loaders["train"], optimizer, loss_fn, device, train=True)
            val_loss, val_true, val_pred = run_epoch(model, loaders["val"], optimizer, loss_fn, device, train=False)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            writer.writerow([epoch, train_loss, val_loss])
            f.flush()

            if epoch == 1 or epoch % 10 == 0:
                print(f"[train] epoch {epoch:4d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.patience:
                    print(f"[train] early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_true, test_pred = run_epoch(model, loaders["test"], optimizer, loss_fn, device, train=False)
    if cfg.task == "classification":
        test_metrics = classification_metrics(test_true, test_pred)
    else:
        test_metrics = regression_metrics(test_true, test_pred)
    print(f"[train] test metrics: {test_metrics}")

    ckpt_path = MODELS_DIR / f"{cfg.run_name}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "atom_feature_dim": ATOM_FEATURE_DIM,
            "num_relations": NUM_BOND_TYPES,
            "test_metrics": test_metrics,
        },
        ckpt_path,
    )
    print(f"[train] saved checkpoint to {ckpt_path}")

    metrics_path = LOGS_DIR / f"{cfg.run_name}_test_metrics.json"
    metrics_path.write_text(json.dumps(test_metrics, indent=2))

    return {
        "checkpoint_path": str(ckpt_path),
        "history_path": str(log_path),
        "test_metrics": test_metrics,
        "best_val_loss": best_val_loss,
        "dataset_sizes": sizes,
    }


def load_model_from_checkpoint(ckpt_path: str | Path, device: str = "cpu") -> tuple[MolecularGNN, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
