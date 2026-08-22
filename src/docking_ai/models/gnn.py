"""A small relational graph convolutional network (R-GCN style) implemented
with dense adjacency matrices (no torch_geometric / torch-scatter dependency,
which keeps installation trivial and portable across platforms including
Windows). Fine for the small, drug-like molecules this project targets
(typically well under a few hundred atoms per graph).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGCNLayer(nn.Module):
    """One message-passing step with a separate learned transform per bond
    type (single/double/triple/aromatic), plus a self-loop transform.
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int):
        super().__init__()
        self.num_relations = num_relations
        self.rel_weight = nn.Parameter(torch.empty(num_relations, in_dim, out_dim))
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.rel_weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, in_dim)
        adj: (B, C, N, N) per-relation binary adjacency (C == num_relations)
        mask: (B, N)
        """
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)  # (B, C, N, 1)
        agg = torch.einsum("bcnm,bmf->bcnf", adj, x) / deg  # (B, C, N, in_dim)
        rel_out = torch.einsum("bcnf,cfo->bno", agg, self.rel_weight)  # (B, N, out_dim)
        out = rel_out + self.self_linear(x) + self.bias
        out = out * mask.unsqueeze(-1)
        return out


class MolecularGNN(nn.Module):
    """Graph -> scalar (or multi-class logits) predictor.

    task="regression": output_dim should be 1, forward() returns (B,) scores.
    task="classification": output_dim == num_classes, forward() returns (B, C) logits.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        num_relations: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        output_dim: int = 1,
        dropout: float = 0.1,
        task: str = "regression",
    ):
        super().__init__()
        assert task in ("regression", "classification")
        self.task = task
        self.output_dim = output_dim

        dims = [atom_feature_dim] + [hidden_dim] * num_layers
        self.layers = nn.ModuleList(
            [RelationalGCNLayer(dims[i], dims[i + 1], num_relations) for i in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def encode_nodes(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return final-layer per-atom embeddings (B, N, hidden_dim), masked."""
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, adj, mask))
            h = self.dropout(h)
            h = h * mask.unsqueeze(-1)
        return h

    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean pooling over atoms -> (B, hidden_dim)."""
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.encode_nodes(x, adj, mask)
        graph_emb = self.pool(h, mask)
        out = self.head(graph_emb)
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out
