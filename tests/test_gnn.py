import torch

from docking_ai.data.dataset import MoleculeDataset, collate_molecules
from docking_ai.data.molecules import ATOM_FEATURE_DIM, NUM_BOND_TYPES
from docking_ai.models.gnn import MolecularGNN


def _make_batch():
    ds = MoleculeDataset(["CCO", "c1ccccc1", "CC(=O)O"], [0.0, 0.0, 0.0])
    batch = [ds[i] for i in range(len(ds))]
    return collate_molecules(batch)


def test_regression_forward_shape():
    x, adj, mask, labels = _make_batch()
    model = MolecularGNN(ATOM_FEATURE_DIM, NUM_BOND_TYPES, hidden_dim=16, num_layers=2, output_dim=1, task="regression")
    out = model(x, adj, mask)
    assert out.shape == (x.shape[0],)


def test_classification_forward_shape():
    x, adj, mask, labels = _make_batch()
    model = MolecularGNN(
        ATOM_FEATURE_DIM, NUM_BOND_TYPES, hidden_dim=16, num_layers=2, output_dim=1, task="classification"
    )
    out = model(x, adj, mask)
    assert out.shape == (x.shape[0],)


def test_padding_atoms_do_not_affect_output():
    """Two identical molecules with different amounts of trailing zero-padding
    (achieved by embedding in differently-sized batches) must produce the same
    prediction, since padded atoms are masked out at every layer and pooling.
    """
    model = MolecularGNN(ATOM_FEATURE_DIM, NUM_BOND_TYPES, hidden_dim=16, num_layers=2, output_dim=1)
    model.eval()

    ds = MoleculeDataset(["CCO"], [0.0])
    x, adj, label = ds[0]

    # batch of 1: no padding
    x1 = x.unsqueeze(0)
    adj1 = adj.unsqueeze(0)
    mask1 = torch.ones(1, x.shape[0])
    out1 = model(x1, adj1, mask1)

    # manually pad to 10 atoms
    n_pad = 10
    x2 = torch.zeros(1, n_pad, x.shape[1])
    x2[0, : x.shape[0]] = x
    adj2 = torch.zeros(1, adj.shape[0], n_pad, n_pad)
    adj2[0, :, : x.shape[0], : x.shape[0]] = adj
    mask2 = torch.zeros(1, n_pad)
    mask2[0, : x.shape[0]] = 1.0
    out2 = model(x2, adj2, mask2)

    assert torch.allclose(out1, out2, atol=1e-5)


def test_gradients_flow_to_all_parameters():
    x, adj, mask, labels = _make_batch()
    model = MolecularGNN(ATOM_FEATURE_DIM, NUM_BOND_TYPES, hidden_dim=16, num_layers=2, output_dim=1)
    out = model(x, adj, mask)
    loss = (out - labels).pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
