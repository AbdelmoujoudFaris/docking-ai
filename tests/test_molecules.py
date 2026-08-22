from docking_ai.data.molecules import ATOM_FEATURE_DIM, NUM_BOND_TYPES, smiles_to_graph


def test_smiles_to_graph_shapes():
    graph = smiles_to_graph("CCO")  # ethanol, 3 heavy atoms
    assert graph is not None
    assert graph.num_atoms == 3
    assert graph.x.shape == (3, ATOM_FEATURE_DIM)
    assert graph.adj.shape == (NUM_BOND_TYPES, 3, 3)


def test_smiles_to_graph_bonds_symmetric():
    graph = smiles_to_graph("CCO")
    assert (graph.adj == graph.adj.transpose(0, 2, 1)).all()
    assert graph.adj.sum() == 2 * 2  # 2 single bonds, symmetric entries


def test_invalid_smiles_returns_none():
    assert smiles_to_graph("not a smiles!!") is None


def test_aromatic_ring_uses_aromatic_channel():
    from docking_ai.data.molecules import BOND_TYPES
    from rdkit.Chem import rdchem

    graph = smiles_to_graph("c1ccccc1")  # benzene
    aromatic_channel = BOND_TYPES.index(rdchem.BondType.AROMATIC)
    assert graph.adj[aromatic_channel].sum() > 0
