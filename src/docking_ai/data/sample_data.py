"""Small example dataset used to exercise the pipeline end-to-end.

The molecules listed below (`EXAMPLE_MOLECULES`) are real, well-known chemical
structures (common drugs, metabolites, and simple aromatics/aliphatics) drawn
from public chemistry knowledge. Any SMILES that fails to parse/sanitize is
silently dropped, so occasional transcription mistakes do not break the
pipeline.

The regression/classification *labels* attached to them are NOT experimental
measurements. There is no bundled real bioactivity/binding-affinity dataset in
this project. Labels here are a deterministic function of real RDKit
descriptors plus fixed-seed noise, used only so the ML pipeline (training,
evaluation, screening, explainability) has something concrete to run against.

    SYNTHETIC DATA — SOFTWARE TESTING ONLY

To use this project for an actual research question, replace
`build_synthetic_dataset()` with a loader for a real labeled dataset (e.g. a
ChEMBL/BindingDB extract or your own assay results) that yields the same
`(smiles, label)` schema consumed by `docking_ai.data.dataset`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

SYNTHETIC_DATA_DISCLAIMER = "SYNTHETIC DATA — SOFTWARE TESTING ONLY"

# name -> SMILES. Real chemical structures; see module docstring re: labels.
EXAMPLE_MOLECULES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "nicotine": "CN1CCCC1c1cccnc1",
    "benzene": "c1ccccc1",
    "toluene": "Cc1ccccc1",
    "phenol": "Oc1ccccc1",
    "ethanol": "CCO",
    "acetone": "CC(=O)C",
    "warfarin": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O",
    "diazepam": "CN1c2ccc(Cl)cc2C(=NCC1=O)c1ccccc1",
    "naproxen": "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "metformin": "CN(C)C(=N)NC(=N)N",
    "omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)c1nc2ccc(OC)cc2[nH]1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "dopamine": "NCCc1ccc(O)c(O)c1",
    "adrenaline": "CNCC(O)c1ccc(O)c(O)c1",
    "ascorbic_acid": "OCC(O)C1OC(=O)C(O)=C1O",
    "salicylic_acid": "OC(=O)c1ccccc1O",
    "benzocaine": "CCOC(=O)c1ccc(N)cc1",
    "lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
    "procaine": "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "methyl_salicylate": "COC(=O)c1ccccc1O",
    "urea": "NC(=O)N",
    "glycine": "NCC(=O)O",
    "alanine": "CC(N)C(=O)O",
    "phenylalanine": "NC(Cc1ccccc1)C(=O)O",
    "tryptophan": "NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "tyrosine": "NC(Cc1ccc(O)cc1)C(=O)O",
    "histamine": "NCCc1c[nH]cn1",
    "adenine": "Nc1ncnc2[nH]cnc12",
    "guanine": "Nc1nc2[nH]cnc2c(=O)[nH]1",
    "cytosine": "Nc1cc[nH]c(=O)n1",
    "thymine": "Cc1c[nH]c(=O)[nH]c1=O",
    "uracil": "O=c1cc[nH]c(=O)[nH]1",
    "xanthine": "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
    "theobromine": "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
    "theophylline": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "ephedrine": "CNC(C)C(O)c1ccccc1",
    "menthol": "CC(C)C1CCC(C)CC1O",
    "camphor": "CC1(C)C2CCC1(C)C(=O)C2",
    "vanillin": "COc1cc(C=O)ccc1O",
    "cinnamaldehyde": "O=CC=Cc1ccccc1",
    "limonene": "CC(=C)C1CCC(C)=CC1",
    "sulfanilamide": "Nc1ccc(cc1)S(=O)(=O)N",
    "formaldehyde": "C=O",
    "methanol": "CO",
    "acetic_acid": "CC(=O)O",
    "propanol": "CCCO",
    "butane": "CCCC",
    "cyclohexane": "C1CCCCC1",
    "pyridine": "c1ccncc1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "imidazole": "c1cnc[nH]1",
    "indole": "c1ccc2[nH]ccc2c1",
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "styrene": "C=Cc1ccccc1",
    "aniline": "Nc1ccccc1",
    "benzaldehyde": "O=Cc1ccccc1",
    "benzoic_acid": "OC(=O)c1ccccc1",
    "nitrobenzene": "[O-][N+](=O)c1ccccc1",
    "chlorobenzene": "Clc1ccccc1",
    "bromobenzene": "Brc1ccccc1",
    "fluorobenzene": "Fc1ccccc1",
    "iodobenzene": "Ic1ccccc1",
    "p_xylene": "Cc1ccc(C)cc1",
    "o_xylene": "Cc1ccccc1C",
    "m_xylene": "Cc1cccc(C)c1",
    "ethylbenzene": "CCc1ccccc1",
    "cumene": "CC(C)c1ccccc1",
    "acetanilide": "CC(=O)Nc1ccccc1",
    "resorcinol": "Oc1cccc(O)c1",
    "catechol": "Oc1ccccc1O",
    "hydroquinone": "Oc1ccc(O)cc1",
}


def _descriptor_row(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "smiles": Chem.MolToSmiles(mol),
        "mol_wt": Descriptors.MolWt(mol),
        "log_p": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "h_donors": rdMolDescriptors.CalcNumHBD(mol),
        "h_acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
    }


def build_synthetic_dataset(seed: int = 42) -> pd.DataFrame:
    """Build the example (smiles, synthetic label) table.

    Returns a DataFrame with columns: smiles, mol_wt, log_p, tpsa, h_donors,
    h_acceptors, rot_bonds, ring_count, synthetic_score, synthetic_active.

    `synthetic_score` / `synthetic_active` are SYNTHETIC — see module
    docstring. They are a fixed linear combination of the real RDKit
    descriptors plus fixed-seed Gaussian noise, used only to exercise the
    training/evaluation/screening/explainability code paths.
    """
    rows = []
    for name, smiles in EXAMPLE_MOLECULES.items():
        row = _descriptor_row(smiles)
        if row is None:
            continue
        row["name"] = name
        rows.append(row)

    df = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    z = lambda col: (df[col] - df[col].mean()) / (df[col].std() + 1e-8)  # noqa: E731

    signal = (
        1.0 * z("log_p")
        - 0.7 * z("tpsa")
        + 0.4 * z("ring_count")
        - 0.3 * z("rot_bonds")
        - 0.2 * z("h_donors")
    )
    noise = rng.normal(loc=0.0, scale=0.35, size=len(df))
    df["synthetic_score"] = signal + noise
    df["synthetic_active"] = (df["synthetic_score"] > df["synthetic_score"].median()).astype(int)

    df.attrs["disclaimer"] = SYNTHETIC_DATA_DISCLAIMER
    return df.reset_index(drop=True)
