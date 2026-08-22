import pytest

from docking_ai.docking.ligand_prep import LigandPrepError, smiles_to_pdbqt
from docking_ai.docking.receptor_prep import ReceptorPrepError, validate_receptor_pdbqt
from docking_ai.docking.vina_wrapper import DockingUnavailableError, find_vina_executable


def test_smiles_to_pdbqt_writes_file(tmp_path):
    out_path = tmp_path / "ethanol.pdbqt"
    result = smiles_to_pdbqt("CCO", out_path)
    assert result.exists()
    text = result.read_text()
    assert "ROOT" in text
    assert "ATOM" in text


def test_smiles_to_pdbqt_invalid_smiles_raises(tmp_path):
    with pytest.raises(LigandPrepError):
        smiles_to_pdbqt("not a smiles!!", tmp_path / "bad.pdbqt")


def test_validate_receptor_pdbqt_rejects_short_lines(tmp_path):
    bad_file = tmp_path / "receptor.pdbqt"
    bad_file.write_text("ATOM      1  N   VAL A   1      31.904  20.564  33.219\n")
    with pytest.raises(ReceptorPrepError):
        validate_receptor_pdbqt(bad_file)


def test_validate_receptor_pdbqt_accepts_well_formed_line(tmp_path):
    good_file = tmp_path / "receptor.pdbqt"
    good_file.write_text(
        "ATOM      1  N   VAL A   1      31.904  20.564  33.219  1.00 30.00     0.174 N \n"
    )
    summary = validate_receptor_pdbqt(good_file)
    assert summary["n_atoms"] == 1


def test_validate_receptor_pdbqt_missing_file_raises(tmp_path):
    with pytest.raises(ReceptorPrepError):
        validate_receptor_pdbqt(tmp_path / "does_not_exist.pdbqt")


def test_find_vina_executable_raises_when_not_installed():
    """AutoDock Vina is not installed in this environment (no Windows pip
    wheel exists -- see README). We assert the wrapper fails loudly with a
    clear, actionable error rather than silently fabricating docking results.
    If Vina *is* installed on this machine's PATH, this test is skipped.
    """
    import shutil

    if shutil.which("vina") is not None:
        pytest.skip("AutoDock Vina is installed on PATH in this environment")

    with pytest.raises(DockingUnavailableError):
        find_vina_executable("vina")
