"""Thin subprocess wrapper around the AutoDock Vina command-line executable.

AutoDock Vina is NOT installed by this project -- there is no official PyPI
wheel for Windows (the `vina` package on PyPI only ships Linux/macOS wheels
and requires Boost to build from source). Install it yourself:

    https://github.com/ccsb-scripps/AutoDock-Vina/releases

then either put `vina.exe` on PATH or set the `VINA_PATH` environment
variable / pass `vina_path=` explicitly. Everything in this module fails with
a clear `DockingUnavailableError` if the executable can't be found, rather
than silently fabricating a result.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from docking_ai.config import VINA_PATH

_RESULT_ROW_RE = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$")


class DockingUnavailableError(RuntimeError):
    """Raised when the AutoDock Vina executable cannot be located."""


class DockingRunError(RuntimeError):
    """Raised when Vina runs but exits with an error."""


@dataclass
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float


@dataclass
class DockingPose:
    mode: int
    affinity_kcal_mol: float
    rmsd_lb: float
    rmsd_ub: float


@dataclass
class DockingResult:
    receptor: str
    ligand: str
    poses: list[DockingPose]
    out_pdbqt: str
    raw_stdout: str

    @property
    def best_affinity_kcal_mol(self) -> float | None:
        return self.poses[0].affinity_kcal_mol if self.poses else None


def find_vina_executable(vina_path: str | None = None) -> str:
    candidate = vina_path or VINA_PATH
    resolved = shutil.which(candidate)
    if resolved is None:
        raise DockingUnavailableError(
            f"AutoDock Vina executable not found (looked for {candidate!r} on PATH). "
            "Install it from https://github.com/ccsb-scripps/AutoDock-Vina/releases and "
            "either add vina.exe to PATH or set the VINA_PATH environment variable."
        )
    return resolved


def _parse_vina_stdout(stdout: str) -> list[DockingPose]:
    poses = []
    for line in stdout.splitlines():
        m = _RESULT_ROW_RE.match(line)
        if m:
            mode, affinity, rmsd_lb, rmsd_ub = m.groups()
            poses.append(
                DockingPose(
                    mode=int(mode),
                    affinity_kcal_mol=float(affinity),
                    rmsd_lb=float(rmsd_lb),
                    rmsd_ub=float(rmsd_ub),
                )
            )
    return poses


def run_vina(
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str | Path,
    box: DockingBox,
    out_pdbqt: str | Path,
    exhaustiveness: int = 8,
    num_modes: int = 9,
    seed: int = 42,
    vina_path: str | None = None,
    timeout_s: int = 600,
) -> DockingResult:
    """Run `vina` on one receptor/ligand pair and parse the resulting poses.

    Raises DockingUnavailableError if the executable is missing, or
    DockingRunError if Vina exits non-zero.
    """
    exe = find_vina_executable(vina_path)

    receptor_pdbqt = Path(receptor_pdbqt)
    ligand_pdbqt = Path(ligand_pdbqt)
    out_pdbqt = Path(out_pdbqt)
    out_pdbqt.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        exe,
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(box.center_x),
        "--center_y", str(box.center_y),
        "--center_z", str(box.center_z),
        "--size_x", str(box.size_x),
        "--size_y", str(box.size_y),
        "--size_z", str(box.size_z),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--seed", str(seed),
        "--out", str(out_pdbqt),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise DockingRunError(
            f"vina exited with code {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    poses = _parse_vina_stdout(proc.stdout)
    return DockingResult(
        receptor=str(receptor_pdbqt),
        ligand=str(ligand_pdbqt),
        poses=poses,
        out_pdbqt=str(out_pdbqt),
        raw_stdout=proc.stdout,
    )
