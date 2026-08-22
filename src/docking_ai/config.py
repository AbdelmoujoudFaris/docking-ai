"""Central paths and constants for the docking_ai project."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUTS_DIR / "logs"
MODELS_DIR = OUTPUTS_DIR / "models"
DOCKING_DIR = OUTPUTS_DIR / "docking"
SCREENING_DIR = OUTPUTS_DIR / "screening"
EXPLAIN_DIR = OUTPUTS_DIR / "explain"

CONFIGS_DIR = PROJECT_ROOT / "configs"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR, MODELS_DIR, DOCKING_DIR, SCREENING_DIR, EXPLAIN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Path to the AutoDock Vina executable. AutoDock Vina is NOT bundled with this
# project (no official pip wheel exists for Windows). Install it separately from
# https://vina.scripps.edu/ or https://github.com/ccsb-scripps/AutoDock-Vina/releases
# and either place `vina.exe` on PATH or set the VINA_PATH environment variable.
VINA_PATH = os.environ.get("VINA_PATH", "vina")

RANDOM_SEED = 42
