# Examples

Runnable scripts that exercise the pipeline end-to-end using only the
bundled example molecules (`docking_ai.data.sample_data`) -- **no dataset
download required**. Labels on these molecules are synthetic (see the main
README's "SCIENTIFIC INTEGRITY" section); the point of these scripts is to
demonstrate the code paths, not to produce real bioactivity claims.

Run any of them directly with the project's venv, from the project root:

```powershell
cd C:\Users\faris-c\Documents\APK\Pytroch
.\.venv\Scripts\python.exe examples\01_train_and_evaluate.py
.\.venv\Scripts\python.exe examples\02_virtual_screening.py
.\.venv\Scripts\python.exe examples\03_explain_prediction.py
.\.venv\Scripts\python.exe examples\04_ligand_prep_no_docking.py
```

| Script | What it shows | Output |
| --- | --- | --- |
| `01_train_and_evaluate.py` | Train a regression GNN, then evaluate it | checkpoint + metrics report + plot in `outputs/` |
| `02_virtual_screening.py` | Train a classifier, rank the bundled library by predicted activity | ranked table printed to console |
| `03_explain_prediction.py` | Gradient and occlusion atom-importance for aspirin's prediction | annotated molecule PNGs in `outputs/explain/` |
| `04_ligand_prep_no_docking.py` | SMILES -> 3D conformer -> AutoDock-ready PDBQT (no Vina needed) | `.pdbqt` files in `outputs/docking/` |

Real AutoDock Vina docking (`dock_top_k` / `run_vina`) needs an external Vina
binary and a receptor structure you prepare yourself -- that step is
deliberately not bundled (see the main README). Everything else above runs
immediately after `pip install -e .`, with no internet access or data
download.
