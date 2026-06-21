# DGSF Single-Machine Scheduling

Code and reproducibility files for the DeepSets-Guided Scheduling Framework
(DGSF) single-machine total tardiness experiments.

The model predicts a static job-priority vector for a non-preemptive
single-machine scheduling instance. The predicted sequence can then be improved
with local swap refinement and continuous-time MIP post-processing.

## Contents

- `Code/1_data_processing/`: instance generation, feature construction, and
  time-indexed MIP reference-solution scripts.
- `Code/2_dgsf_main/`: DeepSets training, ML inference with swap refinement,
  and continuous-time MIP post-processing.
- `Code/3_evaluation/`: dispatching-rule baselines, schedule evaluation, and
  feature-importance analysis.
- `Data/`: small product tables plus downloaded training/test artifacts and
  trained model checkpoints.
- `Results/`: downloaded or regenerated experiment outputs.
- `environment.yml`: conda environment used for the reproducibility workflow.
- `REPRODUCIBILITY.md`: step-by-step setup and command-line workflow.
- `docs/google_drive_manifest.md`: expected external artifacts and where to
  place them.

Large `.xlsx`, `.csv`, and `.pth` artifacts are not committed. Download them
from the project Google Drive folder and place them in `Data/` and `Results/`
as described in `docs/google_drive_manifest.md`.

Google Drive:
https://drive.google.com/drive/u/2/folders/1Lo8WRZabBUxGNA0nOD50TMavkKyhDwHP

## Setup

```bash
git clone https://github.com/dz5430/dgsf-single-machine.git
cd dgsf-single-machine
conda env create -f environment.yml
conda activate scheduling_env
```

Alternatively, in an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
```

The MIP scripts use Pyomo with Gurobi. Install Gurobi separately, activate a
valid license, and verify access:

```bash
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('gurobi').available(False))"
```

The command should print `True`.

## Data Layout

Place external artifacts as follows:

```text
Data/
    Facility Products/
    Trained Models/
Results/
    F1/
    F2/
    F3/
    F4/
    F5/
```

The expected filenames are listed in `docs/google_drive_manifest.md`.

## Run the Pipeline

The scripts can be run from the repository root.

Run a pretrained DeepSets model with local-swap refinement:

```bash
python Code/2_dgsf_main/Load_ML_swap.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --model "Data/Trained Models/Dev3_30_theta_max_6Itau.pth" \
  --output Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_ml_swap.xlsx
```

Run continuous-time MIP post-processing:

```bash
python Code/2_dgsf_main/Solver_MIP_ct_post.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_ml_swap.xlsx \
  --output Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_mip_post.xlsx \
  --time-limit 60
```

Evaluate a schedule against the reference solution:

```bash
python Code/3_evaluation/Schedule_evaluation.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_mip_post.xlsx \
  --method-obj-col tardiness_mip_post \
  --ref-obj-col tardiness_dtime \
  --pred-rank-col rank_mip_post \
  --ref-rank-col rank_vector_dtime
```

Evaluate static dispatching-rule baselines:

```bash
python Code/3_evaluation/Dispatching_heuristics.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --output Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_heuristics.xlsx
```

## Metric

Reported normalized tardiness gaps use the conventional optimality-gap
definition:

```text
gap (%) = 100 * (TT_method - TT_opt) / TT_opt
```

All reported benchmark summaries use instances with positive `TT_opt`.

## Full Reproducibility Notes

See `REPRODUCIBILITY.md` for the complete workflow, including feature
generation, reference-solution generation, optional retraining, and
post-processing.
