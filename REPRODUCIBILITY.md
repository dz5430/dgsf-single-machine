# Reproducibility Guide

This guide describes how to reproduce the single-machine DGSF workflow from a
fresh clone after downloading the external data/model artifacts.

## 1. Clone the Repository

```bash
git clone https://github.com/dz5430/dgsf-single-machine.git
cd dgsf-single-machine
```

## 2. Create the Environment

The experiments were run with Python 3.10 and the package versions listed in
`environment.yml`.

```bash
conda env create -f environment.yml
conda activate scheduling_env
```

Verify the Python dependencies:

```bash
python -c "import numpy, pandas, scipy, sklearn, pyomo, torch, matplotlib, openpyxl; print('environment ok')"
```

The MIP models require Gurobi through Pyomo. Install Gurobi separately, activate
a valid license, and make sure it is visible from the activated environment:

```bash
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('gurobi').available(False))"
```

The command should print `True`.

## 3. Download and Place External Artifacts

Large experiment artifacts are stored outside git. Download the artifact bundle
from the project Google Drive folder linked in the manuscript materials, then
place files under the paths listed in `docs/google_drive_manifest.md`.

At minimum, a reproduced pretrained-model run needs:

```text
Data/Trained Models/Dev3_30_theta_max_6Itau.pth
Data/Trained Models/Dev3_30_theta_0max_40tau_avg.pth
Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx
```

## 4. Run a Pretrained DGSF Model

Run model inference and local-swap refinement:

```bash
python Code/2_dgsf_main/Load_ML_swap.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --model "Data/Trained Models/Dev3_30_theta_max_6Itau.pth" \
  --output Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_ml_swap.xlsx
```

The output workbook contains the predicted ML rank vector, ML tardiness, local
swap rank vector, and local swap tardiness.

## 5. Run MIP Post-Processing

Use the local-swap sequence as the warm-start/window seed for the continuous-time
MIP postprocessor:

```bash
python Code/2_dgsf_main/Solver_MIP_ct_post.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_ml_swap.xlsx \
  --output Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_mip_post.xlsx \
  --time-limit 60
```

The output workbook adds `rank_mip_post` and `tardiness_mip_post`.

## 6. Evaluate Gaps and Rank Similarity

```bash
python Code/3_evaluation/Schedule_evaluation.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_mip_post.xlsx \
  --method-obj-col tardiness_mip_post \
  --ref-obj-col tardiness_dtime \
  --pred-rank-col rank_mip_post \
  --ref-rank-col rank_vector_dtime
```

The reported normalized tardiness gap is:

```text
gap (%) = 100 * (TT_method - TT_opt) / TT_opt
```

## 7. Evaluate Static Heuristic Baselines

```bash
python Code/3_evaluation/Dispatching_heuristics.py \
  --input Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --output Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_heuristics.xlsx
```

This script computes static/global EDD, SPT, LSF, ATC, and PRTT priority-rule
baselines. The revised manuscript distinguishes these static priority baselines
from recursive dispatching-rule baselines.

## 8. Generate Features or Reference Solutions

If starting from raw generated instances, first construct the ML input features:

```bash
python Code/1_data_processing/Update_input_features.py \
  --input Data/Raw_file.xlsx \
  --output Data/Updated_file.xlsx
```

Reference solutions can be generated with the time-indexed MIP:

```bash
python Code/1_data_processing/Solver_discrete_time.py \
  --input Data/Updated_file.xlsx \
  --output Results/Updated_file_with_reference.xlsx
```

## 9. Optional Retraining

Training requires a solved training file with all per-job features and reference
rank vectors.

```bash
python Code/2_dgsf_main/DeepSets_SMS_Scheduling_training.py \
  --input Data/Training_file.csv \
  --output-model "Data/Trained Models/Model_name.pth" \
  --loss-plot Results/loss_curves.png \
  --device cuda \
  --epochs 1000
```

Use `--device cpu` if CUDA is unavailable.

## 10. Code Archive

For manuscript reproducibility, archive the final GitHub release through Zenodo
or an equivalent DOI-minting archive and cite the DOI-linked release in the
revised manuscript. Record the exact git commit hash associated with the
archived release.
