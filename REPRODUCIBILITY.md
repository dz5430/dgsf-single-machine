# Reproducibility Guide

This guide describes how to reproduce the single-machine DGSF workflow from a
fresh clone after downloading the external data archive. The archive is
organized to merge directly into the repository; preserve its `Data/` and
`Results/` subfolders when downloading.

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

The MIP models require Gurobi through Pyomo. Results were produced with Gurobi
10.0.1. Install Gurobi separately, activate a valid license, and make sure it
is visible from the activated environment:

```bash
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('gurobi').available(False))"
```

The command should print `True`.

## 3. Download and Place External Artifacts

Large experiment artifacts are stored outside git. Download the complete data
archive and merge it into the repository root, preserving the archive folder
structure. The exact paths are listed in `docs/artifact_manifest.md`.

Artifact archive:
https://drive.google.com/drive/u/2/folders/1Lo8WRZabBUxGNA0nOD50TMavkKyhDwHP

At minimum, the pretrained-model example needs:

```text
Data/Trained Models/30_theta_max_6Itau_Dev3_50k_dev9_lean.pth
Results/F1/input/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx
```

## 4. Run the Supplied Dev9-Lean DGSF Model

Run model inference and local-swap refinement:

```bash
python Code/2_dgsf_main/evaluate_sms_model.py \
  --input Results/F1/input/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --model "Data/Trained Models/30_theta_max_6Itau_Dev3_50k_dev9_lean.pth" \
  --architecture dev9_lean \
  --device cpu \
  --output Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf.xlsx
```

The output workbook contains the predicted ML rank vector, ML tardiness, local
swap rank vector, and local swap tardiness. The output path matches the Drive
`Results/F1/output/F1_DGSF/` folder.

## 5. Run MIP Post-Processing

Use the local-swap sequence as the warm-start/window seed for the continuous-time
MIP postprocessor:

```bash
python Code/2_dgsf_main/Solver_MIP_ct_post.py \
  --input Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf.xlsx \
  --output Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf_mip.xlsx \
  --time-limit 60
```

The output workbook adds the continuous-time MIP rank, tardiness, solve-time,
and termination columns used by the evaluation script.

## 6. Evaluate Gaps and Rank Similarity

```bash
python Code/3_evaluation/Schedule_evaluation.py \
  --input Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf_mip.xlsx \
  --method-obj-col tardiness_dgsf_mip \
  --ref-obj-col tardiness_dtime \
  --pred-rank-col rank_dgsf_mip \
  --ref-rank-col rank_vector_dtime
```

The reported normalized tardiness gap is:

```text
gap (%) = 100 * (TT_method - TT_opt) / TT_opt
```

## 7. Evaluate Static Heuristic Baselines

```bash
python Code/3_evaluation/Dispatching_heuristics.py \
  --input Results/F1/input/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --output Results/F1/output/F1_Recursive/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_heuristics.xlsx
```

This script computes static/global EDD, SPT, LSF, ATC, and PRTT priority-rule
baselines. The revised manuscript distinguishes these static priority baselines
from recursive dispatching-rule baselines.

## 8. Generate Features or Reference Solutions for New Instance Types

This preprocessing is required when training or evaluating a new instance type
from raw generated instances. It is not required to reproduce the reported
manuscript tables if the processed workbooks from the data archive are
downloaded.
First construct the ML input features:

```bash
python Code/1_data_processing/Update_input_features.py \
  --input Results/F1/input/Raw_file.xlsx \
  --output Results/F1/input/Updated_file.xlsx
```

Reference solutions can be generated with the time-indexed MIP:

```bash
python Code/1_data_processing/Solver_discrete_time.py \
  --input Results/F1/input/Updated_file.xlsx \
  --output Results/F1/output/F1_DGSF/Updated_file_with_reference.xlsx
```

## 9. Train on a New Instance Type

Training is optional for reproducing the manuscript, but is the standard path
for adding a new instance type. Prepare a solved training table containing the
per-job feature columns, `rank_vector_dtime`, `tau`, `eps`, `rho`, and
`tardiness_dtime`. The table may be CSV or Excel; pass its path with `--input`.

After generating features and reference solutions as in Section 8, train the
current Dev9-Lean model:

```bash
python Code/2_dgsf_main/DeepSets_SMS_Scheduling_training_dev9_lean.py \
  --input Data/<instance_type>_train.csv \
  --output-model "Data/Trained Models/<instance_type>_dev9_lean.pth" \
  --metrics-csv Data/<instance_type>_dev9_lean_metrics.csv \
  --loss-plot Data/<instance_type>_dev9_lean_loss.png \
  --device cuda \
  --epochs 1000
```

Use `--device cpu` if CUDA is unavailable. Use the resulting checkpoint with
`evaluate_sms_model.py` as in Section 4, replacing the input workbook and model
paths for the new instance type.

## 10. Reproduce the Max-Tardiness Diagnostic

The two diagnostic workbooks are in the Drive folder
`Results/F1/Max Tardiness Evaluation/`. The capped workbook's stored
`rank_swap` column is the sequence produced by the DGSF local-swap step.
The following command recreates the capped time-indexed MIP references and
capped DGSF post-processing solutions for threshold multipliers 10, 12, and
15:

```bash
python Code/3_evaluation/Max_tardiness_diagnostic.py \
  --input "Results/F1/Max Tardiness Evaluation/Table6_capped_d6_t120.xlsx" \
  --output "Results/F1/Max Tardiness Evaluation/Table6_capped_d6_t120_recreated.xlsx" \
  --quiet
```

The script is deliberately fixed to the paper configuration: rank window
`delta = 6` and a 120-second MIP post-processing limit. The unconstrained
baseline is retained in the supplied workbook.
