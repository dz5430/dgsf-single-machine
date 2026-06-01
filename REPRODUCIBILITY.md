# Reproducibility Guide

This guide describes the software environment and command-line workflow used for the
experiments in "A DeepSets-Guided Scheduling Framework for Total Tardiness
Minimization in Single-Machine Scheduling".

## 1. Archive the code snapshot

For a DOI-linked code snapshot, create a release from the exact Git commit used for
the revised manuscript and archive that release with Zenodo.

1. Push the final revision commit to GitHub.
2. In Zenodo, enable GitHub archiving for `dz5430/dgsf-single-machine`.
3. Create a GitHub release, for example `v1.0-ejor-r1`.
4. Zenodo will mint a DOI for that release. Add the DOI badge/link to `README.md`
   and cite the archived software snapshot in the manuscript.

Record the Git commit hash in the response letter and in the Zenodo metadata.

## 2. Create the conda environment

The reported experiments used Python 3.10 and the package versions listed in
`environment.yml`.

```powershell
git clone https://github.com/dz5430/dgsf-single-machine.git
cd dgsf-single-machine
conda env create -f environment.yml
conda activate scheduling_env
```

Verify that the main Python dependencies are available:

```powershell
python -c "import numpy, pandas, scipy, sklearn, pyomo, torch, matplotlib, openpyxl; print('environment ok')"
```

The MIP components use Pyomo with the Gurobi command-line solver. The reported
experiments used Gurobi Optimizer 12.0.1. Install Gurobi separately, activate a
valid academic/commercial license, and make sure `gurobi_cl` is on the system
path. Then verify solver access:

```powershell
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('gurobi').available(False))"
```

The command should print `True`.

## 3. Download data, trained models, and result files

Large `.xlsx`, `.csv`, and `.pth` artifacts are not tracked in Git. Download the
artifact bundle from the project Google Drive linked in `README.md`, then place
the files under the matching folders:

```text
Data/
Results/
```

The workflow below assumes paths inside those folders. You may use other paths by
passing them through the command-line options shown below.

## 4. Reproduce the main DGSF pipeline

The scripts can be run from the repository root.

Generate ML input features:

```powershell
python Code/1_data_processing/Update_input_features.py --input Data/Raw_file.xlsx --output Data/Updated_file.xlsx
```

Compute instance characteristics:

```powershell
python Code/1_data_processing/SMS_instance_classification.py --input Data/Raw_file.xlsx --output Data/Instances_characteristics.xlsx
```

Evaluate dispatching-rule baselines, including the release-date-aware PRTT rule:

```powershell
python Code/3_evaluation/Dispatching_heuristics.py --input Data/Updated_file.xlsx --output Results/Dispatching_heuristics.xlsx
```

Generate reference solutions with the time-indexed MIP:

```powershell
python Code/1_data_processing/Solver_discrete_time.py --input Data/Updated_file.xlsx --output Results/Updated_file_with_reference.xlsx
```

Run the pretrained DeepSets model and local-swap refinement:

```powershell
python Code/2_dgsf_main/Load_ML_swap.py --input Data/Updated_file.xlsx --model "Data/Trained Models/Dev3_30_theta_max_6Itau.pth" --output Results/Post_ML_file.xlsx
```

Run the continuous-time MIP post-processing step:

```powershell
python Code/2_dgsf_main/Solver_MIP_ct_post.py --input Results/Post_ML_file.xlsx --output Results/Final_prediction.xlsx --time-limit 60
```

Evaluate objective gaps and rank-vector similarity:

```powershell
python Code/3_evaluation/Schedule_evaluation.py --input Results/Final_prediction.xlsx --method-obj-col tardiness_mip_post_tight_old --ref-obj-col tardiness_dtime --pred-rank-col rank_mip_post_tight_old --ref-rank-col rank_vector_dtime
```

## 5. Optional: retrain the DeepSets model

Training requires a solved training file with the feature columns and reference
columns described in the paper.

```powershell
python Code/2_dgsf_main/DeepSets_SMS_Scheduling_training.py --input Data/Training_file.csv --output-model "Data/Trained Models/Model_name.pth" --loss-plot Results/loss_curves.png --device cuda --epochs 1000
```

Use `--device cpu` if CUDA is unavailable.
