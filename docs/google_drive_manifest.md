# External Artifact Manifest

Large experiment artifacts are stored outside git. Download the complete
artifact bundle and merge it into the repository root, preserving the Drive
folder structure. The paths below are the current frozen-bundle layout.

Google Drive:
https://drive.google.com/drive/u/2/folders/1Lo8WRZabBUxGNA0nOD50TMavkKyhDwHP

## Product Tables

These small CSV files are included in git:

```text
Data/Facility Products/products_F1.csv
Data/Facility Products/products_F2.csv
Data/Facility Products/products_F3.csv
Data/Facility Products/products_F4.csv
Data/Facility Products/products_F5.csv
```

## Trained Model Checkpoints

```text
Data/Trained Models/30_theta_max_6Itau_Dev3_50k_dev9_lean.pth
Data/Trained Models/30_theta_0max_40tau_avg_Dev3_50k_dev9_lean.pth
```

## Result Folder Layout

The Drive `Results/` folder contains the following top-level folders:

```text
Results/Tables/
Results/F1/
Results/F2/
Results/F3/
Results/F4/
Results/F5/
```

### F1

The F1 folder is organized as follows:

```text
Results/F1/input/
Results/F1/output/F1_DGSF/
Results/F1/output/F1_Recursive/
Results/F1/output/Time resolution/
Results/F1/Max Tardiness Evaluation/
```

The current F1 input folder contains eight `_u4.xlsx` benchmark workbooks
covering 60, 80, 100, and 120 jobs for the two instance types, plus the raw
CSV used for feature processing:

```text
Results/F1/input/Dev3_singlemachine_instances_60_theta_max_6Itau_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_80_theta_max_6Itau_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_120_theta_max_6Itau_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_60_theta_0max_40tau_avg_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_80_theta_0max_40tau_avg_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_100_theta_0max_40tau_avg_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_120_theta_0max_40tau_avg_u4.xlsx
Results/F1/input/Dev3_singlemachine_instances_80_theta_max_6Itau_u4.csv
```

Download the complete `F1_DGSF/`, `F1_Recursive/`, and `Time resolution/`
subfolders; their filenames are retained exactly as stored in Drive. The
`F1_DGSF/` files use the `newML` and `newML_mip` suffixes, while recursive
baselines are stored under `F1_Recursive/`.

### F2-F5

Each of the remaining facility folders contains `input/` and `output/`
subfolders. Download each complete folder without renaming files:

```text
Results/F2/input/
Results/F2/output/
Results/F3/input/
Results/F3/output/
Results/F4/input/
Results/F4/output/
Results/F5/input/
Results/F5/output/
```

### Summary Tables

The `Results/Tables/` folder contains the CSV/XLSX summary files used to check
the manuscript tables, including the F1/F4/F5 time-resolution summary, the F2/F3
generalization summary, and the recursive-versus-DGSF gap summaries.

## Max-Tardiness Diagnostic

The one-instance Type B, 150-job diagnostic used in Table 6 is distributed as
two cleaned workbooks, preserving the original distinction between the
unconstrained baseline and the capped sensitivity sweep:

```text
Results/F1/Max Tardiness Evaluation/Table6_unconstrained_baseline.xlsx
Results/F1/Max Tardiness Evaluation/Table6_capped_d6_t120.xlsx
```

Use `Code/3_evaluation/Max_tardiness_diagnostic.py` to recreate the capped
time-indexed MIP and DGSF post-processing schedules. The paper configuration
is fixed at rank window `delta = 6` and a 120-second post-processing limit.

## Notes

- `_u4` workbooks contain feature columns and reference solution columns used by
  the ML and post-processing scripts.
- The capped diagnostic workbook contains the stored raw ML and local-swap
  ranks used to seed the capped post-processing runs.
- `.pth`, `.xlsx`, and `.csv` files are ignored by git to keep the repository
  lightweight.
- The scripts accept command-line paths, so artifacts may be stored elsewhere if
  the paths in the example commands are adjusted.
