# External Artifact Manifest

Large experiment artifacts are stored outside git. Download the artifact bundle
from the project Google Drive folder and place files under the paths below.

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
Data/Trained Models/Dev3_30_theta_max_6Itau.pth
Data/Trained Models/Dev3_30_theta_0max_40tau_avg.pth
```

## F1 Instance and Result Workbooks

The F1 folder contains representative single-machine benchmark workbooks used
by the reproducibility examples:

```text
Results/F1/Dev3_singlemachine_instances_60_theta_max_6Itau.xlsx
Results/F1/Dev3_singlemachine_instances_60_theta_max_6Itau_u4.xlsx
Results/F1/Dev3_singlemachine_instances_80_theta_max_6Itau.xlsx
Results/F1/Dev3_singlemachine_instances_80_theta_max_6Itau_u4.xlsx
Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau.xlsx
Results/F1/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx
Results/F1/Dev3_singlemachine_instances_120_theta_max_6Itau.xlsx
Results/F1/Dev3_singlemachine_instances_120_theta_max_6Itau_u4.xlsx
Results/F1/Dev3_singlemachine_instances_60_theta_0max_40tau_avg.xlsx
Results/F1/Dev3_singlemachine_instances_60_theta_0max_40tau_avg_u4.xlsx
Results/F1/Dev3_singlemachine_instances_80_theta_0max_40tau_avg.xlsx
Results/F1/Dev3_singlemachine_instances_80_theta_0max_40tau_avg_u4.xlsx
Results/F1/Dev3_singlemachine_instances_100_theta_0max_40tau_avg.xlsx
Results/F1/Dev3_singlemachine_instances_100_theta_0max_40tau_avg_u4.xlsx
Results/F1/Dev3_singlemachine_instances_120_theta_0max_40tau_avg.xlsx
Results/F1/Dev3_singlemachine_instances_120_theta_0max_40tau_avg_u4.xlsx
```

Additional F2-F5 experiment outputs can be placed under:

```text
Results/F2/
Results/F3/
Results/F4/
Results/F5/
```

## Notes

- `_u4` workbooks contain feature columns and reference solution columns used by
  the ML and post-processing scripts.
- `.pth`, `.xlsx`, and `.csv` files are ignored by git to keep the repository
  lightweight.
- The scripts accept command-line paths, so artifacts may be stored elsewhere if
  the paths in the example commands are adjusted.
