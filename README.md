# DGSF: A DeepSets-Guided Scheduling Framework for Single-Machine Total Tardiness

Code and reproducibility materials for the DeepSets-Guided Scheduling Framework
(DGSF) experiments on the single-machine total tardiness problem.

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
- `Data/`: product tables, trained checkpoints, and external benchmark data.
- `Results/`: supplied or regenerated experiment outputs (`Tables/`, `F1/`,
  `F2/`, `F3/`, `F4/`, and `F5/`).
- `environment.yml`: conda environment used for the reproducibility workflow.
- `REPRODUCIBILITY.md`: step-by-step setup and command-line workflow.
- `docs/artifact_manifest.md`: expected external artifacts and where to place
  them.

Large `.xlsx`, `.csv`, and `.pth` artifacts are not committed. Download the
project data archive, preserve its folder structure, and merge it into this
repository. The exact layout and filenames are listed in
`docs/artifact_manifest.md`.

Artifact archive:
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

## Instance Types, Facilities, and Filename Tokens

The manuscript studies two instance types and five facility configurations.

| Token | Meaning |
|---|---|
| `theta_max_6Itau` | Type A instances, with all jobs released at time zero. |
| `theta_0max_40tau_avg` | Type B instances, with staggered release times. |
| `F1` | Base facility with integer processing times (time resolution 1.0). |
| `F2`, `F3` | Alternative facilities used in the generalization study. |
| `F4`, `F5` | F1 evaluated at time resolutions 0.5 and 0.1, respectively. |

Artifact filenames retain the identifiers used to generate the reported
results. `Dev3` identifies the data-generation configuration, `_u4` identifies
a processed workbook containing model features and reference-solution columns,
`50k` denotes 50,000 training instances, and `dev9_lean` identifies the
DeepSets model architecture. The `_dgsf` and `_mip` suffixes distinguish DGSF
outputs and their MIP-postprocessed counterparts. These filenames are retained
so that the repository paths correspond directly to the accompanying data
archive.

## Data Layout

Place external artifacts as follows:

```text
Data/
    Facility Products/
    Trained Models/
Results/
    Tables/
    F1/
        input/
        output/
            F1_DGSF/
            F1_Recursive/
            Time resolution/
        Max Tardiness Evaluation/
    F2/
        input/
        output/
    F3/
        input/
        output/
    F4/
        input/
        output/
    F5/
        input/
        output/
```

The expected filenames are listed in `docs/artifact_manifest.md`.

## Run the Pipeline

The scripts can be run from the repository root.

Run the supplied Dev9-Lean model with local-swap refinement:

```bash
python Code/2_dgsf_main/evaluate_sms_model.py \
  --input Results/F1/input/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --model "Data/Trained Models/30_theta_max_6Itau_Dev3_50k_dev9_lean.pth" \
  --architecture dev9_lean \
  --device cpu \
  --output Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf.xlsx
```

Run continuous-time MIP post-processing:

```bash
python Code/2_dgsf_main/Solver_MIP_ct_post.py \
  --input Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf.xlsx \
  --output Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf_mip.xlsx \
  --time-limit 60
```

Evaluate a schedule against the reference solution:

```bash
python Code/3_evaluation/Schedule_evaluation.py \
  --input Results/F1/output/F1_DGSF/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_dgsf_mip.xlsx \
  --method-obj-col tardiness_dgsf_mip \
  --ref-obj-col tardiness_dtime \
  --pred-rank-col rank_dgsf_mip \
  --ref-rank-col rank_vector_dtime
```

Evaluate static dispatching-rule baselines:

```bash
python Code/3_evaluation/Dispatching_heuristics.py \
  --input Results/F1/input/Dev3_singlemachine_instances_100_theta_max_6Itau_u4.xlsx \
  --output Results/F1/output/F1_Recursive/Dev3_singlemachine_instances_100_theta_max_6Itau_u4_heuristics.xlsx
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
