# Data Files

This directory layout is part of the production-ready `v1.0.0` release.

This folder stores input datasets and trained model checkpoints used by the
single-machine DGSF workflow.

Large `.xlsx`, `.csv`, and `.pth` artifacts are not stored in git. Download the
`Data/` folder from the external artifact archive into this directory,
preserving its two child folders, according to `../docs/artifact_manifest.md`.

Expected subfolders:

```text
Facility Products/
Trained Models/
```

The main scripts accept command-line paths, so reviewers can keep the data in a
different location if they adjust the paths in the commands.
