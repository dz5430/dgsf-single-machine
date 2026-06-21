# Data Files

This folder stores input datasets and trained model checkpoints used by the
single-machine DGSF workflow.

Large `.xlsx`, `.csv`, and `.pth` artifacts are not stored in git. Download
them from the project Google Drive folder and place them here according to
`../docs/google_drive_manifest.md`.

Expected subfolders:

```text
Facility Products/
Trained Models/
```

The main scripts accept command-line paths, so reviewers can keep the data in a
different location if they adjust the paths in the commands.
