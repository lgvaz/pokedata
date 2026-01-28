# pokedata

A data management tool.

## Installation

Install in editable mode:

```bash
pip install -e .
```

## Abstractions
---

```
Record        → source world
RecordPlan   → plans transition to canonical world
DatasetPlan  → immutable transition artifact
```

## Instructions

### Updating DVC dataset

Add changes and push:
```bash
dvc add data/canonical
dvc push
```

Commit push:
```bash
git commit -m "data releases v0.2.0"
git push
```

**Release via github UI**
Don't forget to add what changed to the notes


### Creating a new dataset from scratch

Initialiaze DVC:
```bash
dvc init
git add .dvc .dvcignore .gitignore
git commit -m "chore(dvc): init"
```

Configure remote:
```bash
dvc remote add -d storage s3://ags-ai-cards-training-dataset/dvcstore
```

Commit:
```bash
git add .dvc/config
git commit -m "chore(dvc): add remote"
```

Track dataset with DVC:
```bash
dvc add data/ags-cards/canonical
git add data/ags-cards/canonical.dvc .gitignore
git commit -m "data(ags-cards): canonical v0.1.0"
```

Push:
```bash
dvc push
```

Tag datasets:
```bash
git tag -a data/v0.1.0 -m "canonical dataset v0.1.0"
```