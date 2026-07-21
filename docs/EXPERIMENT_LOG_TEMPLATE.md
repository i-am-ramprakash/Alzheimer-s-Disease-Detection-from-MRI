# Experiment Log Template

Copy this file for each experiment, for example `artifacts/notes/svm-c3.md`. Change one main variable
at a time so you can explain why the result changed.

## Question

What specific question does this experiment answer?

## Hypothesis

What do you expect, and why?

## Environment

- Date:
- Git commit (`git rev-parse --short HEAD`):
- Python version (`python --version`):
- Key package versions (`python -m pip freeze`):
- CPU/GPU:

## Data

- Source:
- Split directory:
- Split seed:
- Train/validation/test counts:
- Patient-grouped split available: yes/no
- Exact-duplicate check performed:

## Model and changed variable

- Workflow: SVM / custom CNN / MobileNetV2
- Baseline settings:
- Single intentional change:
- Output directory:

## Command

```powershell
# Paste the exact command here.
```

## Results

| Metric | Baseline | This run |
| --- | ---: | ---: |
| Accuracy | | |
| Macro F1 | | |
| Weighted F1 | | |

- Per-class behavior:
- Confusion-matrix observations:
- Training time:
- Unexpected warnings or failures:

## Interpretation

Did the evidence support the hypothesis? What alternative explanations exist? Was model selection made
using validation data rather than the final test set?

## Next experiment

What is the smallest useful next change?
