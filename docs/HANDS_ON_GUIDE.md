# Hands-on Guide

Run all commands from the repository root in PowerShell.

## Lab 1: Set up the environment

The project requires Python 3.8 through 3.11. TensorFlow 2.13 is not compatible with Python 3.12.

```powershell
python --version
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell blocks activation, you can run `.venv\Scripts\python.exe` directly in every command.
Activation is convenient but not required.

What to learn: a virtual environment isolates this project's exact package versions from other Python
projects. `requirements.txt` pins versions so the environment can be reproduced.

## Lab 2: Run the automated tests

```powershell
python -m pytest
```

The tests are intentionally fast. Read the files in `tests/` after they pass. Each test follows the
arrange-act-assert pattern:

1. arrange inputs and temporary directories;
2. act by calling one project function;
3. assert the observable result.

The tests do not retrain a model and do not prove model accuracy. They protect basic software behavior.

## Lab 3: Predict with the included model

Choose an image from your local dataset and keep its path quoted if it contains spaces:

```powershell
python -m alzheimer_detection.classical predict `
  --model models\alzheimer_svm.joblib `
  --image "data\alzheimer_mri_clean\test\NonDemented\YOUR_IMAGE.jpg"
```

Expected output structure:

```json
{
  "predicted_class": "Non Demented",
  "decision_scores": [
    {"class": "Non Demented", "score": 0.68}
  ],
  "scores_are_calibrated_probabilities": false,
  "educational_use_only": true
}
```

Your numeric values and ranking depend on the image. The list contains all four classes. Do not treat
the score as a medical probability.

Try one image from every test class. Record actual label, predicted label, and whether they match.
Directory names provide the dataset's actual labels for this exercise.

## Lab 4: Inspect the feature vector in Python

Start an interactive session:

```powershell
python
```

Then enter, replacing the path:

```python
from pathlib import Path
from alzheimer_detection.classical import image_features

features = image_features(Path(r"data\alzheimer_mri_clean\test\NonDemented\YOUR_IMAGE.jpg"))
print(features.shape)
print(features.dtype)
print(features.min(), features.max())
print(features[:10])
```

Expected concepts:

- shape `(1024,)` because 32 x 32 = 1,024;
- `float32` values;
- values between 0 and 1 before standardization;
- the first ten values represent the first ten pixels in flattened order.

Exit with `exit()`.

## Lab 5: Prepare a clean dataset

The source directory should contain the four class directories, without an existing train/test split.
Keep the source unchanged and choose a new output path:

```powershell
python -m alzheimer_detection.prepare_dataset `
  --source-dir "C:\path\to\source" `
  --output-dir data\my_split `
  --test-fraction 0.20 `
  --seed 42
```

Inspect the generated manifest:

```powershell
Get-Content data\my_split\split_manifest.json
```

Run the command again with another new output path and the same seed. File assignments should match.
Then try a different seed and observe that counts stay approximately equal while assigned files change.

The tool refuses an output directory that already exists. To avoid losing work, choose a new name for
each experiment rather than deleting or overwriting a previous split.

## Lab 6: Audit the dataset

Fast structural audit:

```powershell
python -m alzheimer_detection.audit --data-dir data\alzheimer_mri_clean
```

Full image-read audit:

```powershell
python -m alzheimer_detection.audit `
  --data-dir data\alzheimer_mri_clean `
  --verify-images
```

Read the reported count for each class. Notice the imbalance: `ModerateDemented` is much smaller than
the other classes. This motivates balanced class weights and metrics beyond overall accuracy.

## Lab 7: Retrain the SVM

Use a new output directory so the published model remains untouched:

```powershell
python -m alzheimer_detection.classical train `
  --data-dir data\alzheimer_mri_clean `
  --output-dir artifacts\my-svm-run `
  --c 10
```

This can take several minutes and uses significant memory. When complete, inspect:

```powershell
Get-Content artifacts\my-svm-run\evaluation.json
Get-Content artifacts\my-svm-run\model_metadata.json
Get-Content artifacts\my-svm-run\dataset_audit.json
```

Open `artifacts\my-svm-run\confusion_matrix.png` using your normal image viewer. Confirm that the
test sample count is 1,280 if you used the same prepared split.

## Lab 8: Read the published confusion matrix manually

The published matrix is:

```text
                         Predicted
Actual             Mild  Moderate  Non  Very Mild
Mild                177      0       1       1
Moderate              0     13       0       0
Non                   0      0     638       2
Very Mild             0      0       6     442
```

Answer these questions before checking [the explanation](METRICS_AND_LIMITATIONS.md):

1. How many test predictions are correct?
2. How many Mild images were incorrectly called Non Demented?
3. Which actual class produced the most errors?
4. Why is 13/13 weaker evidence than 638/640, despite being 100%?

## Lab 9: Run a short CNN learning experiment

This is for understanding the workflow, not for reaching the published accuracy. A two-epoch run is a
quick smoke test:

```powershell
python -m alzheimer_detection.training `
  --data-dir data\alzheimer_mri_clean `
  --output-dir artifacts\cnn-learning-run `
  --epochs 2 `
  --verify-images
```

Compare `training_history.csv`, `history.json`, and `evaluation.json`. Training accuracy describes seen
data, validation metrics guide training, and test metrics estimate performance only after model choices
are fixed.

GPU configuration and TensorFlow installation vary by machine. CPU training is valid but slower.

## Lab 10: Explore command help and source code

```powershell
python -m alzheimer_detection.classical --help
python -m alzheimer_detection.classical train --help
python -m alzheimer_detection.prepare_dataset --help
python -m alzheimer_detection.audit --help
```

For each command, locate its `main()` function and argument parser in the corresponding module. Follow
the function calls until you reach preprocessing, model fitting, or auditing. This is an effective way
to learn an unfamiliar command-line project.

## Common problems

### `Dataset directory does not exist`

Run `Get-Location`, verify the relative path, and use `Test-Path data\alzheimer_mri_clean`.

### Missing or unrecognized classes

Ensure all four class directories are present. Supported forms include `MildDemented`,
`ModerateDemented`, `NonDemented`, and `VeryMildDemented`.

### Metadata does not exist

The model and `model_metadata.json` must be in the same directory unless `--metadata` explicitly points
to another file.

### TensorFlow import or DLL error

Confirm Python is 3.8-3.11, activate the project environment, run `python -m pip check`, and reinstall
from `requirements.txt` if necessary. The SVM prediction path itself uses scikit-learn, not TensorFlow.

### Memory pressure during SVM training

Close other memory-heavy programs. RBF SVM training scales poorly as sample count grows. It is suitable
for this 6,400-image educational dataset but not automatically suitable for a much larger collection.

### A random outside image gives an odd result

That is expected. The model assumes the same kind of centered, aligned, preprocessed 2D slice as its
training images. It has no reliable "not a compatible MRI" rejection mechanism.
