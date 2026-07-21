# Educational Alzheimer MRI Classification

An educational four-class classifier for standardized brain MRI slice images:

- Non Demented
- Very Mild Demented
- Mild Demented
- Moderate Demented

The recommended model is an RBF-SVM trained on aligned 32x32 grayscale pixel features. A small CNN
and a MobileNetV2 transfer-learning pipeline are included as comparison experiments.

> Educational use only. This project is not a medical device, does not provide diagnoses, and must
> not be used for health decisions.

## Verified result

The published SVM was evaluated on a deterministic 80/20 image-level split of 6,400 unique images.
No exact file hash occurs in both training and test sets.

| Metric | Result |
| --- | ---: |
| Test accuracy | 99.2188% |
| Weighted F1 | 99.2185% |
| Macro F1 | 99.4325% |
| Correct test predictions | 1,270 / 1,280 |
| Moderate Demented recall | 100% (13 / 13) |

See [the full evaluation](results/evaluation.json), [confusion matrix](results/confusion_matrix.png),
and [model card](MODEL_CARD.md).

This high score applies only to these standardized, spatially aligned images. The source does not
provide reliable patient IDs, so related slices from the same person may cross the image-level
split. It is not evidence of patient-independent or clinical performance.

## Quick prediction with the included model

Create the environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Classify one image:

```powershell
python -m alzheimer_detection.classical predict `
  --model models\alzheimer_svm.joblib `
  --image "C:\path\to\mri-image.jpg"
```

Only load the bundled Joblib model from a repository you trust. Joblib files use Python pickle
internally and can execute code when loaded.

## Dataset

The MRI images are not committed to this repository. The experiment used the four-class
[Alzheimer MRI image dataset on Kaggle](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images).
Review the dataset's current terms and provenance before using or redistributing it.

The preparation command expects one directory containing the four class directories:

```text
source/
|-- MildDemented/
|-- ModerateDemented/
|-- NonDemented/
`-- VeryMildDemented/
```

Create the deterministic stratified split:

```powershell
python -m alzheimer_detection.prepare_dataset `
  --source-dir "C:\path\to\source" `
  --output-dir data\alzheimer_mri_clean `
  --test-fraction 0.20 `
  --seed 42
```

The command uses space-saving hard links when the source and destination are on the same drive,
falling back to copies when necessary. It refuses to overwrite an existing output directory.

Audit the prepared data:

```powershell
python -m alzheimer_detection.audit `
  --data-dir data\alzheimer_mri_clean `
  --verify-images
```

## Train the recommended model

```powershell
python -m alzheimer_detection.classical train `
  --data-dir data\alzheimer_mri_clean `
  --output-dir artifacts\svm-01 `
  --c 10
```

The training command verifies every image, trains on the complete training split, evaluates once on
the held-out test split, and saves the model, metadata, JSON metrics, and confusion matrix.

## Neural-network comparison experiments

Train the custom grayscale CNN:

```powershell
python -m alzheimer_detection.training `
  --data-dir data\alzheimer_mri_clean `
  --output-dir artifacts\cnn-01 `
  --epochs 30
```

Train and fine-tune ImageNet MobileNetV2:

```powershell
python -m alzheimer_detection.transfer_training `
  --data-dir data\alzheimer_mri_clean `
  --output-dir artifacts\transfer-01
```

On the same image-level test split, the custom CNN achieved 54.06% and MobileNetV2 achieved 60.23%.
These experiments demonstrate that a model designed for aligned low-resolution images can outperform
generic image transfer learning on this dataset.

## Tests

```powershell
python -m pytest
```

## Repository layout

```text
alzheimer_detection/   Training, evaluation, prediction, and data preparation
models/                Included high-accuracy SVM and metadata
results/               Published held-out metrics and confusion matrix
tests/                 Lightweight automated tests
MODEL_CARD.md           Model details and limitations
requirements.txt       Pinned Python environment
```

## Limitations

- The evaluation is image-level, not patient-level.
- The data contains 2D slices rather than complete 3D MRI volumes.
- Only 64 source images belong to the Moderate Demented class.
- The model expects similarly aligned and preprocessed grayscale MRI images.
- Decision scores are rankings, not calibrated probabilities.
- Accuracy on another dataset, scanner, hospital, or population may be much lower.
