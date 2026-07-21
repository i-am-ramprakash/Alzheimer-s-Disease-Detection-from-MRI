# Model Card: Alzheimer MRI RBF-SVM

## Intended use

This model demonstrates four-class machine learning on standardized MRI slice images for coursework
and software-learning purposes. It is not intended for diagnosis, screening, triage, prognosis, or
any other medical use.

## Model

- Algorithm: scikit-learn RBF support-vector classifier
- Input: one spatially aligned grayscale MRI image
- Preprocessing: bilinear resize to 32x32, flatten, scale pixel values to 0-1, standardize features
- Hyperparameters: `C=10`, `gamma="scale"`, balanced class weights
- Output classes: Mild Demented, Moderate Demented, Non Demented, Very Mild Demented
- Serialized artifact: `models/alzheimer_svm.joblib`
- Training environment: Python 3.8, scikit-learn 1.3.2, NumPy 1.24.3

## Data and split

The experiment used 6,400 unique 128x128 grayscale images from the four-class Alzheimer MRI image
dataset linked in the README. A deterministic seed-42 stratified split assigned 5,120 images to
training and 1,280 to testing. SHA-256 auditing found no exact duplicate image files within the source
or across the prepared train/test boundary.

Class counts in the test set:

| Class | Images |
| --- | ---: |
| Mild Demented | 179 |
| Moderate Demented | 13 |
| Non Demented | 640 |
| Very Mild Demented | 448 |

The dataset does not expose reliable subject identifiers. The split therefore cannot guarantee that
all slices from one person stay in a single partition.

## Held-out image results

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Mild Demented | 1.0000 | 0.9888 | 0.9944 |
| Moderate Demented | 1.0000 | 1.0000 | 1.0000 |
| Non Demented | 0.9891 | 0.9969 | 0.9930 |
| Very Mild Demented | 0.9933 | 0.9866 | 0.9899 |

Overall accuracy is 99.2188%. See `results/evaluation.json` for machine-readable metrics.

## Limitations and risks

- Image-level performance may be inflated by subject-level correlation between different slices.
- The smallest test class contains only 13 images, so its 100% result has high uncertainty.
- The model relies on spatial alignment and dataset-specific preprocessing patterns.
- Decision-function scores are not calibrated probabilities or diagnostic confidence.
- External validation has not been performed.
- Joblib uses pickle internally. Never load a model artifact from an untrusted source.

## Reproducibility

Prepare data with `alzheimer_detection.prepare_dataset`, then train with
`alzheimer_detection.classical`. Both commands and exact arguments are documented in the README.
