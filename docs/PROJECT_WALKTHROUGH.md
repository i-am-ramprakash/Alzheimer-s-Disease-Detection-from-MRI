# Project Walkthrough

## 1. The problem being solved

The project receives one standardized 2D brain MRI image and assigns one of four dataset labels:

```text
Mild Demented
Moderate Demented
Non Demented
Very Mild Demented
```

The published model is a classical machine-learning pipeline:

```text
image file
   |
   v
grayscale conversion -> resize to 32 x 32 -> flatten to 1,024 values -> divide by 255
   |
   v
StandardScaler (learned from training data)
   |
   v
RBF support-vector classifier
   |
   v
class index -> readable class label
```

Training uses many labeled images to learn the scaler and classifier. Prediction reuses those learned
objects on one new image. Evaluation compares predicted labels with known test labels.

## 2. Repository map

```text
alzheimer_detection/
|-- constants.py          Shared class names and supported image extensions
|-- config.py             Validated CNN training configuration
|-- dataset.py            Directory discovery, auditing, and TensorFlow datasets
|-- prepare_dataset.py    Deterministic stratified train/test splitter
|-- audit.py              Dataset-audit command-line interface
|-- classical.py          Recommended SVM training and prediction workflow
|-- model.py              Custom CNN and MobileNetV2 model definitions
|-- training.py           Custom CNN training workflow
|-- transfer_training.py  Two-stage MobileNetV2 training workflow
|-- prediction.py         Prediction for TensorFlow .h5 models
`-- evaluation.py         Reports and confusion-matrix images

models/                   Included trained SVM and its metadata
results/                  Published test metrics and confusion matrix
tests/                    Fast automated behavior checks
docs/                     Learning material
```

## 3. The data contract

Machine-learning code is only reliable when its expected input is explicit. This project expects:

```text
data/alzheimer_mri_clean/
|-- train/
|   |-- MildDemented/
|   |-- ModerateDemented/
|   |-- NonDemented/
|   `-- VeryMildDemented/
`-- test/
    |-- MildDemented/
    |-- ModerateDemented/
    |-- NonDemented/
    `-- VeryMildDemented/
```

`constants.py` defines a fixed canonical class order. `dataset.py` accepts small directory-name
variations by removing punctuation and capitalization, then mapping aliases to canonical keys. A
fixed order is essential: label `0` must mean the same class in training, evaluation, and prediction.

The audit checks that:

- `train` and `test` exist;
- all four recognized class directories exist;
- every class has at least one supported image;
- counts can be reported consistently;
- optionally, Pillow can open and verify every image.

The audit does not establish patient independence. The source data has no reliable subject identifier.

## 4. Preparing the split

`prepare_dataset.py` starts with one directory per class. For each class independently, it:

1. finds supported image files recursively;
2. sorts paths to remove filesystem-order variation;
3. shuffles using a seed combined with the class key;
4. places approximately 20% in `test` and the remainder in `train`;
5. uses a hard link when possible, otherwise copies the file;
6. writes `split_manifest.json` with the seed and counts.

This is a stratified split because each class is split independently. The same seed and same source
files produce the same assignment. The command refuses to overwrite an existing output directory,
which protects an experiment from accidental mutation.

## 5. The recommended SVM path

The important functions in `classical.py` are:

### `image_features(image_path)`

Pillow opens the image, converts it to one grayscale channel, resizes it to 32 by 32 pixels with
bilinear interpolation, flattens it into a vector, and scales byte intensities from 0-255 to 0-1.

```text
(height=32, width=32) -> 32 * 32 = 1,024 features
```

Spatial position is retained: feature 0 is the top-left pixel and feature 1 is the next pixel. This is
why the model can work unusually well on consistently aligned images and can fail on shifted, cropped,
rotated, or differently processed images.

### `load_split(split_path, class_directories)`

Images are loaded in the canonical class order. Each feature vector is appended to `X`; its integer
class index is appended to `y`. In common ML notation:

```text
X shape = (number_of_images, 1024)
y shape = (number_of_images,)
```

### `make_pipeline(StandardScaler(), SVC(...))`

`StandardScaler` learns the mean and standard deviation of every pixel position from training data.
It then standardizes each feature. Keeping it inside a scikit-learn pipeline is important: prediction
automatically applies the exact training transformation, and test data does not influence the fitted
scaler.

The SVC settings mean:

- `kernel="rbf"`: learn nonlinear boundaries based on similarity between samples;
- `C=10`: penalize training errors fairly strongly;
- `gamma="scale"`: let scikit-learn derive the RBF width from feature count and variance;
- `class_weight="balanced"`: give rare classes more weight than frequent classes;
- `cache_size=2048`: permit a larger kernel cache to speed training.

An RBF kernel can be understood as measuring similarity. For two feature vectors `x` and `z`:

```text
K(x, z) = exp(-gamma * ||x - z||^2)
```

Similar standardized images have a kernel value near 1; very different images have a value near 0.
The classifier uses selected training points, called support vectors, to define class boundaries.

### Saved training artifacts

Training writes:

- `best_model.joblib`: scaler and SVC together;
- `model_metadata.json`: class order and preprocessing contract;
- `dataset_audit.json`: exact directories and counts used;
- `evaluation.json`: held-out metrics;
- `confusion_matrix.png`: visual error summary.

The published artifact is named `models/alzheimer_svm.joblib`, with matching metadata next to it.
Joblib uses pickle internally, so only load artifacts from trusted sources.

## 6. Prediction path

The `predict` function in `classical.py`:

1. validates the model, image, and metadata paths;
2. reads the human-readable label order from metadata;
3. loads the fitted pipeline;
4. applies the same `image_features` preprocessing;
5. calls `model.predict` for the winning class;
6. ranks transformed decision scores for display.

The displayed scores are not probabilities. Applying a softmax makes decision values easier to rank,
but it does not calibrate them against real frequencies. A score of `0.90` must not be read as a 90%
medical likelihood or even as a calibrated 90% model confidence.

## 7. Evaluation path

For every held-out test image, the training command compares the predicted integer with the true
directory-derived integer. It creates a classification report and confusion matrix.

The confusion matrix uses rows for actual classes and columns for predicted classes. A perfect model
would place every count on the main diagonal. See [metrics and limitations](METRICS_AND_LIMITATIONS.md)
for calculations and interpretation of the published result.

## 8. Neural-network comparison paths

The SVM is the recommended result, but two neural approaches are retained for learning.

### Custom CNN

`model.py::build_model` processes 128x128 grayscale images using:

```text
rescaling and augmentation
-> Conv2D / BatchNorm / ReLU / Pool / Dropout (three blocks)
-> separable convolution
-> global average pooling
-> dense layer
-> four-unit softmax output
```

`training.py` creates a validation subset from the training directory, applies balanced class weights,
and uses callbacks for checkpoints, early stopping, learning-rate reduction, and CSV history logging.
Only after training does it evaluate the best checkpoint on `test`.

### MobileNetV2 transfer learning

`transfer_training.py` converts grayscale source images to RGB because MobileNetV2 expects three
channels. Training has two stages:

1. freeze the ImageNet feature extractor and train the new classification head;
2. unfreeze only its top 30 non-batch-normalization layers and fine-tune at a small learning rate.

The workflow compares validation loss from the best frozen and fine-tuned checkpoints, selects one,
and evaluates that selected model on the test set.

These neural models scored lower here. More architectural complexity does not guarantee better results,
especially on small, standardized data whose alignment is directly exploitable by an RBF SVM.

## 9. How the modules depend on each other

```text
constants.py
   +--> dataset.py --> audit.py
   |        |          training.py -----------+
   |        |          transfer_training.py --+--> evaluation.py
   |        +--------> classical.py ----------+
   |
   +--> model.py --> training.py / transfer_training.py
   +--> prepare_dataset.py (also uses dataset class-name mapping)

models/model_metadata.json --> classical.py prediction
```

## 10. Safe ways to modify the project

Change one variable at a time and write results to a new artifact directory. For example, compare
`C=3`, `C=10`, and `C=30`, but do not repeatedly select parameters from the held-out test result. The
proper workflow is train -> validation for model selection -> one final test evaluation.

Good starter modifications include:

- compare SVM feature sizes such as 16x16 and 32x32;
- visualize the resized feature image;
- calculate class weights by hand and compare them with the code;
- remove `class_weight="balanced"` and inspect per-class recall;
- add tests for invalid paths and unknown directory names;
- add a patient-group split only if a future dataset supplies trustworthy patient IDs.

Avoid claiming improvement based only on the same test set used repeatedly during experimentation.
