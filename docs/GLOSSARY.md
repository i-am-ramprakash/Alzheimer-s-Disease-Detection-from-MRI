# Project Glossary

## Data terms

**Artifact**: A file produced by a run, such as a trained model, metrics JSON, training history, or plot.

**Class / label**: The category assigned to an image. This project has four dataset labels.

**Data leakage**: Information reaching model training that should only be available later. Leakage can
make evaluation look much better than real-world performance.

**Dataset split**: A separation of data by purpose. Training data fits the model, validation data guides
model choices, and test data is reserved for final evaluation.

**Distribution shift**: A difference between training data and future input, such as another scanner,
crop, slice position, or population.

**Feature**: A numerical input to a model. The SVM uses 1,024 resized grayscale pixel values.

**Patient-level split**: A split that keeps every image from one patient in only one partition. This is
not possible with the current source because trustworthy patient IDs are unavailable.

**Stratified split**: A split that approximately preserves each class's proportion.

## Model terms

**Class weight**: A multiplier that makes errors on rare classes count more during fitting.

**CNN (convolutional neural network)**: A neural network that learns local visual patterns using
convolution filters.

**Epoch**: One complete pass through the neural network's training data.

**Fine-tuning**: Continuing training on some layers of a pretrained network using a small learning rate.

**Hyperparameter**: A setting selected by the developer rather than learned directly, such as SVM `C`,
learning rate, batch size, or number of epochs.

**Inference / prediction**: Applying a trained model to an input without changing learned parameters.

**Overfitting**: Learning training-specific details that do not generalize to unseen data.

**Pipeline**: An ordered chain of transformations and a model. The included Joblib artifact contains
both `StandardScaler` and `SVC` so preprocessing cannot be accidentally skipped.

**RBF kernel**: A similarity function that lets an SVM learn nonlinear boundaries.

**Standardization**: Transforming each feature using a training-set mean and standard deviation.

**Support vector**: A training sample that helps define an SVM's decision boundary.

**Transfer learning**: Starting with a network trained on another task, such as ImageNet, and adapting
it to a new task.

## Evaluation terms

**Accuracy**: The fraction of all predictions that match their labels.

**Calibration**: Agreement between predicted probabilities and observed frequencies. The SVM's
displayed scores are not calibrated.

**Confusion matrix**: A table counting actual-versus-predicted class combinations.

**F1 score**: The harmonic mean of precision and recall.

**Held-out test set**: Data excluded from fitting and intended for final evaluation.

**Macro average**: A metric averaged with equal importance for each class.

**Precision**: Among samples predicted as one class, the fraction carrying that label.

**Recall**: Among samples carrying one label, the fraction predicted as that class.

**Support**: The number of labeled examples included in a metric.

**Weighted average**: A class metric averaged according to each class's support.

## Software terms

**CLI (command-line interface)**: A program used through commands and options such as `--data-dir`.

**Deterministic**: Designed to reproduce the same output from the same input and seed, within the limits
of the underlying software and hardware.

**Joblib / pickle**: Python serialization used by the SVM artifact. Loading an untrusted file can execute
malicious code, so model files must be treated like executable software.

**Metadata**: Structured information describing a model's class order, input size, and preprocessing.

**Module**: A Python source file that can be imported or executed with `python -m`.

**Reproducibility**: The ability to rerun an experiment from recorded code, data split, configuration,
dependencies, and random seed.

**Virtual environment**: An isolated Python installation for one project's dependencies.
