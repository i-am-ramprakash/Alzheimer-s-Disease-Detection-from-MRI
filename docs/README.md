# Learning Center

This directory explains the project as a learning exercise. Start with the files in this order:

1. [Hands-on guide](HANDS_ON_GUIDE.md) - run the project, inspect data, predict, train, and test.
2. [Project walkthrough](PROJECT_WALKTHROUGH.md) - understand the architecture and every module.
3. [Metrics and limitations](METRICS_AND_LIMITATIONS.md) - correctly interpret the 99.22% result.
4. [Project glossary](GLOSSARY.md) - look up data, model, evaluation, and software terminology.
5. [Experiment log template](EXPERIMENT_LOG_TEMPLATE.md) - record your own controlled experiments.

## Suggested study plan

| Session | Topic | Practical outcome |
| --- | --- | --- |
| 1 | Repository and environment | Run tests and one included-model prediction |
| 2 | Dataset preparation | Create and audit a deterministic train/test split |
| 3 | Image features | Convert an MRI image into the SVM's 1,024 numeric inputs |
| 4 | SVM pipeline | Retrain the recommended model and inspect its artifacts |
| 5 | Evaluation | Read precision, recall, F1, and the confusion matrix |
| 6 | Neural networks | Compare the custom CNN and MobileNetV2 code paths |
| 7 | Responsible interpretation | Explain why image accuracy is not clinical accuracy |
| 8 | Your experiment | Change exactly one variable and document the result |

You do not need medical or deep-learning expertise to begin. Basic Python, paths, functions, arrays,
and command-line usage are enough. The walkthrough introduces the machine-learning terms used here.

## What this project teaches

- organizing a Python machine-learning package;
- validating and splitting image data reproducibly;
- converting images into numerical features;
- training an imbalanced four-class classifier;
- separating training, prediction, and evaluation;
- writing artifacts that make an experiment reproducible;
- comparing classical ML with CNN and transfer-learning approaches;
- identifying data leakage and limits in medical-imaging demonstrations.

## What this project does not teach

It does not establish a diagnostic method. It does not process clinical DICOM studies or complete 3D
MRI volumes, identify patients, estimate real-world diagnostic performance, or replace medical review.
The class names come from the educational dataset and should be treated as dataset labels only.
