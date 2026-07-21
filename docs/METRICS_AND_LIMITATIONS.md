# Metrics and Limitations

## Published result

The included SVM correctly classified 1,270 of 1,280 held-out images:

```text
accuracy = correct predictions / all predictions
         = 1270 / 1280
         = 0.9921875
         = 99.21875%
```

This exceeds 95% on the project's deterministic image-level test split. It does not guarantee 95% on
new patients, a different dataset, clinical scans, or differently processed images.

## Confusion matrix

Rows are actual labels and columns are predicted labels:

| Actual \ Predicted | Mild | Moderate | Non | Very Mild |
| --- | ---: | ---: | ---: | ---: |
| Mild | 177 | 0 | 1 | 1 |
| Moderate | 0 | 13 | 0 | 0 |
| Non | 0 | 0 | 638 | 2 |
| Very Mild | 0 | 0 | 6 | 442 |

Diagonal cells are correct. Off-diagonal cells are errors. The ten errors are:

- one Mild image predicted as Non;
- one Mild image predicted as Very Mild;
- two Non images predicted as Very Mild;
- six Very Mild images predicted as Non.

## Precision, recall, and F1

For one class:

```text
precision = true positives / all images predicted as that class
recall    = true positives / all actual images of that class
F1        = 2 * precision * recall / (precision + recall)
```

Precision asks, "When the model predicts this class, how often does it match the dataset label?"
Recall asks, "Of all images carrying this dataset label, how many did the model find?"

For `Very Mild Demented`:

```text
true positives = 442
false negatives = 6
false positives = 3  (1 Mild + 2 Non)

precision = 442 / (442 + 3) = 99.33%
recall    = 442 / (442 + 6) = 98.66%
```

The classification report computes an F1 of approximately 98.99% for that class.

## Macro versus weighted averages

- Macro F1 calculates each class's F1 and gives all four classes equal weight.
- Weighted F1 weights each class by its number of test images.

The published macro F1 is 99.43%, while weighted F1 is 99.22%. Both are useful because the dataset is
imbalanced. Always inspect individual class support as well: `Moderate Demented` has only 13 test images.

## Why the 100% Moderate result is uncertain

Thirteen correct predictions out of thirteen is the observed result, but it is a small sample. One
error would immediately reduce recall to 12/13, or 92.31%. A percentage without its denominator hides
this uncertainty, which is why the README reports `100% (13 / 13)`.

## Data leakage checks performed

The source images were hashed and the prepared split was checked for exact file duplicates. No exact
SHA-256 hash appeared on both sides of the train/test boundary. This prevents one straightforward form
of leakage: the identical file appearing in both sets.

It does not prevent:

- different slices from the same patient crossing the boundary;
- nearly duplicate images that have been recompressed or slightly transformed;
- dataset-specific processing marks or alignment patterns being learned;
- choices being indirectly influenced by repeatedly viewing test results.

## The central limitation: image split versus patient split

The source does not provide reliable patient IDs. Splitting by image can therefore put correlated
images from one person in both training and test sets. A model can exploit person-specific or
acquisition-specific similarities, producing an optimistic test score.

A stronger evaluation would use a dataset with patient identifiers and group all images from one
patient into exactly one of train, validation, or test. An external dataset collected independently is
stronger still. Until then, describe this result as an image-level benchmark on this dataset.

## Distribution shift

The SVM uses each aligned pixel location as a feature. Performance can change sharply with:

- different cropping or head position;
- different image dimensions, contrast, noise, or scanner;
- another anatomical slice position;
- non-MRI images or incompatible MRI sequences;
- a population unlike the source data.

Resizing an incompatible image to 32x32 makes its dimensions acceptable to the software; it does not
make its content compatible with the learned task.

## Scores are not calibrated probabilities

SVC produces decision-function values that support ranking. The prediction function transforms these
values with softmax for readable relative scores. This does not perform probability calibration.

Therefore:

- do say "the model ranked Non Demented highest for this image";
- do not say "there is a 92% chance the person is non-demented";
- do not use the output for health decisions.

## A responsible result statement

Use wording like:

> The RBF-SVM achieved 99.22% accuracy (1,270/1,280) on a deterministic held-out image split of the
> selected standardized dataset. Exact-file overlap was excluded, but reliable patient identifiers
> were unavailable, so the result is not patient-independent and is not evidence of clinical utility.

Avoid wording such as "the system diagnoses Alzheimer's with 99% accuracy."

## What would be required for stronger evidence

For a serious medical-imaging study, the next steps would include:

1. documented dataset provenance and label-generation procedures;
2. patient-level train/validation/test grouping;
3. preprocessing defined for actual clinical image formats and sequences;
4. model selection without using the final test set;
5. confidence intervals and subgroup analysis;
6. external, multi-site validation;
7. calibration and out-of-distribution detection;
8. review by qualified clinical, statistical, ethical, and regulatory experts.

Those steps are outside this educational repository's intended scope.
