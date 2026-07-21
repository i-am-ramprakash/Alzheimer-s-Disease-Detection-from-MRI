"""Evaluation helpers shared by training and the standalone evaluator."""

import json
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

from .constants import CLASS_KEYS, CLASS_LABELS


def evaluate_model(model, dataset, output_dir: Path) -> Dict[str, object]:
    from sklearn.metrics import classification_report, confusion_matrix

    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict(dataset, verbose=0)
    predicted = np.argmax(probabilities, axis=1)
    actual = np.concatenate([labels.numpy() for _, labels in dataset]).astype(int)
    labels = list(range(len(CLASS_KEYS)))
    display_names = [CLASS_LABELS[key] for key in CLASS_KEYS]

    keras_metrics = {
        key: float(value)
        for key, value in model.evaluate(dataset, verbose=0, return_dict=True).items()
    }
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=labels)
    results = {
        "keras": keras_metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(actual.size),
    }

    (output_dir / "evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    _save_confusion_matrix(matrix, display_names, output_dir / "confusion_matrix.png")
    return results


def _save_confusion_matrix(matrix, labels: Sequence[str], output_path: Path) -> None:
    matplotlib_config = output_path.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="Actual class",
        xlabel="Predicted class",
        title="Test-set confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
