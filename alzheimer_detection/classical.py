"""High-accuracy RBF-SVM workflow for aligned 128x128 MRI slice images."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .constants import CLASS_KEYS, CLASS_LABELS, IMAGE_EXTENSIONS
from .dataset import audit_dataset
from .evaluation import _save_confusion_matrix


FEATURE_SIZE = (32, 32)


def image_features(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        resized = image.convert("L").resize(FEATURE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32).reshape(-1) / 255.0


def load_split(split_path: Path, class_directories) -> Tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for label, class_key in enumerate(CLASS_KEYS):
        class_path = split_path / class_directories[class_key]
        paths = sorted(
            path
            for path in class_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for path in paths:
            features.append(image_features(path))
            labels.append(label)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def train_and_evaluate(data_dir: Path, output_dir: Path, c_value: float = 10.0) -> Path:
    summary = audit_dataset(data_dir, verify_images=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(summary.train.path)
    test_path = Path(summary.test.path)
    x_train, y_train = load_split(train_path, summary.train.class_directories)
    x_test, y_test = load_split(test_path, summary.test.class_directories)

    model = make_pipeline(
        StandardScaler(),
        SVC(
            C=c_value,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            cache_size=2048,
        ),
    )
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    labels = list(range(len(CLASS_KEYS)))
    display_names = [CLASS_LABELS[key] for key in CLASS_KEYS]
    report = classification_report(
        y_test,
        predicted,
        labels=labels,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_test, predicted, labels=labels)
    results = {
        "accuracy": float(np.mean(predicted == y_test)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "sample_count": int(y_test.size),
    }

    model_path = output_dir / "best_model.joblib"
    joblib.dump(model, model_path, compress=3)
    (output_dir / "evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    metadata = {
        "model_file": model_path.name,
        "model_type": "scikit-learn RBF SVM",
        "class_keys": list(CLASS_KEYS),
        "class_labels": display_names,
        "directory_names": [summary.train.class_directories[key] for key in CLASS_KEYS],
        "image_size": list(FEATURE_SIZE),
        "color_mode": "grayscale",
        "c_value": c_value,
        "probabilities_calibrated": False,
        "educational_use_only": True,
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    _save_confusion_matrix(matrix, display_names, output_dir / "confusion_matrix.png")
    return model_path


def predict(model_path: Path, image_path: Path, metadata_path: Optional[Path] = None):
    model_path = model_path.expanduser().resolve()
    image_path = image_path.expanduser().resolve()
    metadata_path = metadata_path or model_path.with_name("model_metadata.json")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Model metadata does not exist: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    labels = metadata["class_labels"]
    model = joblib.load(model_path)
    feature = image_features(image_path).reshape(1, -1)
    predicted_index = int(model.predict(feature)[0])
    decision = np.asarray(model.decision_function(feature)[0], dtype=np.float64)
    decision -= decision.max()
    scores = np.exp(decision) / np.exp(decision).sum()
    ranking = sorted(
        (
            {"class": label, "score": float(score)}
            for label, score in zip(labels, scores)
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    return {
        "predicted_class": labels[predicted_index],
        "decision_scores": ranking,
        "scores_are_calibrated_probabilities": False,
        "educational_use_only": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or use the aligned-image RBF SVM.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-dir", type=Path, required=True)
    train_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/svm"))
    train_parser.add_argument("--c", type=float, default=10.0)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--metadata", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "train":
            model_path = train_and_evaluate(args.data_dir, args.output_dir, args.c)
            results = json.loads(
                (args.output_dir / "evaluation.json").read_text(encoding="utf-8")
            )
            print(f"Model saved to: {model_path}")
            print(f"Held-out test accuracy: {results['accuracy']:.4%}")
        else:
            print(json.dumps(predict(args.model, args.image, args.metadata), indent=2))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print("Educational use only; this output is not a medical diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
