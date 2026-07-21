"""Command-line training workflow."""

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import TrainingConfig
from .constants import CLASS_KEYS, CLASS_LABELS
from .dataset import audit_dataset, load_tensorflow_datasets
from .evaluation import evaluate_model
from .model import build_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the educational four-class Alzheimer MRI classifier."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/latest"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-images", action="store_true", help="Open every image before training."
    )
    return parser


def train(config: TrainingConfig) -> Path:
    config.validate()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.keras.utils.set_random_seed(config.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    summary = audit_dataset(config.data_dir, config.verify_images)
    bundle = load_tensorflow_datasets(
        summary,
        config.image_size,
        config.batch_size,
        config.validation_fraction,
        config.seed,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "dataset_audit.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    (config.output_dir / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    # TensorFlow 2.13 on Windows rejects callback options for the native .keras
    # format. HDF5 provides a reliable full-model checkpoint for this environment.
    best_model_path = config.output_dir / "best_model.h5"
    model = build_model(config.image_size, config.learning_rate)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best_model_path), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.CSVLogger(str(config.output_dir / "training_history.csv")),
    ]
    history = model.fit(
        bundle.train,
        validation_data=bundle.validation,
        epochs=config.epochs,
        class_weight=dict(bundle.class_weights),
        callbacks=callbacks,
        verbose=2,
    )
    (config.output_dir / "history.json").write_text(
        json.dumps(
            {key: [float(value) for value in values] for key, values in history.history.items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    best_model = tf.keras.models.load_model(str(best_model_path))
    metadata = {
        "model_file": best_model_path.name,
        "class_keys": list(CLASS_KEYS),
        "class_labels": [CLASS_LABELS[key] for key in CLASS_KEYS],
        "directory_names": list(bundle.actual_class_names),
        "image_size": list(config.image_size),
        "color_mode": "grayscale",
        "educational_use_only": True,
    }
    (config.output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    evaluate_model(best_model, bundle.test, config.output_dir)
    return best_model_path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    config = TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        verify_images=args.verify_images,
    )
    try:
        path = train(config)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Best model saved to: {path}")
    print("Educational use only; this output is not a medical diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
