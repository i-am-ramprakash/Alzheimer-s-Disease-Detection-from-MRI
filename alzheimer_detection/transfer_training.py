"""Two-stage MobileNetV2 transfer-learning workflow."""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .constants import CLASS_KEYS, CLASS_LABELS
from .dataset import audit_dataset, load_tensorflow_datasets
from .evaluation import evaluate_model
from .model import (
    build_transfer_model,
    compile_transfer_model,
    enable_transfer_fine_tuning,
)


def _callbacks(tf, output_dir: Path, checkpoint_name: str, append: bool):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            str(output_dir / checkpoint_name), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.35, patience=2, min_lr=1e-7
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_dir / "training_history.csv"), append=append
        ),
    ]


def train_transfer(
    data_dir: Path,
    output_dir: Path,
    image_size: int = 96,
    batch_size: int = 64,
    frozen_epochs: int = 12,
    fine_tune_epochs: int = 18,
    seed: int = 42,
) -> Path:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = audit_dataset(data_dir)
    bundle = load_tensorflow_datasets(
        summary,
        (image_size, image_size),
        batch_size,
        0.20,
        seed,
        color_mode="rgb",
    )
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    settings = {
        "architecture": "MobileNetV2 ImageNet transfer learning",
        "image_size": [image_size, image_size],
        "batch_size": batch_size,
        "frozen_epochs": frozen_epochs,
        "fine_tune_epochs": fine_tune_epochs,
        "seed": seed,
        "class_weight_strategy": "square-root balanced",
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    model, base_model = build_transfer_model((image_size, image_size), 3e-4)
    class_weights = {
        index: float(weight) ** 0.5
        for index, weight in bundle.class_weights.items()
    }
    frozen_history = model.fit(
        bundle.train,
        validation_data=bundle.validation,
        epochs=frozen_epochs,
        class_weight=class_weights,
        callbacks=_callbacks(tf, output_dir, "best_frozen.h5", append=False),
        verbose=2,
    )

    # Continue from the best frozen checkpoint before fine-tuning feature layers.
    model = tf.keras.models.load_model(str(output_dir / "best_frozen.h5"))
    base_model = next(
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model) and layer.name.startswith("mobilenetv2")
    )
    enable_transfer_fine_tuning(base_model, trainable_layers=30)
    compile_transfer_model(model, 1e-5)
    completed_frozen_epochs = len(frozen_history.history["loss"])
    fine_history = model.fit(
        bundle.train,
        validation_data=bundle.validation,
        initial_epoch=completed_frozen_epochs,
        epochs=completed_frozen_epochs + fine_tune_epochs,
        class_weight=class_weights,
        callbacks=_callbacks(tf, output_dir, "best_finetuned.h5", append=True),
        verbose=2,
    )

    combined_history = {
        key: [float(value) for value in frozen_history.history.get(key, [])]
        + [float(value) for value in fine_history.history.get(key, [])]
        for key in set(frozen_history.history) | set(fine_history.history)
    }
    (output_dir / "history.json").write_text(
        json.dumps(combined_history, indent=2), encoding="utf-8"
    )
    metadata = {
        "model_file": "best_model.h5",
        "architecture": "MobileNetV2",
        "class_keys": list(CLASS_KEYS),
        "class_labels": [CLASS_LABELS[key] for key in CLASS_KEYS],
        "directory_names": list(bundle.actual_class_names),
        "image_size": [image_size, image_size],
        "color_mode": "rgb",
        "educational_use_only": True,
    }
    (output_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    frozen_model = tf.keras.models.load_model(str(output_dir / "best_frozen.h5"))
    fine_model = tf.keras.models.load_model(str(output_dir / "best_finetuned.h5"))
    frozen_loss = float(frozen_model.evaluate(bundle.validation, verbose=0)[0])
    fine_loss = float(fine_model.evaluate(bundle.validation, verbose=0)[0])
    selected_path = (
        output_dir / "best_finetuned.h5"
        if fine_loss < frozen_loss
        else output_dir / "best_frozen.h5"
    )
    shutil.copy2(selected_path, output_dir / "best_model.h5")
    best_model = tf.keras.models.load_model(str(output_dir / "best_model.h5"))
    evaluate_model(best_model, bundle.test, output_dir)
    return output_dir / "best_model.h5"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the higher-accuracy MobileNetV2 Alzheimer classifier."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transfer"))
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--frozen-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=18)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        path = train_transfer(
            args.data_dir,
            args.output_dir,
            args.image_size,
            args.batch_size,
            args.frozen_epochs,
            args.fine_tune_epochs,
            args.seed,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Best transfer model saved to: {path}")
    print("Educational use only; this output is not a medical diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
