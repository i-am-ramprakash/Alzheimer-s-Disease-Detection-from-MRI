"""Single-image prediction command."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def predict(model_path: Path, image_path: Path, metadata_path: Optional[Path] = None):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

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
    image_size = tuple(metadata["image_size"])
    labels = metadata["class_labels"]
    color_mode = metadata.get("color_mode", "grayscale")
    image = tf.keras.utils.load_img(
        image_path, color_mode=color_mode, target_size=image_size
    )
    array = tf.keras.utils.img_to_array(image)
    probabilities = model_predict(tf, model_path, array)
    ranking = sorted(
        (
            {"class": label, "probability": float(probability)}
            for label, probability in zip(labels, probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "predicted_class": ranking[0]["class"],
        "confidence": ranking[0]["probability"],
        "probabilities": ranking,
        "educational_use_only": True,
    }


def model_predict(tf, model_path: Path, image_array: np.ndarray) -> np.ndarray:
    model = tf.keras.models.load_model(str(model_path))
    return model.predict(np.expand_dims(image_array, axis=0), verbose=0)[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one Alzheimer MRI image.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = predict(args.model, args.image, args.metadata)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(result, indent=2))
    print("Educational use only; this output is not a medical diagnosis.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
