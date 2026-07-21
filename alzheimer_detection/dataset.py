"""Dataset discovery, validation, and TensorFlow input construction."""

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Tuple

from .constants import CLASS_ALIASES, CLASS_KEYS, IMAGE_EXTENSIONS


def _normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_class_name(directory_name: str) -> Optional[str]:
    """Map common directory-name variations to a canonical class key."""
    return CLASS_ALIASES.get(_normalized_name(directory_name))


@dataclass(frozen=True)
class DatasetLayout:
    train_dir: Path
    test_dir: Path
    validation_dir: Optional[Path]


@dataclass(frozen=True)
class SplitSummary:
    path: str
    class_directories: Mapping[str, str]
    counts: Mapping[str, int]
    total: int


@dataclass(frozen=True)
class DatasetSummary:
    train: SplitSummary
    test: SplitSummary
    validation: Optional[SplitSummary]
    verified_images: bool

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetBundle:
    train: Any
    validation: Any
    test: Any
    actual_class_names: Tuple[str, ...]
    class_weights: Mapping[int, float]


def resolve_layout(data_dir: Path) -> DatasetLayout:
    root = data_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise ValueError(
            "Dataset must contain 'train' and 'test' directories. "
            "See README.md for the expected layout."
        )

    validation_dir = None
    for candidate in (root / "validation", root / "val"):
        if candidate.is_dir():
            validation_dir = candidate
            break
    return DatasetLayout(train_dir, test_dir, validation_dir)


def _class_directories(split_dir: Path) -> Dict[str, Path]:
    mapped: Dict[str, Path] = {}
    unknown = []
    for path in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        canonical = canonical_class_name(path.name)
        if canonical is None:
            unknown.append(path.name)
            continue
        if canonical in mapped:
            raise ValueError(
                f"Multiple directories map to '{canonical}' in {split_dir}: "
                f"{mapped[canonical].name!r} and {path.name!r}."
            )
        mapped[canonical] = path

    missing = [name for name in CLASS_KEYS if name not in mapped]
    if missing:
        details = f" Missing classes: {', '.join(missing)}."
        if unknown:
            details += f" Unrecognized directories: {', '.join(unknown)}."
        raise ValueError(f"Invalid class layout in {split_dir}.{details}")
    return mapped


def _image_paths(class_dir: Path):
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _summarize_split(split_dir: Path, verify_images: bool) -> SplitSummary:
    directories = _class_directories(split_dir)
    counts: Dict[str, int] = {}
    invalid = []

    for class_key in CLASS_KEYS:
        paths = _image_paths(directories[class_key])
        counts[class_key] = len(paths)
        if not paths:
            raise ValueError(
                f"Class '{class_key}' contains no supported images in {split_dir}."
            )
        if verify_images:
            from PIL import Image, UnidentifiedImageError

            for path in paths:
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError) as exc:
                    invalid.append(f"{path}: {exc}")

    if invalid:
        preview = "\n".join(invalid[:10])
        suffix = "" if len(invalid) <= 10 else f"\n...and {len(invalid) - 10} more"
        raise ValueError(f"Unreadable images detected:\n{preview}{suffix}")

    return SplitSummary(
        path=str(split_dir),
        class_directories={key: directories[key].name for key in CLASS_KEYS},
        counts=counts,
        total=sum(counts.values()),
    )


def audit_dataset(data_dir: Path, verify_images: bool = False) -> DatasetSummary:
    layout = resolve_layout(data_dir)
    return DatasetSummary(
        train=_summarize_split(layout.train_dir, verify_images),
        test=_summarize_split(layout.test_dir, verify_images),
        validation=(
            _summarize_split(layout.validation_dir, verify_images)
            if layout.validation_dir
            else None
        ),
        verified_images=verify_images,
    )


def _ordered_actual_names(summary: SplitSummary) -> Tuple[str, ...]:
    return tuple(summary.class_directories[key] for key in CLASS_KEYS)


def _class_weights(counts: Mapping[str, int]) -> Dict[int, float]:
    total = sum(counts.values())
    classes = len(CLASS_KEYS)
    return {
        index: total / (classes * counts[key])
        for index, key in enumerate(CLASS_KEYS)
    }


def load_tensorflow_datasets(
    summary: DatasetSummary,
    image_size: Tuple[int, int],
    batch_size: int,
    validation_fraction: float,
    seed: int,
    color_mode: str = "grayscale",
) -> DatasetBundle:
    """Create lazy TensorFlow datasets after the filesystem audit succeeds."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    class_names = _ordered_actual_names(summary.train)
    common = dict(
        labels="inferred",
        label_mode="int",
        class_names=list(class_names),
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
    )

    if summary.validation is None:
        train = tf.keras.utils.image_dataset_from_directory(
            summary.train.path,
            validation_split=validation_fraction,
            subset="training",
            seed=seed,
            shuffle=True,
            **common,
        )
        validation = tf.keras.utils.image_dataset_from_directory(
            summary.train.path,
            validation_split=validation_fraction,
            subset="validation",
            seed=seed,
            shuffle=True,
            **common,
        )
    else:
        validation_names = _ordered_actual_names(summary.validation)
        if validation_names != class_names:
            raise ValueError("Training and validation class directory names must match.")
        train = tf.keras.utils.image_dataset_from_directory(
            summary.train.path, seed=seed, shuffle=True, **common
        )
        validation = tf.keras.utils.image_dataset_from_directory(
            summary.validation.path, shuffle=False, **common
        )

    test_names = _ordered_actual_names(summary.test)
    if test_names != class_names:
        raise ValueError("Training and test class directory names must match.")
    test = tf.keras.utils.image_dataset_from_directory(
        summary.test.path, shuffle=False, **common
    )

    autotune = tf.data.AUTOTUNE
    return DatasetBundle(
        train=train.prefetch(autotune),
        validation=validation.prefetch(autotune),
        test=test.prefetch(autotune),
        actual_class_names=class_names,
        class_weights=_class_weights(summary.train.counts),
    )
