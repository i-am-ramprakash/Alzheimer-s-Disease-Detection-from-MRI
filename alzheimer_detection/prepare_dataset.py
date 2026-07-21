"""Create a deterministic stratified train/test layout from class directories."""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Optional, Sequence

from .constants import CLASS_KEYS, IMAGE_EXTENSIONS
from .dataset import canonical_class_name


def _source_directories(source_dir: Path):
    mapped = {}
    for path in sorted(item for item in source_dir.iterdir() if item.is_dir()):
        class_key = canonical_class_name(path.name)
        if class_key is None:
            continue
        if class_key in mapped:
            raise ValueError(f"Multiple source directories map to {class_key!r}.")
        mapped[class_key] = path
    missing = [key for key in CLASS_KEYS if key not in mapped]
    if missing:
        raise ValueError(f"Source dataset is missing classes: {', '.join(missing)}")
    return mapped


def prepare_dataset(
    source_dir: Path,
    output_dir: Path,
    test_fraction: float = 0.20,
    seed: int = 42,
    copy_files: bool = False,
):
    source_dir = source_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("Test fraction must be between 0 and 1.")

    directories = _source_directories(source_dir)
    summary = {"seed": seed, "test_fraction": test_fraction, "classes": {}}
    output_dir.mkdir(parents=True)
    try:
        for class_key in CLASS_KEYS:
            source_class = directories[class_key]
            files = sorted(
                path
                for path in source_class.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if len(files) < 2:
                raise ValueError(f"Class {class_key!r} needs at least two images.")
            random.Random(f"{seed}:{class_key}").shuffle(files)
            test_count = max(1, int(len(files) * test_fraction + 0.5))
            split_counts = {"train": 0, "test": 0}

            for index, source_path in enumerate(files):
                split = "test" if index < test_count else "train"
                relative_path = source_path.relative_to(source_class)
                destination = output_dir / split / source_class.name / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                if copy_files:
                    shutil.copy2(source_path, destination)
                else:
                    try:
                        os.link(source_path, destination)
                    except OSError:
                        shutil.copy2(source_path, destination)
                split_counts[split] += 1
            summary["classes"][class_key] = split_counts

        (output_dir / "split_manifest.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a clean Alzheimer train/test split.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy images instead of using space-saving hard links when possible.",
    )
    args = parser.parse_args(argv)
    try:
        summary = prepare_dataset(
            args.source_dir,
            args.output_dir,
            args.test_fraction,
            args.seed,
            args.copy_files,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
