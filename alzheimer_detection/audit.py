"""Dataset-audit command that does not require TensorFlow."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .dataset import audit_dataset


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Alzheimer image dataset.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--verify-images", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = audit_dataset(args.data_dir, args.verify_images)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
