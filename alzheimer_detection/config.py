"""Configuration objects used by the training pipeline."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    output_dir: Path
    image_size: Tuple[int, int] = (128, 128)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    validation_fraction: float = 0.20
    seed: int = 42
    verify_images: bool = False

    def validate(self) -> None:
        if min(self.image_size) <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("Batch size and epochs must be positive.")
        if not 0.0 < self.learning_rate:
            raise ValueError("Learning rate must be positive.")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("Validation fraction must be between 0 and 1.")

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["data_dir"] = str(self.data_dir.resolve())
        data["output_dir"] = str(self.output_dir.resolve())
        data["image_size"] = list(self.image_size)
        return data
