import tempfile
import unittest
from pathlib import Path

from PIL import Image

from alzheimer_detection.constants import CLASS_KEYS
from alzheimer_detection.dataset import audit_dataset
from alzheimer_detection.prepare_dataset import prepare_dataset


NAMES = {
    "mild_demented": "MildDemented",
    "moderate_demented": "ModerateDemented",
    "non_demented": "NonDemented",
    "very_mild_demented": "VeryMildDemented",
}


class PrepareDatasetTests(unittest.TestCase):
    def test_stratified_split_is_reproducible_and_auditable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            for class_index, class_key in enumerate(CLASS_KEYS):
                class_dir = source / NAMES[class_key]
                class_dir.mkdir(parents=True)
                for image_index in range(10):
                    Image.new("L", (8, 8), color=class_index * 10 + image_index).save(
                        class_dir / f"{image_index}.png"
                    )

            summary = prepare_dataset(source, output, test_fraction=0.20, seed=42)
            audited = audit_dataset(output, verify_images=True)
            self.assertEqual(audited.train.total, 32)
            self.assertEqual(audited.test.total, 8)
            for class_key in CLASS_KEYS:
                self.assertEqual(summary["classes"][class_key]["train"], 8)
                self.assertEqual(summary["classes"][class_key]["test"], 2)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            output.mkdir()
            with self.assertRaises(ValueError):
                prepare_dataset(source, output)
