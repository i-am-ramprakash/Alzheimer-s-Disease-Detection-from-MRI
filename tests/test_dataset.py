import tempfile
import unittest
from pathlib import Path

from alzheimer_detection.constants import CLASS_KEYS
from alzheimer_detection.dataset import audit_dataset, canonical_class_name, resolve_layout


DIRECTORY_NAMES = {
    "mild_demented": "MildDemented",
    "moderate_demented": "ModerateDemented",
    "non_demented": "NonDemented",
    "very_mild_demented": "VeryMildDemented",
}


class DatasetTests(unittest.TestCase):
    def test_common_class_names_are_normalized(self):
        self.assertEqual(canonical_class_name("Non_Demented"), "non_demented")
        self.assertEqual(
            canonical_class_name("Very Mild Demented"), "very_mild_demented"
        )
        self.assertIsNone(canonical_class_name("brain_tumor"))

    def test_layout_requires_train_and_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                resolve_layout(Path(temporary))

    def test_audit_counts_supported_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("train", "test"):
                for key in CLASS_KEYS:
                    directory = root / split / DIRECTORY_NAMES[key]
                    directory.mkdir(parents=True)
                    (directory / "sample.jpg").write_bytes(b"fast audit does not decode")

            summary = audit_dataset(root)
            self.assertEqual(summary.train.total, 4)
            self.assertEqual(summary.test.total, 4)
            self.assertIsNone(summary.validation)


if __name__ == "__main__":
    unittest.main()
