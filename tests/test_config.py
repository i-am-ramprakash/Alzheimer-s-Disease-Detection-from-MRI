import tempfile
import unittest
from pathlib import Path

from alzheimer_detection.config import TrainingConfig


class ConfigTests(unittest.TestCase):
    def test_default_configuration_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            TrainingConfig(root, root / "output").validate()

    def test_invalid_validation_fraction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = TrainingConfig(root, root / "output", validation_fraction=1.0)
            with self.assertRaises(ValueError):
                config.validate()


if __name__ == "__main__":
    unittest.main()
