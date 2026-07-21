import tempfile
import unittest
from pathlib import Path

from PIL import Image

from alzheimer_detection.classical import FEATURE_SIZE, image_features


class ClassicalFeatureTests(unittest.TestCase):
    def test_image_features_are_normalized_and_fixed_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "image.png"
            Image.new("L", (128, 128), color=255).save(path)
            features = image_features(path)
            self.assertEqual(features.shape, (FEATURE_SIZE[0] * FEATURE_SIZE[1],))
            self.assertAlmostEqual(float(features.min()), 1.0)
            self.assertAlmostEqual(float(features.max()), 1.0)
