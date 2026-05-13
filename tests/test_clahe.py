"""
CLAHE yöntemi unit testleri.
"""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.clahe import CLAHEMethod


class TestCLAHEMethod(unittest.TestCase):
    """CLAHEMethod sınıfı testleri."""

    def setUp(self):
        self.method = CLAHEMethod()

    def test_get_name_lab(self):
        method = CLAHEMethod(color_space="LAB")
        self.assertIn("LAB", method.get_name())

    def test_get_name_hsv(self):
        method = CLAHEMethod(color_space="HSV")
        self.assertIn("HSV", method.get_name())

    def test_process_output_shape(self):
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.shape, image.shape)

    def test_process_output_dtype(self):
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.dtype, np.uint8)

    def test_process_output_range(self):
        image = np.random.randint(50, 200, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_hsv_mode(self):
        method = CLAHEMethod(color_space="HSV")
        image = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
        result = method.process(image)
        self.assertEqual(result.shape, image.shape)

    def test_invalid_color_space(self):
        with self.assertRaises(ValueError):
            CLAHEMethod(color_space="RGB")

    def test_custom_params(self):
        method = CLAHEMethod(clip_limit=5.0, tile_grid_size=(16, 16))
        params = method.get_params()
        self.assertEqual(params["clip_limit"], 5.0)
        self.assertEqual(params["tile_grid_size"], (16, 16))


if __name__ == "__main__":
    unittest.main()
