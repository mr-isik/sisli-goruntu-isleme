"""
Retinex yöntemi unit testleri.
"""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.retinex import RetinexMethod


class TestRetinexMethod(unittest.TestCase):
    """RetinexMethod sınıfı testleri."""

    def setUp(self):
        self.method = RetinexMethod()

    def test_get_name(self):
        name = self.method.get_name()
        self.assertIn("MSRCR", name)

    def test_get_params(self):
        params = self.method.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn("sigma_list", params)
        self.assertIn("gain", params)

    def test_process_output_shape(self):
        image = np.random.randint(30, 220, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.shape, image.shape)

    def test_process_output_dtype(self):
        image = np.random.randint(30, 220, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.dtype, np.uint8)

    def test_process_output_range(self):
        image = np.random.randint(30, 220, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_custom_sigmas(self):
        method = RetinexMethod(sigma_list=[10, 50, 100])
        params = method.get_params()
        self.assertEqual(params["sigma_list"], [10, 50, 100])

    def test_process_dark_image(self):
        """Karanlık görüntü işlenebilmeli."""
        image = np.random.randint(5, 30, (80, 80, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.shape, image.shape)


if __name__ == "__main__":
    unittest.main()
