"""
DCP (Dark Channel Prior) yöntemi unit testleri.
"""

import sys
from pathlib import Path
import unittest

import numpy as np

# Proje kökünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent))

from methods.dcp import DCPMethod


class TestDCPMethod(unittest.TestCase):
    """DCPMethod sınıfı testleri."""

    def setUp(self):
        """Her test için taze bir DCP instance oluştur."""
        self.method = DCPMethod()

    def test_get_name(self):
        """Yöntem adı doğru döndürülmeli."""
        self.assertEqual(self.method.get_name(), "DCP (Dark Channel Prior)")

    def test_get_params(self):
        """Parametreler dict olarak döndürülmeli."""
        params = self.method.get_params()
        self.assertIsInstance(params, dict)
        self.assertIn("patch_size", params)
        self.assertIn("omega", params)
        self.assertIn("t0", params)

    def test_process_output_shape(self):
        """Çıktı, girdi ile aynı boyutta olmalı."""
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.shape, image.shape)

    def test_process_output_dtype(self):
        """Çıktı uint8 olmalı."""
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.dtype, np.uint8)

    def test_process_output_range(self):
        """Çıktı değerleri 0-255 aralığında olmalı."""
        image = np.random.randint(50, 200, (100, 150, 3), dtype=np.uint8)
        result = self.method.process(image)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 255))

    def test_process_white_image(self):
        """Tamamen beyaz (sisli) görüntü işlenebilmeli."""
        image = np.full((80, 80, 3), 240, dtype=np.uint8)
        result = self.method.process(image)
        self.assertEqual(result.shape, image.shape)

    def test_custom_params(self):
        """Özel parametreler ile oluşturulabilmeli."""
        method = DCPMethod(patch_size=7, omega=0.8, t0=0.2)
        params = method.get_params()
        self.assertEqual(params["patch_size"], 7)
        self.assertEqual(params["omega"], 0.8)
        self.assertEqual(params["t0"], 0.2)


if __name__ == "__main__":
    unittest.main()
