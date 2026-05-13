"""
Kalite metrikleri unit testleri.
"""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.quality_metrics import (
    PSNRMetric,
    SSIMMetric,
    EntropyMetric,
    MeanBrightnessMetric,
    ContrastMetric,
    ColorfulnessMetric,
    EdgeIntensityMetric,
    get_all_metrics,
    get_no_reference_metrics,
)


class TestPSNR(unittest.TestCase):
    def test_identical_images(self):
        """Aynı görüntüler arasında PSNR sonsuz (çok yüksek) olmalı."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        metric = PSNRMetric()
        value = metric.calculate(image, image)
        self.assertTrue(value > 50)  # Aynı görüntüde çok yüksek

    def test_different_images(self):
        """Farklı görüntüler arasında PSNR sonlu olmalı."""
        img1 = np.random.randint(0, 128, (50, 50, 3), dtype=np.uint8)
        img2 = np.random.randint(128, 255, (50, 50, 3), dtype=np.uint8)
        metric = PSNRMetric()
        value = metric.calculate(img1, img2)
        self.assertTrue(0 < value < 50)

    def test_higher_is_better(self):
        self.assertTrue(PSNRMetric().higher_is_better)


class TestSSIM(unittest.TestCase):
    def test_identical_images(self):
        """Aynı görüntüler arasında SSIM ~1 olmalı."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        metric = SSIMMetric()
        value = metric.calculate(image, image)
        self.assertAlmostEqual(value, 1.0, places=2)

    def test_higher_is_better(self):
        self.assertTrue(SSIMMetric().higher_is_better)


class TestEntropy(unittest.TestCase):
    def test_uniform_image(self):
        """Tek renkli görüntünün entropy'si 0 olmalı."""
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        metric = EntropyMetric()
        value = metric.calculate(image, image)
        self.assertAlmostEqual(value, 0.0, places=1)

    def test_random_image(self):
        """Rastgele görüntünün entropy'si yüksek olmalı."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        metric = EntropyMetric()
        value = metric.calculate(image, image)
        self.assertTrue(value > 5.0)

    def test_higher_is_better(self):
        self.assertTrue(EntropyMetric().higher_is_better)


class TestContrast(unittest.TestCase):
    def test_uniform_image(self):
        """Tek renkli görüntünün kontrastı 0 olmalı."""
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        metric = ContrastMetric()
        value = metric.calculate(image, image)
        self.assertAlmostEqual(value, 0.0, places=1)

    def test_higher_is_better(self):
        self.assertTrue(ContrastMetric().higher_is_better)


class TestColorfulness(unittest.TestCase):
    def test_grayscale_image(self):
        """Gri tonlamalı görüntü düşük colorfulness'a sahip olmalı."""
        gray_val = 128
        image = np.full((50, 50, 3), gray_val, dtype=np.uint8)
        metric = ColorfulnessMetric()
        value = metric.calculate(image, image)
        self.assertAlmostEqual(value, 0.0, places=1)

    def test_higher_is_better(self):
        self.assertTrue(ColorfulnessMetric().higher_is_better)


class TestGetMetrics(unittest.TestCase):
    def test_no_reference_count(self):
        metrics = get_no_reference_metrics()
        self.assertEqual(len(metrics), 5)

    def test_all_metrics_with_reference(self):
        metrics = get_all_metrics(include_reference=True)
        self.assertEqual(len(metrics), 7)

    def test_all_metrics_no_reference(self):
        metrics = get_all_metrics(include_reference=False)
        self.assertEqual(len(metrics), 5)


if __name__ == "__main__":
    unittest.main()
