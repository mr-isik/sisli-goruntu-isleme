"""
Dark Channel Prior (DCP) Sis Giderme Yöntemi.

Referans: He, K., Sun, J., & Tang, X. (2009).
"Single Image Haze Removal Using Dark Channel Prior."
IEEE CVPR, 1956-1963.

Teori:
- Açık hava görüntülerinde, sis içermeyen bölgelerin en az bir renk kanalında
  çok düşük yoğunluk değerleri bulunur (dark channel).
- Sis, bu düşük değerleri yükselterek görüntüyü beyazlaştırır.
- Dark channel'dan transmission map tahmin edilir ve atmosferik ışık modeli
  ile orijinal sahne radyansı kurtarılır.
"""

from typing import Any

import cv2
import numpy as np

from core.interfaces import IDehazingMethod
import config


class DCPMethod(IDehazingMethod):
    """
    Dark Channel Prior tabanlı sis giderme implementasyonu.

    Atmospheric scattering model:
        I(x) = J(x) * t(x) + A * (1 - t(x))

    Burada:
        I(x) = Gözlemlenen sisli görüntü
        J(x) = Sahne radyansı (kurtarılacak)
        t(x) = Transmission map (sis geçirgenliği)
        A    = Global atmosferik ışık
    """

    def __init__(
        self,
        patch_size: int = config.DCP_PATCH_SIZE,
        omega: float = config.DCP_OMEGA,
        t0: float = config.DCP_T0,
        guided_radius: int = config.DCP_GUIDED_RADIUS,
        guided_eps: float = config.DCP_GUIDED_EPS,
        **kwargs,
    ):
        self._patch_size = patch_size
        self._omega = omega
        self._t0 = t0
        self._guided_radius = guided_radius
        self._guided_eps = guided_eps

    # ─── IDehazingMethod Interface ───────────────────────────────────────

    def process(self, image: np.ndarray) -> np.ndarray:
        """DCP tabanlı sis giderme uygular."""
        img = image.astype(np.float64) / 255.0

        # 1. Dark channel hesapla
        dark_channel = self._compute_dark_channel(img)

        # 2. Atmosferik ışık tahmin et
        atmospheric_light = self._estimate_atmospheric_light(img, dark_channel)

        # 3. Transmission map hesapla
        raw_transmission = self._estimate_transmission(img, atmospheric_light)

        # 4. Guided filter ile transmission'ı iyileştir
        gray_guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
        refined_transmission = self._guided_filter(gray_guide, raw_transmission)

        # 5. Sahne radyansını kurtar
        result = self._recover_scene_radiance(img, refined_transmission, atmospheric_light)

        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def get_name(self) -> str:
        return f"DCP (patch={self._patch_size})"

    def get_params(self) -> dict[str, Any]:
        return {
            "patch_size": self._patch_size,
            "omega": self._omega,
            "t0": self._t0,
            "guided_radius": self._guided_radius,
            "guided_eps": self._guided_eps,
        }

    # ─── Private Methods ─────────────────────────────────────────────────

    def _compute_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """
        Dark channel hesaplar: her pikselin komşuluk bölgesindeki
        tüm kanallar arasındaki minimum değer.

        D(x) = min_{y ∈ Ω(x)} ( min_{c ∈ {r,g,b}} I^c(y) )
        """
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self._patch_size, self._patch_size)
        )
        dark = cv2.erode(min_channel, kernel)
        return dark

    def _estimate_atmospheric_light(
        self, image: np.ndarray, dark_channel: np.ndarray
    ) -> np.ndarray:
        """
        Atmosferik ışığı tahmin eder.
        Dark channel'daki en parlak %0.1 piksellerin
        orijinal görüntüdeki karşılıklarının ortalaması alınır.
        """
        h, w = dark_channel.shape
        num_pixels = h * w
        top_count = max(int(num_pixels * 0.001), 1)

        # Dark channel'ı düzleştir ve en parlak piksellerin indekslerini bul
        flat_dark = dark_channel.ravel()
        indices = np.argpartition(flat_dark, -top_count)[-top_count:]

        # Bu indekslere karşılık gelen piksellerin yoğunluk ortalaması
        flat_image = image.reshape(num_pixels, 3)
        atmospheric_light = np.mean(flat_image[indices], axis=0)

        return atmospheric_light

    def _estimate_transmission(
        self, image: np.ndarray, atmospheric_light: np.ndarray
    ) -> np.ndarray:
        """
        Transmission map tahmin eder.

        t(x) = 1 - ω * D( I(x) / A )

        omega: Uzak nesnelerdeki hafif sis hissini korumak için
               tam kaldırmama faktörü.
        """
        # Görüntüyü atmosferik ışığa göre normalize et
        normalized = image / (atmospheric_light + 1e-10)

        # Normalize edilmiş görüntünün dark channel'ını hesapla
        dark_normalized = self._compute_dark_channel(normalized)

        # Transmission map
        transmission = 1.0 - self._omega * dark_normalized

        return transmission

    def _guided_filter(
        self, guide: np.ndarray, source: np.ndarray
    ) -> np.ndarray:
        """
        Guided filter ile transmission map'i kenar-koruyucu şekilde düzleştirir.
        Bu, soft matting yaklaşımının hızlı alternatifidir.
        """
        r = self._guided_radius
        eps = self._guided_eps

        mean_guide = cv2.boxFilter(guide, -1, (r, r))
        mean_source = cv2.boxFilter(source, -1, (r, r))
        mean_gs = cv2.boxFilter(guide * source, -1, (r, r))
        mean_gg = cv2.boxFilter(guide * guide, -1, (r, r))

        cov_gs = mean_gs - mean_guide * mean_source
        var_g = mean_gg - mean_guide * mean_guide

        a = cov_gs / (var_g + eps)
        b = mean_source - a * mean_guide

        mean_a = cv2.boxFilter(a, -1, (r, r))
        mean_b = cv2.boxFilter(b, -1, (r, r))

        result = mean_a * guide + mean_b
        return np.clip(result, self._t0, 1.0)

    def _recover_scene_radiance(
        self,
        image: np.ndarray,
        transmission: np.ndarray,
        atmospheric_light: np.ndarray,
    ) -> np.ndarray:
        """
        Sahne radyansını kurtarır.

        J(x) = (I(x) - A) / max(t(x), t0) + A
        """
        t = np.maximum(transmission, self._t0)
        t_3d = t[:, :, np.newaxis]

        radiance = (image - atmospheric_light) / t_3d + atmospheric_light
        return radiance
