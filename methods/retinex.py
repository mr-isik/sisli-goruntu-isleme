"""
Multi-Scale Retinex with Color Restoration (MSRCR) Sis Giderme Yöntemi.

Referans: Jobson, D.J., Rahman, Z., & Woodell, G.A. (1997).
"A Multiscale Retinex for Bridging the Gap Between Color Images
and the Human Observation of Scenes."
IEEE Transactions on Image Processing, 6(7), 965-976.

Teori:
- Retinex teorisi (Edwin Land, 1971): İnsan görme sistemi bir sahnenin
  aydınlatmasını (illumination) yansıtıcılığından (reflectance) ayırır.
- Matematiksel olarak: I(x,y) = R(x,y) * L(x,y)
  → log(R) = log(I) - log(L)
- Aydınlatma, Gaussian blur ile tahmin edilir.
- Farklı sigma değerleri ile çoklu ölçekte (multi-scale) hesaplanarak
  hem lokal hem global kontrast iyileştirilir.
- Color Restoration, renk tutarlılığını sağlar.
"""

from typing import Any

import cv2
import numpy as np

from core.interfaces import IDehazingMethod
import config


class RetinexMethod(IDehazingMethod):
    """
    Multi-Scale Retinex with Color Restoration (MSRCR) implementasyonu.

    SSR → MSR → MSRCR pipeline'ı uygular.
    """

    def __init__(
        self,
        sigma_list: list[float] = None,
        gain: float = config.RETINEX_GAIN,
        offset: float = config.RETINEX_OFFSET,
        color_restore_factor: float = config.RETINEX_COLOR_RESTORE_FACTOR,
        color_restore_gain: float = config.RETINEX_COLOR_RESTORE_GAIN,
        **kwargs,
    ):
        self._sigma_list = sigma_list or list(config.RETINEX_SIGMA_LIST)
        self._gain = gain
        self._offset = offset
        self._color_restore_factor = color_restore_factor
        self._color_restore_gain = color_restore_gain

    # ─── IDehazingMethod Interface ───────────────────────────────────────

    def process(self, image: np.ndarray) -> np.ndarray:
        """MSRCR tabanlı sis giderme uygular."""
        img = image.astype(np.float64) + 1.0  # log(0) önleme

        # 1. Multi-Scale Retinex hesapla
        msr = self._multi_scale_retinex(img)

        # 2. Color Restoration faktörü hesapla ve uygula
        color_restoration = self._compute_color_restoration(img)
        msrcr = color_restoration * msr

        # 3. Gain/Offset ile normalize et
        result = self._gain * (msrcr - np.mean(msrcr)) / (np.std(msrcr) + 1e-10) + self._offset

        # 4. Değer aralığını kırp ve uint8'e çevir
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_name(self) -> str:
        sigmas = ",".join(str(int(s)) for s in self._sigma_list)
        return f"MSRCR (σ=[{sigmas}])"

    def get_params(self) -> dict[str, Any]:
        return {
            "sigma_list": self._sigma_list,
            "gain": self._gain,
            "offset": self._offset,
            "color_restore_factor": self._color_restore_factor,
            "color_restore_gain": self._color_restore_gain,
        }

    # ─── Private Methods ─────────────────────────────────────────────────

    def _single_scale_retinex(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """
        Tek ölçekli Retinex (SSR).

        R(x,y) = log(I(x,y)) - log(I(x,y) * G(x,y,σ))

        G: Gaussian kernel (aydınlatma tahmini)
        σ: Gaussian standart sapması
          - Küçük σ → Lokal kontrast iyileştirme (detay)
          - Büyük σ → Global kontrast iyileştirme (ton)
        """
        # Kernel boyutu sigma'nın en az 6 katı olmalı (3σ kuralı her yönde)
        ksize = int(np.ceil(sigma * 6)) | 1  # Tek sayı garantisi
        blur = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        retinex = np.log10(image) - np.log10(blur + 1.0)
        return retinex

    def _multi_scale_retinex(self, image: np.ndarray) -> np.ndarray:
        """
        Çok ölçekli Retinex (MSR).

        MSR = (1/N) * Σ SSR(σ_i)

        Farklı ölçeklerdeki SSR'ların ağırlıklı ortalaması.
        Hem lokal detayları hem global tonu korur.
        """
        retinex = np.zeros_like(image, dtype=np.float64)
        weight = 1.0 / len(self._sigma_list)

        for sigma in self._sigma_list:
            retinex += weight * self._single_scale_retinex(image, sigma)

        return retinex

    def _compute_color_restoration(self, image: np.ndarray) -> np.ndarray:
        """
        Renk geri yükleme faktörü (Color Restoration Function).

        C(x,y) = β * log( α * I_c(x,y) / Σ I_k(x,y) )

        Kanallar arası renk oranını koruyarak renk kaymasını önler.
        """
        # Her pikselin toplam yoğunluğu
        channel_sum = np.sum(image, axis=2, keepdims=True) + 1e-10

        # Renk oranı
        color_ratio = image / channel_sum

        # Color restoration
        cr = self._color_restore_gain * np.log10(
            1.0 + self._color_restore_factor * color_ratio
        )

        return cr
