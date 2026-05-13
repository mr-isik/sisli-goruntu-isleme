"""
CLAHE (Contrast Limited Adaptive Histogram Equalization) Sis Giderme Yöntemi.

Referans: Zuiderveld, K. (1994).
"Contrast Limited Adaptive Histogram Equalization."
Graphics Gems IV, Academic Press, 474-485.

Teori:
- Sisli görüntülerin histogramı dar bir aralıkta yoğunlaşır.
- CLAHE, görüntüyü küçük bölgelere ayırıp her bölgenin histogramını
  ayrı ayrı eşitler. Clip limit ile aşırı amplifikasyonu önler.
- LAB veya HSV renk uzayında sadece parlaklık kanalına uygulanarak
  renk bozulması minimize edilir.
"""

from typing import Any

import cv2
import numpy as np

from core.interfaces import IDehazingMethod
import config


class CLAHEMethod(IDehazingMethod):
    """
    CLAHE tabanlı sis giderme implementasyonu.

    LAB uzayında L kanalına veya HSV uzayında V kanalına
    adaptive histogram equalization uygular.
    """

    def __init__(
        self,
        clip_limit: float = config.CLAHE_CLIP_LIMIT,
        tile_grid_size: tuple[int, int] = config.CLAHE_TILE_GRID_SIZE,
        color_space: str = config.CLAHE_COLOR_SPACE,
        **kwargs,
    ):
        self._clip_limit = clip_limit
        self._tile_grid_size = tile_grid_size
        self._color_space = color_space.upper()

        if self._color_space not in ("LAB", "HSV"):
            raise ValueError(f"Desteklenmeyen renk uzayı: {self._color_space}. 'LAB' veya 'HSV' kullanın.")

    # ─── IDehazingMethod Interface ───────────────────────────────────────

    def process(self, image: np.ndarray) -> np.ndarray:
        """CLAHE tabanlı kontrast iyileştirme uygular."""
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit,
            tileGridSize=self._tile_grid_size,
        )

        if self._color_space == "LAB":
            return self._process_lab(image, clahe)
        else:
            return self._process_hsv(image, clahe)

    def get_name(self) -> str:
        return f"CLAHE ({self._color_space})"

    def get_params(self) -> dict[str, Any]:
        return {
            "clip_limit": self._clip_limit,
            "tile_grid_size": self._tile_grid_size,
            "color_space": self._color_space,
        }

    # ─── Private Methods ─────────────────────────────────────────────────

    def _process_lab(self, image: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
        """
        LAB renk uzayında CLAHE uygular.

        LAB uzayı:
        - L: Parlaklık (Lightness) → CLAHE burada uygulanır
        - A: Yeşil-kırmızı renk ekseni
        - B: Mavi-sarı renk ekseni

        Avantaj: Renk bilgisini korur, sadece parlaklığı düzenler.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        channels = list(cv2.split(lab))

        # L kanalına CLAHE uygula
        channels[0] = clahe.apply(channels[0])

        lab_result = cv2.merge(channels)
        return cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

    def _process_hsv(self, image: np.ndarray, clahe: cv2.CLAHE) -> np.ndarray:
        """
        HSV renk uzayında CLAHE uygular.

        HSV uzayı:
        - H: Ton (Hue) → Dokunulmaz
        - S: Doygunluk (Saturation) → Dokunulmaz
        - V: Değer/Parlaklık (Value) → CLAHE burada uygulanır

        Avantaj: Ton ve doygunluğu korur.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        channels = list(cv2.split(hsv))

        # V kanalına CLAHE uygula
        channels[2] = clahe.apply(channels[2])

        hsv_result = cv2.merge(channels)
        return cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)
