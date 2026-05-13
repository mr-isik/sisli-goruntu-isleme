"""
Görüntü okuma/yazma yardımcı fonksiyonları.
"""

from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


def load_image(filepath: str | Path) -> np.ndarray | None:
    """
    Görüntü dosyasını BGR formatında yükler.

    Args:
        filepath: Dosya yolu.

    Returns:
        BGR numpy dizisi veya None (yükleme başarısızsa).
    """
    filepath = str(filepath)
    image = cv2.imread(filepath)
    if image is None:
        logger.error(f"Görüntü okunamadı: {filepath}")
    return image


def save_image(image: np.ndarray, filepath: str | Path) -> bool:
    """
    Görüntüyü dosyaya kaydeder.

    Args:
        image: BGR numpy dizisi.
        filepath: Hedef dosya yolu.

    Returns:
        Başarılı ise True.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    try:
        success = cv2.imwrite(str(filepath), image)
        if success:
            logger.debug(f"Kaydedildi: {filepath}")
        else:
            logger.error(f"Kayıt başarısız: {filepath}")
        return success
    except Exception as e:
        logger.error(f"Kayıt hatası ({filepath}): {e}")
        return False


def resize_if_needed(
    image: np.ndarray, max_dimension: int = 2048
) -> np.ndarray:
    """Görüntü çok büyükse küçültür."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dimension:
        return image

    scale = max_dimension / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
