"""
Görüntü kalite metrikleri.

SOLID - Single Responsibility:
Her metrik sınıfı tek bir kalite ölçütünü hesaplar.

SOLID - Liskov Substitution:
Tüm metrikler IMetric arayüzünü tam olarak implemente eder.

Metrikler iki kategoriye ayrılır:
1. Referanslı (Full-Reference): Orijinal ve işlenmiş çift gerektirir (PSNR, SSIM)
2. Referanssız (No-Reference): Sadece işlenmiş görüntü yeterli
   (Entropy, Contrast, Colorfulness, Edge Intensity, Mean Brightness)
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from core.interfaces import IMetric


# ═══════════════════════════════════════════════════════════════════════════
# REFERANSLI METRİKLER (Full-Reference)
# ═══════════════════════════════════════════════════════════════════════════


class PSNRMetric(IMetric):
    """
    Peak Signal-to-Noise Ratio (PSNR).

    PSNR = 10 * log10(MAX² / MSE)

    - Yüksek PSNR → Düşük bozulma
    - Tipik değerler: 20-40 dB (> 30 dB iyi kabul edilir)
    - Ground-truth gerektirir.
    """

    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        return float(peak_signal_noise_ratio(original, processed))

    def get_name(self) -> str:
        return "PSNR (dB)"

    @property
    def higher_is_better(self) -> bool:
        return True


class SSIMMetric(IMetric):
    """
    Structural Similarity Index (SSIM).

    Parlaklık, kontrast ve yapısal benzerliği birleştirir.
    - Değer aralığı: [-1, 1] (1 = mükemmel benzerlik)
    - İnsan algısına PSNR'dan daha yakın sonuçlar verir.
    - Ground-truth gerektirir.
    """

    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        # Çok kanallı görüntüler için channel_axis belirt
        min_dim = min(original.shape[0], original.shape[1])
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        return float(
            structural_similarity(
                original, processed, multichannel=True, channel_axis=2, win_size=win_size
            )
        )

    def get_name(self) -> str:
        return "SSIM"

    @property
    def higher_is_better(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════
# REFERANSSIZ METRİKLER (No-Reference)
# ═══════════════════════════════════════════════════════════════════════════


class EntropyMetric(IMetric):
    """
    Shannon Entropy (Bilgi İçeriği).

    H = -Σ p(x) * log2(p(x))

    - Yüksek entropy → Daha fazla bilgi / detay
    - Sisli görüntüler genellikle düşük entropy'ye sahiptir.
    - Değer aralığı: [0, 8] (8-bit görüntü için)
    """

    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        histogram = histogram / histogram.sum()

        # Sıfır olasılıkları filtrele
        nonzero = histogram[histogram > 0]
        entropy = -np.sum(nonzero * np.log2(nonzero))
        return float(entropy)

    def get_name(self) -> str:
        return "Entropy"

    @property
    def higher_is_better(self) -> bool:
        return True


class MeanBrightnessMetric(IMetric):
    """
    Ortalama Parlaklık.

    Görüntünün genel aydınlık seviyesi.
    - İdeal değer ~128 (orta ton)
    - Çok düşük → Karanlık, çok yüksek → Aşırı parlak/soluk
    """

    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def get_name(self) -> str:
        return "Ort. Parlaklık"

    @property
    def higher_is_better(self) -> bool:
        # Parlaklık için "higher is better" değil; 128'e yakınlık önemli
        # Ancak sisli görüntüler genelde parlak, dehazing sonrası azalır
        return False


class ContrastMetric(IMetric):
    """
    Kontrast (Standart Sapma tabanlı).

    σ = sqrt( (1/N) * Σ(I(x) - μ)² )

    - Yüksek std → Yüksek kontrast
    - Sisli görüntüler düşük kontrasta sahiptir.
    """

    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY).astype(np.float64)
        return float(np.std(gray))

    def get_name(self) -> str:
        return "Kontrast (Std)"

    @property
    def higher_is_better(self) -> bool:
        return True


class ColorfulnessMetric(IMetric):
    """
    Renk Canlılığı (Colorfulness).

    Hasler & Süsstrunk (2003) metriği:
    M = sqrt(σ_rgyb² + μ_rgyb²)

    - Yüksek değer → Daha canlı renkler
    - Sisli görüntüler genelde soluk renklere sahiptir.
    """

    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        b, g, r = processed[:, :, 0].astype(np.float64), \
                   processed[:, :, 1].astype(np.float64), \
                   processed[:, :, 2].astype(np.float64)

        # rg ve yb renk karşıtlık kanalları
        rg = r - g
        yb = 0.5 * (r + g) - b

        # Her kanalın standart sapması ve ortalaması
        sigma = np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
        mu = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)

        colorfulness = sigma + 0.3 * mu
        return float(colorfulness)

    def get_name(self) -> str:
        return "Renk Canlılığı"

    @property
    def higher_is_better(self) -> bool:
        return True


class EdgeIntensityMetric(IMetric):
    """
    Kenar Yoğunluğu (Sobel tabanlı).

    Sobel operatörü ile kenar büyüklüğü hesaplanır.
    - Yüksek değer → Daha belirgin kenarlar / detay
    - Sis, kenarları bulanıklaştırır.
    """

    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Sobel gradyanları
        sobel_x = ndimage.sobel(gray, axis=1)
        sobel_y = ndimage.sobel(gray, axis=0)

        # Gradient büyüklüğü
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        return float(np.mean(magnitude))

    def get_name(self) -> str:
        return "Kenar Yoğunluğu"

    @property
    def higher_is_better(self) -> bool:
        return True


# ─── Yardımcı: Tüm metrikleri listele ───────────────────────────────────

def get_no_reference_metrics() -> list[IMetric]:
    """Referanssız metriklerin listesini döndürür."""
    return [
        EntropyMetric(),
        ContrastMetric(),
        ColorfulnessMetric(),
        EdgeIntensityMetric(),
        MeanBrightnessMetric(),
    ]


def get_full_reference_metrics() -> list[IMetric]:
    """Referanslı metriklerin listesini döndürür."""
    return [
        PSNRMetric(),
        SSIMMetric(),
    ]


def get_all_metrics(include_reference: bool = False) -> list[IMetric]:
    """Tüm metriklerin listesini döndürür."""
    metrics = get_no_reference_metrics()
    if include_reference:
        metrics = get_full_reference_metrics() + metrics
    return metrics
