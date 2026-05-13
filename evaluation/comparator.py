"""
Yöntem Karşılaştırma Motoru.

SOLID - Dependency Inversion:
Bu sınıf somut yöntem ve metrik sınıflarına değil,
IDehazingMethod ve IMetric arayüzlerine bağımlıdır.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from tabulate import tabulate

from core.interfaces import IDehazingMethod, IMetric
from evaluation.quality_metrics import get_no_reference_metrics, get_full_reference_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


class MethodComparator:
    """
    Birden fazla dehazing yöntemini çalıştırıp metriklerle karşılaştırır.
    """

    def __init__(
        self,
        methods: list[IDehazingMethod],
        metrics: list[IMetric] | None = None,
        ref_metrics: list[IMetric] | None = None,
    ):
        self._methods = methods
        self._metrics = metrics or get_no_reference_metrics()
        self._ref_metrics = ref_metrics or get_full_reference_metrics()

    def compare_single(
        self,
        image: np.ndarray,
        filename: str = "",
        reference_image: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """
        Tek bir görüntü üzerinde tüm yöntemleri çalıştırıp karşılaştırır.

        Args:
            image: Sisli (hazy) BGR görüntü.
            filename: Dosya adı.
            reference_image: Temiz (clear) referans görüntü (opsiyonel).
                Verilirse PSNR/SSIM gibi full-reference metrikler de hesaplanır.

        Returns:
            {
                "filename": str,
                "has_reference": bool,
                "results": {
                    "DCP": {"image": np.ndarray, "metrics": {...}, "time_ms": float},
                    "CLAHE": {...},
                    ...
                },
                "original_metrics": {...}
            }
        """
        result = {
            "filename": filename,
            "has_reference": reference_image is not None,
            "results": {},
            "original_metrics": {},
        }

        # Orijinal görüntünün no-reference metriklerini hesapla
        for metric in self._metrics:
            result["original_metrics"][metric.get_name()] = metric.calculate(image, image)

        # Orijinal sisli görüntünün referansa göre PSNR/SSIM'ini hesapla (baseline)
        if reference_image is not None:
            for metric in self._ref_metrics:
                try:
                    value = metric.calculate(reference_image, image)
                    result["original_metrics"][metric.get_name()] = round(value, 4)
                except Exception as e:
                    logger.warning(f"  ⚠ Orijinal ref-metrik hatası ({metric.get_name()}): {e}")

        # Her yöntemi çalıştır
        for method in self._methods:
            method_name = method.get_name()
            logger.debug(f"  → {method_name} uygulanıyor...")

            try:
                # Zamanı ölç
                start = time.perf_counter()
                processed = method.process(image)
                elapsed_ms = (time.perf_counter() - start) * 1000

                # No-reference metrikleri hesapla
                metrics_dict = {}
                for metric in self._metrics:
                    value = metric.calculate(image, processed)
                    metrics_dict[metric.get_name()] = round(value, 4)

                # Full-reference metrikleri hesapla (referans varsa)
                if reference_image is not None:
                    # Boyut eşitle
                    ref = reference_image
                    ph, pw = processed.shape[:2]
                    rh, rw = ref.shape[:2]
                    if (ph, pw) != (rh, rw):
                        import cv2
                        ref = cv2.resize(ref, (pw, ph), interpolation=cv2.INTER_AREA)

                    for metric in self._ref_metrics:
                        try:
                            value = metric.calculate(ref, processed)
                            metrics_dict[metric.get_name()] = round(value, 4)
                        except Exception as e:
                            logger.warning(f"  ⚠ Ref-metrik hatası ({metric.get_name()}): {e}")

                metrics_dict["İşlem Süresi (ms)"] = round(elapsed_ms, 2)

                result["results"][method_name] = {
                    "image": processed,
                    "metrics": metrics_dict,
                    "time_ms": elapsed_ms,
                }

            except Exception as e:
                logger.error(f"  ✗ {method_name} hatası: {e}")
                result["results"][method_name] = {
                    "image": None,
                    "metrics": {},
                    "time_ms": 0,
                    "error": str(e),
                }

        return result

    def compare_batch(
        self, images: list[tuple[str, np.ndarray]]
    ) -> list[dict[str, Any]]:
        """
        Bir batch görüntü üzerinde karşılaştırma yapar.

        Args:
            images: (dosya_adı, görüntü) tuple listesi.

        Returns:
            Her görüntü için compare_single sonuçlarının listesi.
        """
        results = []
        for filename, image in images:
            logger.info(f"İşleniyor: {filename}")
            result = self.compare_single(image, filename)
            results.append(result)
        return results

    def aggregate_results(self, all_results: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Tüm sonuçları birleştirip özet DataFrame oluşturur.

        Returns:
            Satırlar: Metrikler, Sütunlar: Yöntemler
        """
        if not all_results:
            return pd.DataFrame()

        # Yöntem adlarını topla
        method_names = []
        for result in all_results:
            for name in result["results"]:
                if name not in method_names:
                    method_names.append(name)

        # Her yöntem için metrik değerlerini topla
        aggregated: dict[str, dict[str, list[float]]] = {
            name: {} for name in method_names
        }

        for result in all_results:
            for method_name, method_result in result["results"].items():
                if "error" in method_result:
                    continue
                for metric_name, value in method_result["metrics"].items():
                    if metric_name not in aggregated[method_name]:
                        aggregated[method_name][metric_name] = []
                    aggregated[method_name][metric_name].append(value)

        # Ortalama hesapla
        summary = {}
        for method_name in method_names:
            summary[method_name] = {}
            for metric_name, values in aggregated[method_name].items():
                if values:
                    summary[method_name][metric_name] = round(np.mean(values), 4)

        df = pd.DataFrame(summary)
        return df

    def print_comparison_table(self, df: pd.DataFrame, total_images: int = 0) -> str:
        """
        Karşılaştırma tablosunu formatlanmış şekilde konsola yazdırır.

        Returns:
            Formatlanmış tablo string'i.
        """
        if df.empty:
            return "Sonuç bulunamadı."

        header = "\n" + "═" * 72
        header += "\n   SİSLİ GÖRÜNTÜ İYİLEŞTİRME SONUÇLARI"
        if total_images > 0:
            header += f"\n   Toplam Görüntü: {total_images}"
        header += "\n" + "═" * 72

        table = tabulate(
            df,
            headers="keys",
            tablefmt="rounded_grid",
            floatfmt=".4f",
            showindex=True,
        )

        # En iyi sonuçları belirle
        best_results = self._find_best_results(df)

        footer = "\n" + "─" * 72
        for metric_name, (best_method, best_value) in best_results.items():
            footer += f"\n  ✅ En iyi {metric_name}: {best_method} ({best_value})"

        output = f"{header}\n\n{table}\n{footer}\n"
        print(output)
        return output

    def _find_best_results(self, df: pd.DataFrame) -> dict[str, tuple[str, float]]:
        """Her metrik için en iyi yöntemi belirler."""
        best = {}
        metric_directions = {}

        # Metrik yönlerini belirle
        for metric in self._metrics:
            metric_directions[metric.get_name()] = metric.higher_is_better

        # Full-reference metrik yönlerini de ekle
        for metric in self._ref_metrics:
            metric_directions[metric.get_name()] = metric.higher_is_better

        # İşlem süresi → düşük daha iyi
        metric_directions["İşlem Süresi (ms)"] = False

        for metric_name in df.index:
            row = df.loc[metric_name]
            higher_better = metric_directions.get(metric_name, True)

            if higher_better:
                best_method = row.idxmax()
            else:
                best_method = row.idxmin()

            best[metric_name] = (best_method, round(row[best_method], 4))

        return best
