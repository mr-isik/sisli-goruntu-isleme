"""
Paralel Batch İşleme Pipeline'ı.

SOLID - Single Responsibility:
Yalnızca batch'leri paralel olarak işlemek ve sonuçları toplamakla ilgilenir.

SOLID - Dependency Inversion:
IDehazingMethod ve IDataLoader arayüzlerine bağımlı, somut sınıflara değil.
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
from tqdm import tqdm

from core.interfaces import IDehazingMethod, IDataLoader
from evaluation.comparator import MethodComparator
from evaluation.quality_metrics import get_no_reference_metrics, IMetric
from utils.logger import get_logger
import config

logger = get_logger(__name__)


def _unpack_batch_item(item):
    """
    Batch elemanını (2-tuple veya 3-tuple) normalize eder.

    Returns:
        (filename, image, reference_image_or_None)
    """
    if len(item) == 3:
        return item[0], item[1], item[2]
    else:
        return item[0], item[1], None


def _process_single_image(
    image_data: tuple,
    method_configs: list[dict],
    metric_names: list[str],
) -> dict[str, Any] | None:
    """
    Tek bir görüntüyü tüm yöntemlerle işler (process worker).

    Not: ProcessPoolExecutor ile çalışabilmesi için top-level fonksiyon
    olmalı (pickle serializable).
    """
    from methods import create_method
    from evaluation.quality_metrics import get_no_reference_metrics

    filename, image, reference = _unpack_batch_item(image_data)

    try:
        methods = [create_method(cfg["name"], **cfg.get("params", {})) for cfg in method_configs]
        metrics = get_no_reference_metrics()
        comparator = MethodComparator(methods, metrics)
        result = comparator.compare_single(image, filename, reference_image=reference)

        # Görüntüleri bellekten kaldır (sadece metrikleri tut)
        for method_name in result["results"]:
            if "image" in result["results"][method_name]:
                del result["results"][method_name]["image"]

        return result

    except Exception as e:
        logger.error(f"İşlem hatası ({filename}): {e}")
        return None


class BatchProcessor:
    """
    Büyük veri setleri için paralel batch işleme motoru.

    Özellikler:
    - ProcessPoolExecutor ile çoklu CPU çekirdeği kullanımı
    - Batch bazlı bellek yönetimi
    - İlerleme çubuğu (tqdm)
    - Hata toleransı (tek görüntü hatası pipeline'ı durdurmaz)
    - 3-tuple desteği: (filename, hazy_image, clear_reference)
    """

    def __init__(
        self,
        methods: list[IDehazingMethod],
        data_loader: IDataLoader,
        metrics: list[IMetric] | None = None,
        batch_size: int = config.DEFAULT_BATCH_SIZE,
        max_workers: int = config.DEFAULT_WORKERS,
    ):
        self._methods = methods
        self._data_loader = data_loader
        self._metrics = metrics or get_no_reference_metrics()
        self._batch_size = batch_size
        self._max_workers = max_workers
        self._comparator = MethodComparator(methods, self._metrics)

    def run(self, parallel: bool = True) -> list[dict[str, Any]]:
        """
        Tüm veri setini işler.

        Args:
            parallel: True ise ProcessPoolExecutor kullanır.

        Returns:
            Tüm görüntülerin karşılaştırma sonuçları.
        """
        total = self._data_loader.get_total_count()
        logger.info(f"🚀 İşlem başlatılıyor: {total} görüntü")
        logger.info(f"   Batch boyutu: {self._batch_size} | Worker: {self._max_workers}")
        logger.info(f"   Yöntemler: {', '.join(m.get_name() for m in self._methods)}")

        start_time = time.perf_counter()
        all_results: list[dict[str, Any]] = []

        progress = tqdm(
            total=total,
            desc="İşleniyor",
            unit="img",
            bar_format="{l_bar}{bar:30}{r_bar}",
        )

        for batch in self._data_loader.load_batch(self._batch_size):
            if parallel and len(batch) > 1:
                batch_results = self._process_batch_parallel(batch)
            else:
                batch_results = self._process_batch_sequential(batch)

            all_results.extend(batch_results)
            progress.update(len(batch))

        progress.close()

        elapsed = time.perf_counter() - start_time
        successful = len([r for r in all_results if r is not None])
        logger.info(f"✅ İşlem tamamlandı: {successful}/{total} başarılı ({elapsed:.1f}s)")

        return [r for r in all_results if r is not None]

    def _process_batch_sequential(self, batch: list) -> list[dict]:
        """Batch'i sıralı olarak işler (debug ve küçük veri setleri için)."""
        results = []
        for item in batch:
            filename, image, reference = _unpack_batch_item(item)
            result = self._comparator.compare_single(image, filename, reference_image=reference)
            # Görüntüleri bellekten kaldır
            for method_name in result["results"]:
                if "image" in result["results"][method_name]:
                    del result["results"][method_name]["image"]
            results.append(result)
        return results

    def _process_batch_parallel(self, batch: list) -> list[dict]:
        """Batch'i paralel olarak işler."""
        method_configs = [
            {"name": type(m).__name__.replace("Method", "").lower()}
            for m in self._methods
        ]

        results = []
        with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {}
            for item in batch:
                filename, image, reference = _unpack_batch_item(item)
                future = executor.submit(
                    _process_single_image,
                    (filename, image, reference),
                    method_configs,
                    [m.get_name() for m in self._metrics],
                )
                futures[future] = filename

            for future in as_completed(futures):
                filename = futures[future]
                try:
                    result = future.result(timeout=120)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Worker hatası ({filename}): {e}")

        return results

    def run_with_export(
        self,
        output_dir: str,
        export_csv: bool = True,
        export_visuals: bool = True,
        parallel: bool = True,
    ) -> dict[str, Any]:
        """
        İşlem + dışa aktarma pipeline'ını çalıştırır.

        Returns:
            {"results": list, "summary_df": DataFrame, "export_paths": list}
        """
        from pipeline.result_exporter import ResultExporter

        # İşle
        all_results = self.run(parallel=parallel)

        # Özetle
        summary_df = self._comparator.aggregate_results(all_results)
        table_str = self._comparator.print_comparison_table(
            summary_df, total_images=len(all_results)
        )

        # Dışa aktar
        export_paths = []
        exporter = ResultExporter()

        if export_csv and not summary_df.empty:
            csv_path = exporter.export_metrics(
                {"summary": summary_df, "details": all_results},
                output_dir,
            )
            export_paths.append(csv_path)
            logger.info(f"📊 CSV kaydedildi: {csv_path}")

        # Görsel karşılaştırmalar için görüntüleri tekrar yükle
        if export_visuals:
            self._export_visual_comparisons(output_dir, exporter, export_paths)

        return {
            "results": all_results,
            "summary_df": summary_df,
            "export_paths": export_paths,
        }

    def _export_visual_comparisons(
        self, output_dir: str, exporter, export_paths: list
    ) -> None:
        """Her görüntü için görsel karşılaştırma grid'i oluşturur."""
        logger.info("🖼️  Görsel karşılaştırmalar oluşturuluyor...")

        # İlk batch'ten birkaç örnek görsel oluştur
        sample_count = 0
        max_samples = 10  # En fazla 10 örnek görsel

        for batch in self._data_loader.load_batch(self._batch_size):
            for item in batch:
                if sample_count >= max_samples:
                    return

                filename, image, reference = _unpack_batch_item(item)

                method_results = {}
                for method in self._methods:
                    try:
                        processed = method.process(image)
                        method_results[method.get_name()] = processed
                    except Exception as e:
                        logger.warning(f"Görsel oluşturma hatası ({method.get_name()}): {e}")

                if method_results:
                    path = exporter.export_comparison_grid(
                        image, method_results, filename, output_dir,
                        reference_image=reference,
                    )
                    export_paths.append(path)
                    sample_count += 1
