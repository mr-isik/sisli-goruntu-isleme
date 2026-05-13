"""
Sonuç Dışa Aktarma Modülü.

SOLID - Single Responsibility:
Yalnızca sonuçları dosyaya kaydetme ve görselleştirme ile ilgilenir.

SOLID - Interface Segregation:
IResultExporter arayüzünü implemente eder.
"""

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

from core.interfaces import IResultExporter
from utils.logger import get_logger

logger = get_logger(__name__)


class ResultExporter(IResultExporter):
    """
    Sonuçları CSV ve görsel formatlarda dışa aktarır.
    """

    def export_metrics(self, results: dict, output_dir: str) -> str:
        """
        Metrik sonuçlarını CSV dosyasına kaydeder.

        Args:
            results: {"summary": DataFrame, "details": list}
            output_dir: Çıktı dizini.

        Returns:
            Oluşturulan CSV dosyasının yolu.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Özet tablo
        if "summary" in results and not results["summary"].empty:
            summary_path = output_path / f"sonuc_ozet_{timestamp}.csv"
            results["summary"].to_csv(str(summary_path), encoding="utf-8-sig")
            logger.info(f"  📄 Özet CSV: {summary_path}")

        # Detaylı sonuçlar
        if "details" in results and results["details"]:
            details_path = output_path / f"sonuc_detay_{timestamp}.csv"
            self._export_detailed_csv(results["details"], str(details_path))
            return str(details_path)

        return str(output_path / f"sonuc_ozet_{timestamp}.csv")

    def _export_detailed_csv(self, details: list[dict], filepath: str) -> None:
        """Detaylı sonuçları satır bazlı CSV'ye yazar."""
        rows = []
        for result in details:
            filename = result["filename"]
            for method_name, method_result in result["results"].items():
                if "error" in method_result:
                    continue
                row = {
                    "Dosya": filename,
                    "Yöntem": method_name,
                }
                row.update(method_result.get("metrics", {}))
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            logger.info(f"  📄 Detay CSV: {filepath}")

    def export_comparison_grid(
        self,
        original: np.ndarray,
        results: dict[str, np.ndarray],
        filename: str,
        output_dir: str,
        reference_image: np.ndarray | None = None,
    ) -> str:
        """
        Orijinal ve işlenmiş görüntüleri yan yana grid olarak kaydeder.

        Düzen (referans varsa):
        ┌──────────┬──────────┬──────────┬──────────┬──────────┐
        │ Referans  │ Orijinal │   DCP    │  CLAHE   │ Retinex  │
        └──────────┴──────────┴──────────┴──────────┴──────────┘
        """
        output_path = Path(output_dir) / "karsilastirma"
        output_path.mkdir(parents=True, exist_ok=True)

        # Tüm görselleri aynı boyuta getir
        target_h, target_w = original.shape[:2]
        # Küçük boyutlarda göster
        max_cell_w = 400
        if target_w > max_cell_w:
            scale = max_cell_w / target_w
            target_w = max_cell_w
            target_h = int(target_h * scale)

        images_to_concat = []

        # Referans (temiz) görüntü (varsa)
        if reference_image is not None:
            resized_ref = cv2.resize(reference_image, (target_w, target_h))
            labeled_ref = self._add_label(resized_ref, "Temiz (Referans)")
            images_to_concat.append(labeled_ref)

        # Orijinal (sisli)
        resized_orig = cv2.resize(original, (target_w, target_h))
        labeled_orig = self._add_label(resized_orig, "Orijinal (Sisli)")
        images_to_concat.append(labeled_orig)

        # İşlenmiş görüntüler
        for method_name, processed in results.items():
            if processed is not None:
                resized = cv2.resize(processed, (target_w, target_h))
                # Kısa isim kullan
                short_name = method_name.split("(")[0].strip()
                labeled = self._add_label(resized, short_name)
                images_to_concat.append(labeled)

        # Yan yana birleştir
        grid = np.hstack(images_to_concat)

        # Kaydet
        stem = Path(filename).stem
        save_path = output_path / f"{stem}_karsilastirma.png"
        cv2.imwrite(str(save_path), grid)
        logger.debug(f"  🖼️  Görsel kaydedildi: {save_path}")

        return str(save_path)

    def _add_label(self, image: np.ndarray, label: str) -> np.ndarray:
        """Görüntünün üstüne etiket ekler."""
        h, w = image.shape[:2]
        label_height = 35

        # Etiket alanı oluştur
        label_area = np.zeros((label_height, w, 3), dtype=np.uint8)
        label_area[:] = (40, 40, 40)  # Koyu gri arka plan

        # Metni ortala
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = (label_height + text_size[1]) // 2

        cv2.putText(
            label_area, label, (text_x, text_y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        # Etiket + görüntü birleştir
        return np.vstack([label_area, image])
