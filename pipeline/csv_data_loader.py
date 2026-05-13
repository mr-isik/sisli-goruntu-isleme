"""
Metadata CSV Tabanlı Veri Yükleyici.

SOLID - Single Responsibility:
Metadata CSV dosyasından clear-hazy görüntü çiftlerini yükler.
Her satırdaki clear referans ve hazy variant'ları eşleştirir.

SOLID - Open/Closed:
IDataLoader arayüzünü implemente eder; yeni veri kaynakları
(zip, URL) eklenerek genişletilebilir.
"""

import ast
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pandas as pd

from core.interfaces import IDataLoader
import config
from utils.logger import get_logger

logger = get_logger(__name__)


class MetadataDataLoader(IDataLoader):
    """
    Metadata CSV tabanlı veri yükleyici.

    CSV formatı:
        image_id, clear_image_path, hazy_image_paths
        1400,     clear/1400.png,   ['hazy/1400_1.png', 'hazy/1400_2.png', ...]

    Her clear görüntü için birden fazla hazy variant yüklenir.
    PSNR/SSIM gibi full-reference metrikler için clear-hazy çiftleri sağlar.
    """

    def __init__(
        self,
        csv_path: str | Path,
        dataset_dir: str | Path,
        max_dimension: int = config.MAX_IMAGE_DIMENSION,
        max_hazy_per_image: int | None = config.MAX_HAZY_PER_IMAGE,
    ):
        """
        Args:
            csv_path: Metadata CSV dosya yolu.
            dataset_dir: Görüntü dosyalarının kök dizini (ör: input/indoor/).
            max_dimension: Büyük görselleri bu boyuta küçült.
            max_hazy_per_image: Her clear görüntü başına maks hazy sayısı (None = tümü).
        """
        self._csv_path = Path(csv_path)
        self._dataset_dir = Path(dataset_dir)
        self._max_dimension = max_dimension
        self._max_hazy_per_image = max_hazy_per_image

        self._pairs: list[dict] = []  # {clear_path, hazy_path, image_id}
        self._parse_csv()

    def _parse_csv(self) -> None:
        """Metadata CSV dosyasını parse ederek clear-hazy çiftlerini oluşturur."""
        if not self._csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV bulunamadı: {self._csv_path}")

        df = pd.read_csv(self._csv_path)

        required_cols = {"image_id", "clear_image_path", "hazy_image_paths"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"CSV'de eksik sütunlar: {missing}")

        for _, row in df.iterrows():
            image_id = row["image_id"]
            clear_path = self._dataset_dir / row["clear_image_path"]

            # hazy_image_paths sütunu Python list string formatında
            try:
                hazy_paths = ast.literal_eval(row["hazy_image_paths"])
            except (ValueError, SyntaxError) as e:
                logger.warning(f"  ⚠ Hazy yolları parse edilemedi (ID: {image_id}): {e}")
                continue

            if not isinstance(hazy_paths, list):
                hazy_paths = [hazy_paths]

            # Maksimum hazy sınırlaması uygula
            if self._max_hazy_per_image is not None:
                hazy_paths = hazy_paths[:self._max_hazy_per_image]

            for hazy_rel_path in hazy_paths:
                hazy_path = self._dataset_dir / hazy_rel_path
                self._pairs.append({
                    "image_id": image_id,
                    "clear_path": clear_path,
                    "hazy_path": hazy_path,
                })

        logger.info(f"📂 Metadata CSV okundu: {self._csv_path}")
        logger.info(f"   Toplam clear-hazy çifti: {len(self._pairs)}")

    def load_batch(
        self, batch_size: int = config.DEFAULT_BATCH_SIZE
    ) -> Generator[list[tuple[str, np.ndarray, np.ndarray | None]], None, None]:
        """
        Clear-hazy çiftlerini batch'ler halinde lazy-load eder.

        Yields:
            (hazy_dosya_adı, sisli_görüntü, temiz_referans) tuple listesi.
        """
        batch: list[tuple[str, np.ndarray, np.ndarray | None]] = []

        # Clear görüntüleri cache'le (aynı clear birden fazla hazy ile eşleşir)
        clear_cache: dict[str, np.ndarray | None] = {}

        for pair in self._pairs:
            clear_path = pair["clear_path"]
            hazy_path = pair["hazy_path"]

            # Hazy görüntüyü yükle
            hazy_image = self._load_single(hazy_path)
            if hazy_image is None:
                continue

            # Clear görüntüyü cache'den yükle
            clear_key = str(clear_path)
            if clear_key not in clear_cache:
                clear_cache[clear_key] = self._load_single(clear_path)
            clear_image = clear_cache[clear_key]

            batch.append((hazy_path.name, hazy_image, clear_image))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Son kalan
        if batch:
            yield batch

    def _load_single(self, file_path: Path) -> np.ndarray | None:
        """Tek bir görüntüyü yükler, gerekirse küçültür."""
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                logger.warning(f"  ⚠ Okunamadı: {file_path}")
                return None

            # Büyük görselleri küçült
            h, w = image.shape[:2]
            if max(h, w) > self._max_dimension:
                scale = self._max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                logger.debug(f"  📐 Küçültüldü: {w}x{h} → {new_w}x{new_h} ({file_path.name})")

            return image

        except Exception as e:
            logger.error(f"  ✗ Yükleme hatası ({file_path.name}): {e}")
            return None

    def get_total_count(self) -> int:
        return len(self._pairs)

    def get_file_paths(self) -> list[str]:
        return [str(p["hazy_path"]) for p in self._pairs]


class OutdoorDataLoader(IDataLoader):
    """
    Outdoor veri seti yükleyici.

    Metadata CSV olmayan veri setleri için dosya adı bazlı eşleştirme yapar.
    Hazy dosya adı formatı: {id}_{param1}_{param2}.jpg → Clear: {id}.png

    Örnek: 0001_0.8_0.2.jpg → clear/0001.png
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        max_dimension: int = config.MAX_IMAGE_DIMENSION,
        extensions: set[str] | None = None,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._max_dimension = max_dimension
        self._extensions = extensions or config.SUPPORTED_EXTENSIONS
        self._pairs: list[dict] = []

        self._scan_and_match()

    def _scan_and_match(self) -> None:
        """Hazy ve clear dizinlerini tarayarak dosya adı bazlı eşleştirme yapar."""
        clear_dir = self._dataset_dir / "clear"
        hazy_dir = self._dataset_dir / "hazy"

        if not clear_dir.exists():
            raise FileNotFoundError(f"Clear dizini bulunamadı: {clear_dir}")
        if not hazy_dir.exists():
            raise FileNotFoundError(f"Hazy dizini bulunamadı: {hazy_dir}")

        # Clear dosyalarını indexle (stem → path)
        clear_index: dict[str, Path] = {}
        for f in sorted(clear_dir.iterdir()):
            if f.suffix.lower() in self._extensions:
                clear_index[f.stem] = f

        # Hazy dosyalarını tara ve eşleştir
        for f in sorted(hazy_dir.iterdir()):
            if f.suffix.lower() not in self._extensions:
                continue

            # Dosya adından image ID'yi çıkar: 0001_0.8_0.2.jpg → 0001
            image_id = f.stem.split("_")[0]

            if image_id in clear_index:
                self._pairs.append({
                    "image_id": image_id,
                    "clear_path": clear_index[image_id],
                    "hazy_path": f,
                })
            else:
                logger.warning(f"  ⚠ Eşleşme bulunamadı: {f.name} → clear/{image_id}.*")

        logger.info(f"📂 Outdoor dizin tarandı: {self._dataset_dir}")
        logger.info(f"   Toplam clear-hazy çifti: {len(self._pairs)}")

    def load_batch(
        self, batch_size: int = config.DEFAULT_BATCH_SIZE
    ) -> Generator[list[tuple[str, np.ndarray, np.ndarray | None]], None, None]:
        """Clear-hazy çiftlerini batch'ler halinde lazy-load eder."""
        batch: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        clear_cache: dict[str, np.ndarray | None] = {}

        for pair in self._pairs:
            clear_path = pair["clear_path"]
            hazy_path = pair["hazy_path"]

            hazy_image = self._load_single(hazy_path)
            if hazy_image is None:
                continue

            clear_key = str(clear_path)
            if clear_key not in clear_cache:
                clear_cache[clear_key] = self._load_single(clear_path)
            clear_image = clear_cache[clear_key]

            # Clear ve hazy boyutlarını eşitle (PSNR/SSIM için gerekli)
            if clear_image is not None:
                h, w = hazy_image.shape[:2]
                ch, cw = clear_image.shape[:2]
                if (h, w) != (ch, cw):
                    clear_image = cv2.resize(clear_image, (w, h), interpolation=cv2.INTER_AREA)

            batch.append((hazy_path.name, hazy_image, clear_image))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    def _load_single(self, file_path: Path) -> np.ndarray | None:
        """Tek bir görüntüyü yükler, gerekirse küçültür."""
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                logger.warning(f"  ⚠ Okunamadı: {file_path}")
                return None

            h, w = image.shape[:2]
            if max(h, w) > self._max_dimension:
                scale = self._max_dimension / max(h, w)
                new_w, new_h = int(w * scale), int(h * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

            return image

        except Exception as e:
            logger.error(f"  ✗ Yükleme hatası ({file_path.name}): {e}")
            return None

    def get_total_count(self) -> int:
        return len(self._pairs)

    def get_file_paths(self) -> list[str]:
        return [str(p["hazy_path"]) for p in self._pairs]
