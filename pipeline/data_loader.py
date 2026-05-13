"""
Veri Yükleyici (Büyük Veri Seti Desteği).

SOLID - Single Responsibility:
Yalnızca dosya sistemi taraması ve lazy görüntü yükleme ile ilgilenir.

SOLID - Open/Closed:
Yeni veri kaynakları (zip, URL, veritabanı) IDataLoader arayüzünü
implemente ederek eklenebilir.
"""

from pathlib import Path
from typing import Generator

import cv2
import numpy as np

from core.interfaces import IDataLoader
import config
from utils.logger import get_logger

logger = get_logger(__name__)


class DirectoryDataLoader(IDataLoader):
    """
    Dizin tabanlı veri yükleyici.

    Büyük veri setleri için:
    - Lazy loading: Görüntüler sadece ihtiyaç duyulduğunda yüklenir
    - Batch processing: Belleği verimli kullanır
    - Büyük görselleri otomatik küçültür
    """

    def __init__(
        self,
        directory: str | Path,
        max_dimension: int = config.MAX_IMAGE_DIMENSION,
        extensions: set[str] | None = None,
    ):
        self._directory = Path(directory)
        self._max_dimension = max_dimension
        self._extensions = extensions or config.SUPPORTED_EXTENSIONS
        self._file_paths: list[Path] = []

        self._scan_directory()

    def _scan_directory(self) -> None:
        """Dizini tarayarak desteklenen dosyaları listeler."""
        if not self._directory.exists():
            raise FileNotFoundError(f"Dizin bulunamadı: {self._directory}")

        if self._directory.is_file():
            # Tek dosya verildi
            if self._directory.suffix.lower() in self._extensions:
                self._file_paths = [self._directory]
            else:
                raise ValueError(f"Desteklenmeyen dosya formatı: {self._directory.suffix}")
            return

        # Dizindeki tüm desteklenen dosyaları bul (recursive)
        for ext in self._extensions:
            self._file_paths.extend(sorted(self._directory.rglob(f"*{ext}")))

        # Büyük harfli uzantıları da dahil et
        for ext in self._extensions:
            upper_ext = ext.upper()
            self._file_paths.extend(sorted(self._directory.rglob(f"*{upper_ext}")))

        # Tekrarları kaldır ve sırala
        self._file_paths = sorted(set(self._file_paths))

        logger.info(f"📂 Dizin tarandı: {self._directory}")
        logger.info(f"   Bulunan görüntü sayısı: {len(self._file_paths)}")

    def load_batch(
        self, batch_size: int = config.DEFAULT_BATCH_SIZE
    ) -> Generator[list[tuple[str, np.ndarray]], None, None]:
        """
        Görüntüleri batch'ler halinde lazy-load eder.

        Bellek verimli: Sadece mevcut batch bellekte tutulur.
        """
        batch: list[tuple[str, np.ndarray]] = []

        for file_path in self._file_paths:
            image = self._load_single(file_path)
            if image is not None:
                batch.append((file_path.name, image))

            if len(batch) >= batch_size:
                yield batch
                batch = []

        # Son kalan görüntüler
        if batch:
            yield batch

    def _load_single(self, file_path: Path) -> np.ndarray | None:
        """Tek bir görüntüyü yükler, gerekirse küçültür."""
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                logger.warning(f"  ⚠ Okunamadı: {file_path.name}")
                return None

            # Büyük görselleri küçült (bellek yönetimi)
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
        return len(self._file_paths)

    def get_file_paths(self) -> list[str]:
        return [str(p) for p in self._file_paths]
