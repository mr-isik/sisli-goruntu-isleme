"""
Çekirdek arayüzler (Abstract Base Classes).

SOLID - Interface Segregation & Dependency Inversion:
Her arayüz tek bir sorumluluk alanına odaklanır.
Üst seviye modüller bu soyutlamalara bağımlıdır, somut implementasyonlara değil.
"""

from abc import ABC, abstractmethod
from typing import Any, Generator

import numpy as np


class IDehazingMethod(ABC):
    """
    Tüm sis giderme yöntemlerinin uyması gereken arayüz.

    SOLID - Strategy Pattern:
    Her yöntem bu arayüzü implemente ederek birbiriyle değiştirilebilir hale gelir.
    """

    @abstractmethod
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Sisli bir görüntüyü işleyerek iyileştirilmiş görüntü döndürür.

        Args:
            image: BGR formatında numpy dizisi (uint8, 0-255).

        Returns:
            İyileştirilmiş BGR görüntüsü (uint8, 0-255).
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Yöntemin insan tarafından okunabilir adını döndürür."""
        ...

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Yöntemin mevcut parametre değerlerini döndürür."""
        ...


class IMetric(ABC):
    """
    Kalite metriği arayüzü.

    SOLID - Single Responsibility:
    Her metrik sınıfı tek bir kalite ölçütünü hesaplar.
    """

    @abstractmethod
    def calculate(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        İki görüntü arasındaki kalite metriğini hesaplar.

        Args:
            original: Orijinal (sisli) BGR görüntü.
            processed: İşlenmiş (iyileştirilmiş) BGR görüntü.

        Returns:
            Hesaplanan metrik değeri.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Metriğin adını döndürür."""
        ...

    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """Daha yüksek değerin daha iyi olup olmadığını belirtir."""
        ...


class IDataLoader(ABC):
    """
    Veri yükleme arayüzü.

    SOLID - Open/Closed:
    Farklı veri kaynakları (dizin, zip, veritabanı, metadata CSV) için yeni
    implementasyonlar eklenebilir.

    Batch elemanları:
    - 2-tuple: (dosya_adı, görüntü)  → referanssız (eski format, geriye uyumlu)
    - 3-tuple: (dosya_adı, sisli_görüntü, temiz_referans | None) → referanslı
    """

    @abstractmethod
    def load_batch(
        self, batch_size: int
    ) -> Generator[list[tuple[str, np.ndarray] | tuple[str, np.ndarray, np.ndarray | None]], None, None]:
        """
        Görüntüleri batch'ler halinde lazy-load eder.

        Args:
            batch_size: Her batch'teki görüntü sayısı.

        Yields:
            (dosya_adı, görüntü) veya (dosya_adı, sisli_görüntü, temiz_referans)
            tuple'larından oluşan listeler.
        """
        ...

    @abstractmethod
    def get_total_count(self) -> int:
        """Toplam görüntü sayısını döndürür."""
        ...

    @abstractmethod
    def get_file_paths(self) -> list[str]:
        """Tüm dosya yollarını döndürür."""
        ...


class IResultExporter(ABC):
    """
    Sonuç dışa aktarma arayüzü.

    SOLID - Interface Segregation:
    Dışa aktarma mantığı diğer işlemlerden bağımsızdır.
    """

    @abstractmethod
    def export_metrics(self, results: dict, output_dir: str) -> str:
        """
        Metrik sonuçlarını dışa aktarır.

        Args:
            results: Metrik sonuçları sözlüğü.
            output_dir: Çıktı dizini.

        Returns:
            Oluşturulan dosyanın yolu.
        """
        ...

    @abstractmethod
    def export_comparison_grid(
        self,
        original: np.ndarray,
        results: dict[str, np.ndarray],
        filename: str,
        output_dir: str,
    ) -> str:
        """
        Orijinal ve işlenmiş görüntüleri yan yana karşılaştırma görseli olarak kaydeder.

        Args:
            original: Orijinal görüntü.
            results: {yöntem_adı: işlenmiş_görüntü} sözlüğü.
            filename: Dosya adı.
            output_dir: Çıktı dizini.

        Returns:
            Oluşturulan dosyanın yolu.
        """
        ...
