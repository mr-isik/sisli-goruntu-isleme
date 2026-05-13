"""
Proje yapılandırma sabitleri.
Tüm varsayılan değerler burada merkezi olarak yönetilir.
"""

import os
from pathlib import Path

# ─── Dizin Yapılandırması ────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# ─── Veri Seti Yapılandırması ────────────────────────────────────────────────

INDOOR_DIR = INPUT_DIR / "indoor"
OUTDOOR_DIR = INPUT_DIR / "outdoor"
METADATA_INDOOR_CSV = INPUT_DIR / "metadata_indoor.csv"

# Her clear görüntü başına işlenecek maksimum hazy variant sayısı (None = tümü)
MAX_HAZY_PER_IMAGE: int | None = None

# ─── Desteklenen Görüntü Formatları ─────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# ─── Batch İşleme Yapılandırması ────────────────────────────────────────────

DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = max(1, os.cpu_count() - 1) if os.cpu_count() else 4
MAX_IMAGE_DIMENSION = 2048  # Büyük görselleri bu boyuta küçült

# ─── DCP (Dark Channel Prior) Parametreleri ──────────────────────────────────

DCP_PATCH_SIZE = 15       # Dark channel pencere boyutu
DCP_OMEGA = 0.95          # Sis kaldırma oranı (0-1)
DCP_T0 = 0.1              # Minimum transmission değeri
DCP_GUIDED_RADIUS = 60    # Guided filter yarıçapı
DCP_GUIDED_EPS = 1e-3     # Guided filter epsilon

# ─── CLAHE Parametreleri ─────────────────────────────────────────────────────

CLAHE_CLIP_LIMIT = 3.0        # Kontrast sınırlama eşiği
CLAHE_TILE_GRID_SIZE = (8, 8) # Histogram eşitleme grid boyutu
CLAHE_COLOR_SPACE = "LAB"     # Renk uzayı: "LAB" veya "HSV"

# ─── Retinex Parametreleri ───────────────────────────────────────────────────

RETINEX_SIGMA_LIST = [15, 80, 250]  # Multi-scale Gaussian sigma değerleri
RETINEX_GAIN = 128.0                 # Çıktı kazancı
RETINEX_OFFSET = 128.0               # Çıktı ofseti
RETINEX_COLOR_RESTORE_FACTOR = 6.0   # Renk geri yükleme faktörü
RETINEX_COLOR_RESTORE_GAIN = 2.0     # Renk geri yükleme kazancı

# ─── Loglama ─────────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
