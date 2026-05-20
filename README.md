# 🌫️ Sisli Görüntü İyileştirme Sistemi

> **Haze Removal & Image Enhancement Pipeline**  
> DCP · CLAHE · Multi-Scale Retinex with Color Restoration

Büyük ölçekli sisli görüntü veri setleri üzerinde **paralel batch işleme**, **full-reference metrik hesaplama (PSNR/SSIM)**, **histogram analizi** ve **görsel raporlama** yapabilen, SOLID prensipleriyle tasarlanmış modüler bir Python pipeline'ı.

---

## 📋 İçindekiler

1. [Proje Yapısı](#-proje-yapısı)
2. [Teorik Altyapı](#-teorik-altyapı)
3. [Kurulum](#-kurulum)
4. [Veri Seti Yapısı](#-veri-seti-yapısı)
5. [Temel Kullanım](#-temel-kullanım)
6. [Analiz Aracı](#-analiz-aracı)
7. [Çıktılar](#-çıktılar)
8. [Parametre Referansı](#-parametre-referansı)
9. [Mimari](#-mimari)
10. [Test](#-test)

---

## 📁 Proje Yapısı

```
goruntu-isleme/
│
├── main.py                     # Ana giriş noktası (batch işleme)
├── analyze.py                  # Analiz & raporlama giriş noktası
├── config.py                   # Merkezi yapılandırma sabitleri
├── requirements.txt
│
├── core/
│   └── interfaces.py           # Soyut arayüzler (IDehazingMethod, IDataLoader...)
│
├── methods/                    # Dehazing algoritmaları
│   ├── dcp.py                  # Dark Channel Prior
│   ├── clahe.py                # CLAHE (LAB renk uzayı)
│   └── retinex.py              # Multi-Scale Retinex + Color Restoration
│
├── pipeline/
│   ├── data_loader.py          # DirectoryDataLoader (klasör bazlı)
│   ├── csv_data_loader.py      # MetadataDataLoader + OutdoorDataLoader
│   ├── processor.py            # BatchProcessor (paralel/seri)
│   └── result_exporter.py      # CSV & görsel grid dışa aktarımı
│
├── evaluation/
│   ├── comparator.py           # MethodComparator (no-ref + full-ref metrikler)
│   └── quality_metrics.py      # PSNR, SSIM, Entropy, Kontrast, vb.
│
├── analysis/
│   ├── histogram_analyzer.py   # R/G/B kanal histogram grafikleri
│   └── visual_report.py        # 300 DPI sunum görseli, bar chart, param raporu
│
├── utils/
│   └── logger.py               # Renkli yapılandırılmış loglama
│
├── tests/                      # Pytest birim testleri
│   ├── test_dcp.py
│   ├── test_clahe.py
│   ├── test_retinex.py
│   └── test_metrics.py
│
└── input/
    ├── metadata_indoor.csv     # Indoor veri seti eşleştirme tablosu
    ├── indoor/
    │   ├── clear/              # Temiz referans görüntüler
    │   └── hazy/               # Sisli görüntüler
    └── outdoor/
        ├── clear/              # Temiz referans görüntüler
        └── hazy/               # Sisli görüntüler
```

---

## 🔬 Teorik Altyapı

### 1. Dark Channel Prior (DCP)

He et al. (2009) tarafından önerilen yöntem, açık hava görüntülerindeki istatistiksel bir gözleme dayanır: **sis içermeyen bölgelerde en az bir renk kanalı çok düşük yoğunluk değerine sahiptir.**

**Atmospheric Scattering Modeli:**

```
I(x) = J(x) · t(x) + A · (1 - t(x))
```

| Sembol | Açıklama |
|--------|----------|
| `I(x)` | Gözlemlenen sisli görüntü |
| `J(x)` | Kurtarılacak sahne radyansı |
| `t(x)` | Transmission map (sis geçirgenliği) |
| `A`    | Global atmosferik ışık |

**İşlem adımları:**
1. Dark channel hesapla → `D(x) = min_{y∈Ω(x)} min_{c∈{r,g,b}} Iᶜ(y)`
2. Atmosferik ışığı tahmin et (en parlak %0.1 piksel)
3. Transmission map → `t(x) = 1 - ω · D(I/A)`
4. Guided filter ile kenar-koruyucu düzleştirme
5. Sahne radyansını kurtar → `J(x) = (I(x) - A) / max(t(x), t₀) + A`

**Temel parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `patch_size` | 15 | Dark channel pencere boyutu |
| `omega` | 0.95 | Sis kaldırma oranı (tam kaldırmama faktörü) |
| `t0` | 0.1 | Minimum transmission sınırı |
| `guided_radius` | 60 | Guided filter yarıçapı |

---

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)

Görüntüyü küçük bloklara bölerek her blok için bağımsız histogram eşitleme uygular. **LAB renk uzayında** yalnızca parlaklık (L) kanalına uygulanarak renk bozulması engellenir.

**İşlem adımları:**
1. BGR → LAB dönüşümü
2. L kanalına CLAHE uygula
3. LAB → BGR dönüşümü

**Temel parametreler:**

| Parametre | Varsayılan | Açıklama |
|-----------|-----------|----------|
| `clip_limit` | 3.0 | Kontrast amplifikasyon sınırı |
| `tile_grid_size` | (8, 8) | Blok boyutu |
| `color_space` | LAB | Renk uzayı (LAB veya HSV) |

---

### 3. Multi-Scale Retinex with Color Restoration (MSRCR)

Jobson et al. (1997) tarafından Edwin Land'ın Retinex teorisini genişletir. İnsan görme sisteminin aydınlatmayı yansıtıcılıktan ayırma mekanizmasını modeller.

**Matematiksel temel:**
```
log(R) = log(I) - log(L)        # SSR
MSR = (1/N) · Σ SSR(σᵢ)         # MSR
C(x,y) = β · log(α · Iᶜ / ΣIₖ) # Color Restoration
```

**Sigma değerlerinin etkisi:**
- Küçük σ (15): Lokal kontrast, kenar detayları
- Orta σ (80): Dengeli iyileştirme
- Büyük σ (250/300): Global ton denge

---

## ⚙️ Kurulum

### Gereksinimler
- Python 3.11+
- Linux / macOS / Windows

### Adım 1 — Depoyu klonlayın

```bash
git clone <repo-url>
cd goruntu-isleme
```

### Adım 2 — Sanal ortam oluşturun ve bağımlılıkları yükleyin

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# veya: venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Bağımlılıklar

| Paket | Sürüm | Amaç |
|-------|-------|------|
| `opencv-python` | ≥4.8 | Görüntü işleme çekirdeği |
| `numpy` | ≥1.24 | Sayısal hesaplama |
| `scikit-image` | ≥0.21 | SSIM hesaplama |
| `scipy` | ≥1.11 | Sinyal işleme |
| `pandas` | ≥2.0 | CSV okuma/yazma |
| `matplotlib` | ≥3.7 | Histogram ve raporlama görselleri |
| `tqdm` | ≥4.65 | İlerleme çubuğu |
| `tabulate` | ≥0.9 | Terminal tabloları |

---

## 📂 Veri Seti Yapısı

### Indoor Veri Seti

CSV tabanlı eşleştirme kullanılır. Her bir temiz görüntü için 1–10 arası sisli varyant bulunur.

```
input/
├── metadata_indoor.csv          ← Eşleştirme tablosu
└── indoor/
    ├── clear/
    │   ├── 1400.png
    │   ├── 1401.png
    │   └── ...
    └── hazy/
        ├── 1400_1.png           ← image_id=1400, variant 1
        ├── 1400_2.png           ← image_id=1400, variant 2
        │   ...
        ├── 1400_10.png
        ├── 1401_1.png
        └── ...
```

**`metadata_indoor.csv` formatı:**

```csv
image_id,clear_image_path,hazy_image_paths
1400,clear/1400.png,"['hazy/1400_1.png', 'hazy/1400_2.png', ..., 'hazy/1400_10.png']"
1401,clear/1401.png,"['hazy/1401_1.png', ..., 'hazy/1401_10.png']"
```

### Outdoor Veri Seti

Metadata CSV gerekmez — dosya adı bazlı otomatik eşleştirme:

```
input/outdoor/
├── clear/
│   ├── 0001.png
│   └── 0002.png
└── hazy/
    ├── 0001_0.8_0.2.jpg         ← ID=0001, beta=0.8, A=0.2
    ├── 0001_0.9_0.12.jpg
    └── 0002_0.85_0.08.jpg
```

> **Not:** `0001_0.8_0.2.jpg` dosyası `_` ile ayrılarak `0001` ID'si çıkarılır ve `clear/0001.png` ile eşleştirilir.

---

## 🚀 Temel Kullanım

> **Sanal ortamı aktif etmeyi unutmayın:**
> ```bash
> source venv/bin/activate
> ```

### Tüm veri setlerini işle

```bash
python main.py --dataset all --methods all
```

### Sadece Indoor veri seti

```bash
python main.py --dataset indoor --methods all
```

### Sadece belirli yöntemlerle

```bash
python main.py --dataset indoor --methods dcp clahe
```

### Parametre sweep modu (DCP patch=7 vs 15 + MSRCR sigma karşılaştırması)

```bash
python main.py --dataset indoor --param-sweep --hazy-per-image 3
```

### Her clear görüntü için sadece 1 hazy işle (hız testi için)

```bash
python main.py --dataset indoor --hazy-per-image 1 --no-parallel
```

### Manuel mod (CSV olmadan, klasör bazlı)

```bash
python main.py --input input/indoor/hazy --output output/ --methods all
```

### Tüm seçenekler

```
Argüman              Açıklama
─────────────────────────────────────────────────────────────
--dataset   -d       Veri seti: indoor | outdoor | all
--input     -i       Manuel girdi dizini/dosyası
--output    -o       Çıktı dizini (varsayılan: output/)
--methods   -m       Yöntemler: dcp clahe retinex | all
--batch-size -b      Batch boyutu (varsayılan: 16)
--workers   -w       Paralel worker sayısı (varsayılan: CPU-1)
--hazy-per-image     Her clear başına max hazy sayısı
--param-sweep        DCP ve MSRCR parametre karşılaştırması
--no-parallel        Sıralı işleme (debug için)
--no-visuals         Görsel grid üretme
--export-csv         CSV çıktısı (varsayılan: aktif)
```

---

## 📊 Analiz Aracı

`analyze.py`, histogram analizi ve sunum görselleri üretmek için bağımsız bir araçtır.

### Belirli bir görüntü çiftini analiz et

```bash
# Indoor ID=1400, variant 1, tüm yöntemler
python analyze.py --dataset indoor --image-id 1400 --hazy-variant 1

# Parametre sweep + sunum görseli
python analyze.py --dataset indoor --image-id 1400 --param-sweep --visual-report
```

### Rastgele N örnek analizi

```bash
python analyze.py --dataset indoor --sample-n 5
python analyze.py --dataset outdoor --sample-n 3 --visual-report
```

### Mevcut CSV'den bar chart üret

```bash
python analyze.py --from-csv output/indoor/sonuc_ozet_*.csv
```

### Tüm seçenekler

```
Argüman            Açıklama
──────────────────────────────────────────────────────
--dataset  -d      Veri seti: indoor | outdoor
--image-id         İşlenecek görüntü ID'si
--hazy-variant     Hazy varyant numarası (1-10)
--sample-n         Rastgele örnek sayısı (varsayılan: 3)
--methods  -m      Yöntemler (varsayılan: all)
--param-sweep      DCP ve MSRCR parametre karşılaştırması
--visual-report    Sunum kalitesinde görsel rapor üret
--from-csv         CSV dosyasından bar chart oluştur
--output   -o      Çıktı dizini
```

---

## 📤 Çıktılar

### Dizin Yapısı

```
output/
├── indoor/
│   ├── sonuc_detay_YYYYMMDD_HHMMSS.csv    ← Görüntü bazlı tüm metrikler
│   ├── sonuc_ozet_YYYYMMDD_HHMMSS.csv     ← Yöntem bazlı ortalama özet
│   ├── karsilastirma/
│   │   ├── 1400_1_karsilastirma.png       ← Temiz|Sisli|DCP|CLAHE|MSRCR yan yana
│   │   └── ...
│   ├── analiz/
│   │   ├── histogram_1400_1.png           ← Görüntü + R/G/B histogram
│   │   └── ...
│   └── rapor/
│       ├── metrik_barchart.png            ← 300 DPI yöntem × metrik bar chart
│       └── param_karsilastirma.png        ← Parametre sweep tablosu + görseller
└── outdoor/
    └── (aynı yapı)
```

### CSV Formatları

**`sonuc_detay.csv`** — Her görüntü × yöntem için:

| Dosya | Yöntem | Entropy | Kontrast | Renk Canlılığı | Kenar Yoğunluğu | Ort. Parlaklık | PSNR (dB) | SSIM | İşlem Süresi (ms) |
|-------|--------|---------|----------|----------------|-----------------|----------------|-----------|------|-------------------|
| 1400_1.png | DCP (patch=15) | 7.68 | 61.64 | 91.14 | 47.99 | 110.02 | 20.23 | 0.9032 | 38.98 |
| 1400_1.png | CLAHE (LAB) | 7.77 | 56.89 | 55.44 | 67.30 | 134.54 | 16.25 | 0.7392 | 81.24 |

> PSNR ve SSIM yalnızca referans görüntü mevcutsa hesaplanır (indoor & outdoor veri setleri).

### Terminal Çıktısı Örneği

```
════════════════════════════════════════════════
   SİSLİ GÖRÜNTÜ İYİLEŞTİRME SONUÇLARI
   Toplam Görüntü: 500
════════════════════════════════════════════════

╭───────────────────┬───────────┬──────────┬──────────────╮
│                   │ DCP(p=15) │  CLAHE   │ MSRCR(σ=...) │
├───────────────────┼───────────┼──────────┼──────────────┤
│ PSNR (dB)         │   16.96   │  15.83   │    14.29     │
│ SSIM              │   0.8571  │  0.7670  │    0.6171    │
│ Entropy           │   7.3268  │  7.5105  │    6.7752    │
│ Kontrast (Std)    │   61.12   │  57.23   │    92.26     │
│ Renk Canlılığı    │   45.54   │  19.38   │    35.53     │
│ Kenar Yoğunluğu   │   61.95   │  90.27   │   130.87     │
╰───────────────────┴───────────┴──────────┴──────────────╯

✅ En iyi PSNR (dB): DCP (16.9645)
✅ En iyi SSIM: DCP (0.8571)
✅ En iyi Entropy: CLAHE (7.5105)
```

---

## 🔧 Parametre Referansı

### DCP Parametre Analizi

| `patch_size` | PSNR ↑ | SSIM ↑ | İşlem Süresi |
|:---:|:---:|:---:|:---:|
| **7** | 17.49 dB | 0.8596 | ~48 ms |
| **15** | 20.23 dB | 0.9032 | ~46 ms |

> `patch_size=15` daha iyi referans benzerliği (PSNR/SSIM) sağlar.

### MSRCR Sigma Analizi

| Sigma Listesi | Kontrast ↑ | SSIM ↑ | İşlem Süresi |
|:---:|:---:|:---:|:---:|
| **[15, 80, 250]** | 93.82 | 0.6014 | ~995 ms |
| **[15, 300]** | 95.22 | 0.5905 | ~842 ms |

> `[15, 300]` daha yüksek kontrast, daha kısa işlem süresi; `[15, 80, 250]` daha dengeli.

### `config.py` ile Merkezi Yapılandırma

Tüm varsayılan değerler `config.py` dosyasından yönetilir:

```python
# DCP
DCP_PATCH_SIZE   = 15      # Pencere boyutu
DCP_OMEGA        = 0.95    # Sis kaldırma oranı
DCP_T0           = 0.1     # Min. transmission

# CLAHE
CLAHE_CLIP_LIMIT     = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Retinex
RETINEX_SIGMA_LIST = [15, 80, 250]

# İşleme
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS    = CPU_COUNT - 1
MAX_HAZY_PER_IMAGE = None    # None = tümünü işle
```

---

## 🏗️ Mimari

Proje **SOLID prensiplerine** dayalı katmanlı bir mimariye sahiptir:

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py / analyze.py                 │  ← Giriş Noktaları
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                     BatchProcessor                           │  ← Orchestration
│         (ProcessPoolExecutor | Sequential)                   │
└────────┬──────────────────────────────┬─────────────────────┘
         │                              │
┌────────▼────────┐            ┌────────▼────────────────────┐
│   IDataLoader   │            │      MethodComparator        │
│                 │            │                              │
│ MetadataLoader  │            │  IDehazingMethod × N         │
│ OutdoorLoader   │            │  IMetric × M                 │
│ DirectoryLoader │            │  → No-ref + Full-ref         │
└─────────────────┘            └──────────────────────────────┘
         │                              │
┌────────▼────────┐            ┌────────▼────────────────────┐
│  (filename,     │            │     ResultExporter           │
│   hazy_image,   │            │   HistogramAnalyzer          │
│   clear_image)  │            │   VisualReport               │
└─────────────────┘            └──────────────────────────────┘
```

### Temel Arayüzler (`core/interfaces.py`)

```python
class IDehazingMethod(ABC):
    def process(self, image: np.ndarray) -> np.ndarray: ...
    def get_name(self) -> str: ...
    def get_params(self) -> dict: ...

class IDataLoader(ABC):
    def load_batch(self, batch_size: int) -> Generator[
        list[tuple[str, np.ndarray, np.ndarray | None]], ...]: ...
    def get_total_count(self) -> int: ...

class IMetric(ABC):
    def compute(self, original, processed, reference=None) -> float: ...
    def get_name(self) -> str: ...
    def higher_is_better(self) -> bool: ...
```

### Metrik Sistemi

| Metrik | Tür | Açıklama |
|--------|-----|----------|
| Entropy | No-Reference | Bilgi içeriği zenginliği |
| Kontrast (Std) | No-Reference | Standart sapma bazlı kontrast |
| Renk Canlılığı | No-Reference | Renk doygunluk ortalaması |
| Kenar Yoğunluğu | No-Reference | Sobel gradyan yoğunluğu |
| Ort. Parlaklık | No-Reference | Ortalama piksel değeri |
| **PSNR** | **Full-Reference** | Referansa göre sinyal/gürültü oranı |
| **SSIM** | **Full-Reference** | Referansa göre yapısal benzerlik |

> PSNR ve SSIM yalnızca temiz referans görüntü mevcut olduğunda hesaplanır.

---

## 🧪 Test

```bash
# Sanal ortamı aktif et
source venv/bin/activate

# Tüm testleri çalıştır
python -m pytest tests/ -v

# Belirli bir test modülü
python -m pytest tests/test_dcp.py -v
python -m pytest tests/test_metrics.py -v

# Coverage ile
python -m pytest tests/ --cov=. --cov-report=term-missing
```

### Test Kapsamı

| Test Dosyası | Kapsam |
|---|---|
| `test_dcp.py` | DCP işlem akışı, guided filter, edge-case'ler |
| `test_clahe.py` | CLAHE LAB/HSV dönüşümleri, parametre doğrulama |
| `test_retinex.py` | SSR/MSR/MSRCR hesaplama, color restoration |
| `test_metrics.py` | PSNR, SSIM, no-ref metrik hesaplama |

---

## 📌 Hızlı Başlangıç Özeti

```bash
# 1. Kurulum
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Tüm veri seti, tüm yöntemler
python main.py --dataset all --methods all

# 3. Parametre sweep (hoca sunumu için)
python main.py --dataset indoor --param-sweep --hazy-per-image 3

# 4. Belirli görüntü için histogram analizi
python analyze.py --dataset indoor --image-id 1400 --param-sweep --visual-report

# 5. Mevcut CSV'den bar chart
python analyze.py --from-csv output/indoor/sonuc_ozet_*.csv
```

---

## 📚 Referanslar

1. **DCP:** He, K., Sun, J., & Tang, X. (2009). *Single Image Haze Removal Using Dark Channel Prior.* IEEE CVPR.
2. **CLAHE:** Zuiderveld, K. (1994). *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV.
3. **Retinex:** Jobson, D.J., Rahman, Z., & Woodell, G.A. (1997). *A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes.* IEEE Transactions on Image Processing.
4. **SSIM:** Wang, Z. et al. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE Transactions on Image Processing.
