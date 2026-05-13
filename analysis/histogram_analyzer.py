"""
Histogram Analiz Modülü.

Sisli ve netleştirilmiş görüntülerin histogram grafiklerini
yan yana Matplotlib figure olarak üretir.

SOLID - Single Responsibility:
Yalnızca histogram hesaplama ve görselleştirmeden sorumludur.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from core.interfaces import IDehazingMethod
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Renk teması ─────────────────────────────────────────────────────────────
PALETTE = {
    "background": "#0f1117",
    "surface":    "#1a1d27",
    "panel":      "#22263a",
    "border":     "#2e3350",
    "text":       "#e8eaf0",
    "muted":      "#7c83a0",
    "accent":     "#5c7cfa",
    "good":       "#40c057",
    "warn":       "#fab005",
}

CHANNEL_COLORS = {
    "B": "#4dabf7",   # Mavi
    "G": "#51cf66",   # Yeşil
    "R": "#ff6b6b",   # Kırmızı
}

PANEL_LABELS = {
    0: ("Temiz\n(Referans)", "#40c057"),
    1: ("Orijinal\n(Sisli)",  "#fab005"),
}


class HistogramAnalyzer:
    """
    Sisli / netleştirilmiş görüntü çiftleri için histogram analizi.

    Çıktı layout'u:
    ┌─────────────────────────────────────────────────────────────┐
    │  [Referans]  [Sisli]  [DCP patch=7]  [DCP patch=15]  ...   │
    │   görüntü   görüntü     görüntü         görüntü            │
    ├─────────────────────────────────────────────────────────────┤
    │  histogram  histogram   histogram       histogram     ...   │
    │  (R/G/B)    (R/G/B)     (R/G/B)         (R/G/B)            │
    │             PSNR: -     PSNR:19.4       PSNR:18.2          │
    │             SSIM: -     SSIM:0.88       SSIM:0.86          │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, output_dir: str | Path = "output/analiz"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def analyze_pair(
        self,
        hazy_image: np.ndarray,
        clear_image: np.ndarray | None,
        methods: list[IDehazingMethod],
        filename: str,
        metrics_data: dict[str, dict[str, float]] | None = None,
    ) -> str:
        """
        Tek bir clear-hazy çifti için histogram analiz figürü üretir.

        Args:
            hazy_image: Sisli BGR görüntü.
            clear_image: Temiz referans BGR görüntü (opsiyonel).
            methods: Uygulanacak dehazing yöntemleri.
            filename: Çıktı dosyası için temel ad.
            metrics_data: {method_name: {"PSNR (dB)": x, "SSIM": y}} (opsiyonel).

        Returns:
            Kaydedilen PNG dosyasının yolu.
        """
        # ─── Tüm görüntüleri topla ───────────────────────────────────────
        panels: list[dict[str, Any]] = []

        if clear_image is not None:
            panels.append({
                "image": clear_image,
                "label": "Temiz (Referans)",
                "color": PALETTE["good"],
                "metrics": {},
            })

        panels.append({
            "image": hazy_image,
            "label": "Orijinal (Sisli)",
            "color": PALETTE["warn"],
            "metrics": {},
        })

        for method in methods:
            try:
                processed = method.process(hazy_image)
                mname = method.get_name()
                m = (metrics_data or {}).get(mname, {})
                panels.append({
                    "image": processed,
                    "label": mname,
                    "color": PALETTE["accent"],
                    "metrics": m,
                })
            except Exception as e:
                logger.warning(f"  ⚠ {method.get_name()} işlem hatası: {e}")

        n = len(panels)
        fig = self._build_figure(panels, filename, n)

        stem = Path(filename).stem
        save_path = self._output_dir / f"histogram_{stem}.png"
        fig.savefig(
            str(save_path),
            dpi=150,
            bbox_inches="tight",
            facecolor=PALETTE["background"],
        )
        plt.close(fig)

        logger.info(f"  📊 Histogram kaydedildi: {save_path}")
        return str(save_path)

    # ─── Private ─────────────────────────────────────────────────────────

    def _build_figure(
        self, panels: list[dict], title: str, n: int
    ) -> plt.Figure:
        """Ana figure'ü oluşturur."""
        fig_w = max(5 * n, 14)
        fig = plt.figure(figsize=(fig_w, 9), facecolor=PALETTE["background"])

        # 2 satır: üst = görüntüler, alt = histogramlar
        gs = gridspec.GridSpec(
            2, n,
            figure=fig,
            hspace=0.06,
            wspace=0.04,
            height_ratios=[1.2, 1],
            top=0.88, bottom=0.06,
            left=0.03, right=0.97,
        )

        stem = Path(title).stem
        fig.suptitle(
            f"Histogram Analizi — {stem}",
            fontsize=14, fontweight="bold",
            color=PALETTE["text"], y=0.96,
            fontfamily="monospace",
        )

        for col, panel in enumerate(panels):
            ax_img = fig.add_subplot(gs[0, col])
            ax_hist = fig.add_subplot(gs[1, col])

            self._draw_image_panel(ax_img, panel, col == 0 and len(panels) > 2)
            self._draw_histogram_panel(ax_hist, panel)

        return fig

    def _draw_image_panel(
        self, ax: plt.Axes, panel: dict, is_reference: bool
    ) -> None:
        """Görüntü panelini çizer."""
        img_rgb = cv2.cvtColor(panel["image"], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_xticks([])
        ax.set_yticks([])

        label_color = panel["color"]
        for spine in ax.spines.values():
            spine.set_edgecolor(label_color)
            spine.set_linewidth(2.0 if is_reference else 1.2)

        ax.set_facecolor(PALETTE["background"])

        # Başlık
        short = panel["label"].split("(")[0].strip()
        ax.set_title(
            short,
            fontsize=9, fontweight="bold",
            color=label_color, pad=5,
            fontfamily="monospace",
        )

        # Metrikler (varsa)
        metrics = panel.get("metrics", {})
        psnr = metrics.get("PSNR (dB)")
        ssim = metrics.get("SSIM")
        if psnr is not None and ssim is not None:
            ax.set_xlabel(
                f"PSNR: {psnr:.2f} dB  |  SSIM: {ssim:.4f}",
                fontsize=7.5, color=PALETTE["muted"],
                labelpad=3,
            )

    def _draw_histogram_panel(self, ax: plt.Axes, panel: dict) -> None:
        """Histogram panelini R/G/B kanalları ayrı ayrı çizer."""
        image = panel["image"]
        ax.set_facecolor(PALETTE["panel"])

        bins = np.arange(0, 257, 2)  # 128 bin
        channel_map = {"B": 0, "G": 1, "R": 2}

        for ch_name, ch_idx in channel_map.items():
            channel = image[:, :, ch_idx]
            hist, _ = np.histogram(channel, bins=bins)
            hist = hist / hist.max()  # normalize

            x = (bins[:-1] + bins[1:]) / 2
            color = CHANNEL_COLORS[ch_name]
            ax.plot(x, hist, color=color, linewidth=0.9, alpha=0.9, label=ch_name)
            ax.fill_between(x, hist, alpha=0.12, color=color)

        # Stil
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Piksel Değeri", fontsize=7, color=PALETTE["muted"])
        ax.set_ylabel("Norm. Frekans", fontsize=7, color=PALETTE["muted"])

        for spine in ax.spines.values():
            spine.set_edgecolor(PALETTE["border"])
            spine.set_linewidth(0.8)

        ax.tick_params(colors=PALETTE["muted"], labelsize=6.5)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        ax.grid(True, color=PALETTE["border"], linewidth=0.4, alpha=0.5)
        ax.legend(
            fontsize=6.5, loc="upper left",
            framealpha=0.3,
            labelcolor=PALETTE["text"],
            facecolor=PALETTE["surface"],
            edgecolor=PALETTE["border"],
        )
