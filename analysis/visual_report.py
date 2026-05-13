"""
Sunum Kalitesinde Görsel Rapor Modülü.

Hoca sunumu için 300 DPI Matplotlib figure üretir:
  - Parametre karşılaştırma tablosu + görsel panel
  - Tüm yöntemlerin metrik bar chart'ları
  - Özet karşılaştırma grid'i

SOLID - Single Responsibility:
Yalnızca sunum görseli üretiminden sorumludur.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Tema ────────────────────────────────────────────────────────────────────
BG        = "#0d1117"
SURFACE   = "#161b22"
PANEL     = "#1c2128"
BORDER    = "#30363d"
TEXT      = "#e6edf3"
MUTED     = "#8b949e"
BLUE      = "#58a6ff"
GREEN     = "#3fb950"
YELLOW    = "#d29922"
RED       = "#f85149"
PURPLE    = "#bc8cff"
ORANGE    = "#ffa657"

# Yönteme göre sabit renkler
METHOD_COLORS = [
    "#58a6ff",   # DCP patch=7  (mavi)
    "#3fb950",   # DCP patch=15 (yeşil)
    "#d29922",   # CLAHE        (sarı)
    "#bc8cff",   # MSRCR σ=küçük (mor)
    "#ffa657",   # MSRCR σ=büyük (turuncu)
    "#f85149",   # 6. yöntem (kırmızı)
]

METRIC_HIGHER_BETTER = {
    "PSNR (dB)":        True,
    "SSIM":             True,
    "Entropy":          True,
    "Kontrast (Std)":   True,
    "Renk Canlılığı":   True,
    "Kenar Yoğunluğu":  True,
    "Ort. Parlaklık":   False,
    "İşlem Süresi (ms)": False,
}


class VisualReport:
    """
    Hoca sunumu için yüksek kaliteli görsel rapor üretici.

    Üç farklı rapor türü:
      1. comparison_grid()  → Tek görüntü üzerinde tüm yöntemler (sunum için)
      2. metric_barchart()  → Yöntem × metrik bar chart
      3. param_sweep_report() → Parametre karşılaştırma özeti
    """

    def __init__(self, output_dir: str | Path = "output/rapor"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Public API ──────────────────────────────────────────────────────

    def comparison_grid(
        self,
        hazy_image: np.ndarray,
        processed_images: dict[str, np.ndarray],
        filename: str,
        clear_image: np.ndarray | None = None,
        metrics_data: dict[str, dict[str, float]] | None = None,
        title: str = "",
    ) -> str:
        """
        Orijinal + tüm işlenmiş görüntüleri tek satırda yan yana sunum görseli.

        Args:
            hazy_image: Sisli girdi.
            processed_images: {yöntem_adı: işlenmiş_görüntü}
            filename: Kayıt dosya adı.
            clear_image: Referans temiz görüntü (opsiyonel).
            metrics_data: {yöntem_adı: {metrik: değer}}
            title: Figure başlığı.

        Returns:
            Kaydedilen PNG yolu.
        """
        panels = []
        if clear_image is not None:
            panels.append(("Temiz\n(Referans)", clear_image, GREEN, {}))
        panels.append(("Orijinal\n(Sisli)", hazy_image, YELLOW, {}))

        for i, (mname, img) in enumerate(processed_images.items()):
            color = METHOD_COLORS[i % len(METHOD_COLORS)]
            m = (metrics_data or {}).get(mname, {})
            panels.append((mname, img, color, m))

        n = len(panels)
        cell_w = min(380, 1800 // n)
        cell_h = int(cell_w * 0.70)

        fig_w = (cell_w * n) / 100
        fig_h = (cell_h + 90) / 100

        fig, axes = plt.subplots(
            1, n,
            figsize=(fig_w, fig_h),
            facecolor=BG,
        )
        if n == 1:
            axes = [axes]

        fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.12, wspace=0.04)

        main_title = title or Path(filename).stem
        fig.suptitle(
            main_title,
            fontsize=13, fontweight="bold",
            color=TEXT, y=0.97,
        )

        for ax, (label, img, color, metrics) in zip(axes, panels):
            self._draw_image_cell(ax, img, label, color, metrics, cell_w, cell_h)

        save_path = self._output_dir / f"{Path(filename).stem}_grid.png"
        fig.savefig(str(save_path), dpi=200, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info(f"  🖼️  Grid görseli: {save_path}")
        return str(save_path)

    def metric_barchart(
        self,
        summary_df: pd.DataFrame,
        filename: str = "metrik_barchart",
        title: str = "Yöntem Karşılaştırması — Metrik Özeti",
    ) -> str:
        """
        Her metrik için yöntemleri karşılaştıran bar chart.

        Args:
            summary_df: aggregate_results() çıktısı (index=metrikler, sütunlar=yöntemler).
            filename: Kayıt adı.
            title: Figure başlığı.

        Returns:
            Kaydedilen PNG yolu.
        """
        if summary_df.empty:
            logger.warning("  ⚠ Boş DataFrame, barchart atlandı.")
            return ""

        metrics = list(summary_df.index)
        methods = list(summary_df.columns)
        n_metrics = len(metrics)
        n_methods = len(methods)

        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 5.5, rows * 4),
            facecolor=BG,
        )
        fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.90, bottom=0.08)
        axes_flat = np.array(axes).ravel() if n_metrics > 1 else [axes]

        fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT, y=0.97)

        for ax, metric in zip(axes_flat, metrics):
            vals = [summary_df.loc[metric, m] for m in methods]
            colors = [METHOD_COLORS[i % len(METHOD_COLORS)] for i in range(n_methods)]

            higher = METRIC_HIGHER_BETTER.get(metric, True)
            best_idx = int(np.argmax(vals) if higher else np.argmin(vals))

            bars = ax.barh(
                range(n_methods), vals,
                color=colors, alpha=0.85,
                edgecolor=BORDER, linewidth=0.6,
                height=0.6,
            )

            # En iyi sonucu vurgula
            bars[best_idx].set_alpha(1.0)
            bars[best_idx].set_edgecolor(TEXT)
            bars[best_idx].set_linewidth(1.8)

            # Değer etiketleri
            for i, (bar, val) in enumerate(zip(bars, vals)):
                star = " ★" if i == best_idx else ""
                ax.text(
                    val + abs(max(vals) - min(vals)) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}{star}",
                    va="center", ha="left",
                    fontsize=7.5, color=TEXT if i == best_idx else MUTED,
                    fontweight="bold" if i == best_idx else "normal",
                )

            short_methods = [m.split("(")[0].strip() for m in methods]
            ax.set_yticks(range(n_methods))
            ax.set_yticklabels(short_methods, fontsize=8, color=TEXT)
            ax.set_title(metric, fontsize=9, fontweight="bold", color=TEXT, pad=6)
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            ax.set_xlabel("", fontsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
                spine.set_linewidth(0.7)
            ax.grid(True, axis="x", color=BORDER, linewidth=0.4, alpha=0.6)
            ax.set_xlim(0, max(vals) * 1.22 if max(vals) > 0 else 1)

        # Kullanılmayan eksenleri gizle
        for ax in axes_flat[n_metrics:]:
            ax.set_visible(False)

        save_path = self._output_dir / f"{filename}.png"
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info(f"  📊 Bar chart: {save_path}")
        return str(save_path)

    def param_sweep_report(
        self,
        summary_df: pd.DataFrame,
        sample_images: list[dict],
        filename: str = "param_karsilastirma",
        title: str = "Parametre Karşılaştırması",
    ) -> str:
        """
        Parametre sweep sonuçlarını tablo + örnek görüntülerle birleştirir.

        Args:
            summary_df: aggregate_results() çıktısı.
            sample_images: [{"label": str, "image": np.ndarray}, ...]
            filename: Kayıt adı.
            title: Figure başlığı.

        Returns:
            Kaydedilen PNG yolu.
        """
        if summary_df.empty:
            logger.warning("  ⚠ Boş DataFrame, param raporu atlandı.")
            return ""

        methods = list(summary_df.columns)
        metrics = list(summary_df.index)
        n_methods = len(methods)
        n_samples = min(len(sample_images), 3)  # En fazla 3 örnek görüntü

        # Layout: üst = örnek görüntüler, alt = metrik tablosu
        fig = plt.figure(figsize=(max(14, n_methods * 2.8), 10), facecolor=BG)
        gs = gridspec.GridSpec(
            2, 1,
            figure=fig,
            height_ratios=[1.4, 1],
            hspace=0.08,
            top=0.91, bottom=0.06,
            left=0.03, right=0.97,
        )

        fig.suptitle(title, fontsize=15, fontweight="bold", color=TEXT, y=0.97)

        # ─── Üst: Örnek görüntüler ────────────────────────────────────────
        if n_samples > 0:
            gs_top = gridspec.GridSpecFromSubplotSpec(
                1, n_samples,
                subplot_spec=gs[0],
                wspace=0.04,
            )
            for i in range(n_samples):
                ax = fig.add_subplot(gs_top[i])
                item = sample_images[i]
                img_rgb = cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
                ax.set_xticks([]); ax.set_yticks([])
                color = METHOD_COLORS[i % len(METHOD_COLORS)]
                for spine in ax.spines.values():
                    spine.set_edgecolor(color); spine.set_linewidth(1.8)
                ax.set_title(item["label"], fontsize=8.5, color=color,
                             fontweight="bold", pad=4, fontfamily="monospace")
                m = item.get("metrics", {})
                psnr = m.get("PSNR (dB)"); ssim = m.get("SSIM")
                if psnr and ssim:
                    ax.set_xlabel(f"PSNR {psnr:.2f} dB · SSIM {ssim:.4f}",
                                  fontsize=7, color=MUTED, labelpad=3)
                ax.set_facecolor(BG)

        # ─── Alt: Metrik tablosu ─────────────────────────────────────────
        ax_table = fig.add_subplot(gs[1])
        ax_table.set_facecolor(PANEL)
        ax_table.set_xticks([]); ax_table.set_yticks([])
        for spine in ax_table.spines.values():
            spine.set_visible(False)

        short_methods = [m.replace(" (", "\n(") for m in methods]
        cell_text = []
        cell_colors = []

        for metric in metrics:
            row_vals = [summary_df.loc[metric, m] for m in methods]
            higher = METRIC_HIGHER_BETTER.get(metric, True)
            best_idx = int(np.argmax(row_vals) if higher else np.argmin(row_vals))

            row_text = []
            row_col = []
            for i, v in enumerate(row_vals):
                row_text.append(f"{v:.4f}")
                if i == best_idx:
                    row_col.append("#1a3a1a")   # koyu yeşil vurgu
                else:
                    row_col.append(PANEL)
            cell_text.append(row_text)
            cell_colors.append(row_col)

        tbl = ax_table.table(
            cellText=cell_text,
            rowLabels=metrics,
            colLabels=short_methods,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.7)

        # Hücre stilleri
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor(BORDER)
            cell.set_text_props(color=TEXT)
            if row == 0:
                cell.set_facecolor(SURFACE)
                cell.set_text_props(fontweight="bold", color=BLUE)
            if col == -1:
                cell.set_facecolor(SURFACE)
                cell.set_text_props(color=MUTED, fontsize=8)

        save_path = self._output_dir / f"{filename}.png"
        fig.savefig(str(save_path), dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        logger.info(f"  📑 Param raporu: {save_path}")
        return str(save_path)

    # ─── Private ─────────────────────────────────────────────────────────

    def _draw_image_cell(
        self,
        ax: plt.Axes,
        image: np.ndarray,
        label: str,
        color: str,
        metrics: dict,
        cell_w: int,
        cell_h: int,
    ) -> None:
        """Tek bir görüntü hücresini çizer."""
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Yeniden boyutlandır
        h, w = img_rgb.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        nw, nh = int(w * scale), int(h * scale)
        img_rgb = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

        ax.imshow(img_rgb)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor(BG)

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.8)

        short = label.split("\n")[0].strip()
        ax.set_title(short, fontsize=8.5, fontweight="bold", color=color, pad=5)

        psnr = metrics.get("PSNR (dB)")
        ssim  = metrics.get("SSIM")
        if psnr is not None and ssim is not None:
            ax.set_xlabel(
                f"PSNR: {psnr:.2f} dB   SSIM: {ssim:.4f}",
                fontsize=7.5, color=MUTED, labelpad=4,
            )
