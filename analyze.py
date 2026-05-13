#!/usr/bin/env python3
"""
Görüntü Analiz ve Görselleştirme Aracı.

Histogram analizi, sunum görseli ve parametre karşılaştırma raporları üretir.
Mevcut pipeline çıktılarını (CSV) veya canlı işleme ile çalışabilir.

Kullanım:
    # Belirli bir görüntü çifti analizi (histogram + grid)
    python analyze.py --dataset indoor --image-id 1400 --hazy-variant 1

    # Parametre sweep + sunum görseli
    python analyze.py --dataset indoor --param-sweep --image-id 1400

    # Tüm yöntemlerle N rastgele örnek
    python analyze.py --dataset indoor --sample-n 5 --visual-report

    # Mevcut CSV'den metrik bar chart
    python analyze.py --from-csv output/indoor/sonuc_ozet_20260513.csv
"""

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd

import config
from methods import create_method, METHOD_REGISTRY
from methods.dcp import DCPMethod
from methods.retinex import RetinexMethod
from methods.clahe import CLAHEMethod
from pipeline.csv_data_loader import MetadataDataLoader, OutdoorDataLoader
from analysis.histogram_analyzer import HistogramAnalyzer
from analysis.visual_report import VisualReport
from evaluation.comparator import MethodComparator
from evaluation.quality_metrics import get_no_reference_metrics
from utils.logger import get_logger

logger = get_logger("analyze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Histogram analizi ve görsel raporlama aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  %(prog)s --dataset indoor --image-id 1400 --hazy-variant 1
  %(prog)s --dataset indoor --param-sweep --image-id 1400 --hazy-variant 3
  %(prog)s --dataset indoor --sample-n 5 --visual-report
  %(prog)s --from-csv output/indoor/sonuc_ozet_20260513_204918.csv
        """,
    )

    parser.add_argument("--dataset", "-d", default="indoor",
                        choices=["indoor", "outdoor"],
                        help="Veri seti (varsayılan: indoor)")
    parser.add_argument("--image-id", type=str, default=None,
                        help="İşlenecek görüntü ID'si (indoor: 1400-1449)")
    parser.add_argument("--hazy-variant", type=int, default=1,
                        help="Hazy variant numarası (1-10, varsayılan: 1)")
    parser.add_argument("--sample-n", type=int, default=3,
                        help="Rastgele örnek sayısı (varsayılan: 3)")
    parser.add_argument("--methods", nargs="+",
                        default=["all"],
                        choices=list(METHOD_REGISTRY.keys()) + ["all"],
                        help="Kullanılacak yöntemler")
    parser.add_argument("--param-sweep", action="store_true",
                        help="DCP ve MSRCR parametre karşılaştırması")
    parser.add_argument("--visual-report", action="store_true",
                        help="Sunum kalitesinde görsel rapor üret")
    parser.add_argument("--from-csv", type=str, default=None,
                        help="Mevcut özet CSV'den bar chart üret")
    parser.add_argument("--output", "-o", default=None,
                        help="Çıktı dizini (varsayılan: output/<dataset>/analiz)")

    return parser.parse_args()


def _build_methods(args) -> list:
    """Yöntem listesini oluşturur."""
    if args.param_sweep:
        return [
            DCPMethod(patch_size=7),
            DCPMethod(patch_size=15),
            CLAHEMethod(),
            RetinexMethod(sigma_list=[15, 80, 250]),
            RetinexMethod(sigma_list=[15, 300]),
        ]
    if "all" in args.methods:
        return [create_method(n) for n in METHOD_REGISTRY.keys()]
    return [create_method(n) for n in args.methods]


def _load_pair_by_id(
    dataset: str,
    image_id: str,
    hazy_variant: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, str] | None:
    """
    Belirli image_id ve variant için (hazy, clear, filename) döndürür.

    Returns:
        (hazy_image, clear_image, filename) veya None
    """
    if dataset == "indoor":
        csv_path = config.METADATA_INDOOR_CSV
        df = pd.read_csv(csv_path)
        row = df[df["image_id"].astype(str) == str(image_id)]
        if row.empty:
            logger.error(f"Image ID bulunamadı: {image_id}")
            return None

        row = row.iloc[0]
        clear_rel = row["clear_image_path"]
        hazy_list = ast.literal_eval(row["hazy_image_paths"])

        idx = min(hazy_variant - 1, len(hazy_list) - 1)
        hazy_rel = hazy_list[idx]

        clear_path = config.INDOOR_DIR / clear_rel
        hazy_path  = config.INDOOR_DIR / hazy_rel

        clear = cv2.imread(str(clear_path))
        hazy  = cv2.imread(str(hazy_path))

        if hazy is None:
            logger.error(f"Hazy görüntü okunamadı: {hazy_path}")
            return None

        return hazy, clear, Path(hazy_path).name

    else:  # outdoor
        hazy_dir  = config.OUTDOOR_DIR / "hazy"
        clear_dir = config.OUTDOOR_DIR / "clear"

        # image_id ile eşleşen ilk hazy dosyasını bul
        candidates = sorted(hazy_dir.glob(f"{image_id}_*"))
        if not candidates:
            logger.error(f"Outdoor hazy dosyası bulunamadı: {image_id}_*")
            return None

        idx = min(hazy_variant - 1, len(candidates) - 1)
        hazy_path = candidates[idx]
        img_id_stem = hazy_path.stem.split("_")[0]

        # Clear eşleştir
        clear_candidates = list(clear_dir.glob(f"{img_id_stem}.*"))
        clear = cv2.imread(str(clear_candidates[0])) if clear_candidates else None
        hazy  = cv2.imread(str(hazy_path))

        return hazy, clear, hazy_path.name


def _load_random_samples(dataset: str, n: int = 3) -> list[dict]:
    """Veri setinden n rastgele görüntü çifti yükler."""
    if dataset == "indoor":
        loader = MetadataDataLoader(
            csv_path=config.METADATA_INDOOR_CSV,
            dataset_dir=config.INDOOR_DIR,
            max_hazy_per_image=1,
        )
    else:
        loader = OutdoorDataLoader(dataset_dir=config.OUTDOOR_DIR)

    samples = []
    for batch in loader.load_batch(batch_size=n):
        for item in batch:
            filename, hazy, clear = item[0], item[1], item[2] if len(item) > 2 else None
            samples.append({"filename": filename, "hazy": hazy, "clear": clear})
            if len(samples) >= n:
                break
        break

    return samples


def _run_analysis_for_pair(
    hazy: np.ndarray,
    clear: np.ndarray | None,
    methods: list,
    filename: str,
    analyzer: HistogramAnalyzer,
    reporter: VisualReport,
    args,
) -> dict:
    """Tek bir görüntü çifti için tam analiz çalıştırır."""
    logger.info(f"\n🔬 Analiz: {filename}")

    # Metrikleri hesapla
    comparator = MethodComparator(methods)
    result = comparator.compare_single(hazy, filename, reference_image=clear)

    metrics_data = {
        name: res["metrics"]
        for name, res in result["results"].items()
        if "error" not in res
    }

    # İşlenmiş görüntüler
    processed_images = {
        name: res["image"]
        for name, res in result["results"].items()
        if res.get("image") is not None
    }

    # ─── Histogram analizi ────────────────────────────────────────────
    hist_path = analyzer.analyze_pair(
        hazy_image=hazy,
        clear_image=clear,
        methods=methods,
        filename=filename,
        metrics_data=metrics_data,
    )

    # ─── Sunum görseli ────────────────────────────────────────────────
    grid_path = None
    if args.visual_report or args.param_sweep:
        grid_path = reporter.comparison_grid(
            hazy_image=hazy,
            processed_images=processed_images,
            filename=filename,
            clear_image=clear,
            metrics_data=metrics_data,
            title=f"Yöntem Karşılaştırması — {filename}",
        )

    return {
        "result": result,
        "metrics_data": metrics_data,
        "processed_images": processed_images,
        "hist_path": hist_path,
        "grid_path": grid_path,
    }


def main() -> None:
    args = parse_args()

    print("\n" + "═" * 60)
    print("  📊  Görüntü Analiz ve Raporlama Aracı")
    print("═" * 60)

    # ─── Çıktı dizinleri ─────────────────────────────────────────────
    base_out = Path(args.output) if args.output else Path(config.OUTPUT_DIR) / args.dataset
    analiz_dir = base_out / "analiz"
    rapor_dir  = base_out / "rapor"

    analyzer = HistogramAnalyzer(output_dir=analiz_dir)
    reporter = VisualReport(output_dir=rapor_dir)

    # ─── Mevcut CSV'den bar chart ────────────────────────────────────
    if args.from_csv:
        csv_path = Path(args.from_csv)
        if not csv_path.exists():
            logger.error(f"CSV bulunamadı: {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path, index_col=0)
        out = reporter.metric_barchart(df, filename="metrik_barchart", title="Yöntem Karşılaştırması")
        print(f"\n📊 Bar chart: {out}")
        print("\n✅ Tamamlandı.\n")
        return

    # ─── Yöntemleri oluştur ───────────────────────────────────────────
    methods = _build_methods(args)
    logger.info(f"  Yöntemler: {', '.join(m.get_name() for m in methods)}")

    all_results = []
    all_processed_samples = []

    # ─── Belirli image_id modu ────────────────────────────────────────
    if args.image_id:
        pair = _load_pair_by_id(args.dataset, args.image_id, args.hazy_variant)
        if pair is None:
            sys.exit(1)
        hazy, clear, filename = pair

        analysis = _run_analysis_for_pair(hazy, clear, methods, filename, analyzer, reporter, args)
        all_results.append(analysis["result"])

        # Parametre sweep raporu
        if args.param_sweep:
            for name, img in analysis["processed_images"].items():
                all_processed_samples.append({
                    "label": name,
                    "image": img,
                    "metrics": analysis["metrics_data"].get(name, {}),
                })

    # ─── Rastgele N örnek modu ────────────────────────────────────────
    else:
        samples = _load_random_samples(args.dataset, args.sample_n)
        if not samples:
            logger.error("Hiç görüntü yüklenemedi.")
            sys.exit(1)

        for s in samples:
            analysis = _run_analysis_for_pair(
                s["hazy"], s["clear"], methods, s["filename"],
                analyzer, reporter, args,
            )
            all_results.append(analysis["result"])

            if args.param_sweep and not all_processed_samples:
                for name, img in analysis["processed_images"].items():
                    all_processed_samples.append({
                        "label": name,
                        "image": img,
                        "metrics": analysis["metrics_data"].get(name, {}),
                    })

    # ─── Özet istatistikler + bar chart ───────────────────────────────
    if all_results:
        comparator = MethodComparator(methods)
        summary_df = comparator.aggregate_results(all_results)

        if not summary_df.empty:
            comparator.print_comparison_table(summary_df, len(all_results))

            bar_path = reporter.metric_barchart(
                summary_df,
                filename="metrik_barchart",
                title=f"Yöntem Karşılaştırması — {args.dataset.upper()} (n={len(all_results)})",
            )

            # Parametre sweep raporu
            if args.param_sweep and all_processed_samples:
                sweep_path = reporter.param_sweep_report(
                    summary_df=summary_df,
                    sample_images=all_processed_samples[:3],
                    filename="param_karsilastirma",
                    title=f"Parametre Karşılaştırması — {args.dataset.upper()}",
                )
                print(f"\n📑 Parametre raporu: {sweep_path}")

    print(f"\n📂 Çıktılar: {analiz_dir}")
    print(f"             {rapor_dir}")
    print("\n✅ Tamamlandı.\n")


if __name__ == "__main__":
    main()
