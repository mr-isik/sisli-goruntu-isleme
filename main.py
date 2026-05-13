#!/usr/bin/env python3
"""
Sisli Görüntü İyileştirme - Ana Giriş Noktası

DCP (Dark Channel Prior), CLAHE ve Multi-Scale Retinex with Color Restoration
yöntemlerini kullanarak sisli görüntüleri iyileştirir ve karşılaştırır.

Kullanım:
    # Indoor veri seti (metadata CSV ile)
    python main.py --dataset indoor --methods all

    # Outdoor veri seti
    python main.py --dataset outdoor --methods all

    # Tüm veri setleri
    python main.py --dataset all --methods all

    # Belirli yöntemle, özel batch boyutu
    python main.py --dataset indoor --methods dcp clahe --batch-size 32

    # Her clear başına sadece 3 hazy variant işle
    python main.py --dataset indoor --hazy-per-image 3

    # Sıralı işleme (debug)
    python main.py --dataset indoor --no-parallel

    # Eski mod: dizin bazlı işleme (metadata CSV olmadan)
    python main.py --input input/indoor/hazy --output output/ --methods all
"""

import argparse
import sys
from pathlib import Path

# Proje kökünü Python path'e ekle
sys.path.insert(0, str(Path(__file__).parent))

import config
from methods import create_method, METHOD_REGISTRY
from pipeline.data_loader import DirectoryDataLoader
from pipeline.csv_data_loader import MetadataDataLoader, OutdoorDataLoader
from pipeline.processor import BatchProcessor
from evaluation.quality_metrics import get_no_reference_metrics
from utils.logger import get_logger

logger = get_logger("main")


def parse_args() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
    parser = argparse.ArgumentParser(
        description="Sisli Görüntü İyileştirme ve Karşılaştırma Aracı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  %(prog)s --dataset indoor --methods all
  %(prog)s --dataset outdoor --methods dcp
  %(prog)s --dataset all --batch-size 32 --workers 8
  %(prog)s --input foto.jpg --output output/ --methods dcp
        """,
    )

    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="all",
        choices=["indoor", "outdoor", "all"],
        help="Veri seti seçimi (varsayılan: all)",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Manuel girdi dizini/dosya (--dataset yerine kullanılabilir)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(config.OUTPUT_DIR),
        help=f"Çıktı dizini (varsayılan: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        default=["all"],
        choices=list(METHOD_REGISTRY.keys()) + ["all"],
        help="Kullanılacak yöntemler (varsayılan: all)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=config.DEFAULT_BATCH_SIZE,
        help=f"Batch boyutu (varsayılan: {config.DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=config.DEFAULT_WORKERS,
        help=f"Paralel worker sayısı (varsayılan: {config.DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--hazy-per-image",
        type=int,
        default=None,
        help="Her clear görüntü başına işlenecek maks hazy sayısı (varsayılan: tümü)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        default=True,
        help="CSV çıktısı üret (varsayılan: aktif)",
    )
    parser.add_argument(
        "--no-visuals",
        action="store_true",
        help="Görsel karşılaştırma üretme",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Sıralı işleme (paralel yerine, debug için)",
    )
    parser.add_argument(
        "--param-sweep",
        action="store_true",
        help="DCP (patch=7 vs 15) ve MSRCR (sigma=[15,80,250] vs [15,300]) parametre karşılaştırması",
    )

    return parser.parse_args()


def _create_data_loader(dataset: str, hazy_per_image: int | None = None):
    """Veri seti türüne göre uygun data loader oluşturur."""
    if dataset == "indoor":
        csv_path = config.METADATA_INDOOR_CSV
        if not csv_path.exists():
            logger.error(f"Metadata CSV bulunamadı: {csv_path}")
            sys.exit(1)
        return MetadataDataLoader(
            csv_path=csv_path,
            dataset_dir=config.INDOOR_DIR,
            max_hazy_per_image=hazy_per_image,
        )
    elif dataset == "outdoor":
        if not config.OUTDOOR_DIR.exists():
            logger.error(f"Outdoor dizini bulunamadı: {config.OUTDOOR_DIR}")
            sys.exit(1)
        return OutdoorDataLoader(dataset_dir=config.OUTDOOR_DIR)
    else:
        raise ValueError(f"Bilinmeyen veri seti: {dataset}")


def _build_param_sweep_methods() -> list:
    """
    Parametre sweep için yöntem listesi oluşturur.

    DCP: patch_size = 7 vs 15
    MSRCR: sigma = [15, 80, 250] vs [15, 300]
    CLAHE: sabit (referans)
    """
    from methods.dcp import DCPMethod
    from methods.retinex import RetinexMethod
    from methods.clahe import CLAHEMethod

    return [
        DCPMethod(patch_size=7),
        DCPMethod(patch_size=15),
        CLAHEMethod(),
        RetinexMethod(sigma_list=[15, 80, 250]),
        RetinexMethod(sigma_list=[15, 300]),
    ]


def _run_pipeline(
    data_loader, methods, args, output_dir: str, dataset_name: str
) -> dict:
    """Tek bir veri seti için pipeline çalıştırır."""
    total_images = data_loader.get_total_count()

    if total_images == 0:
        logger.warning(f"⚠ {dataset_name} veri setinde hiç görüntü bulunamadı, atlanıyor.")
        return {"results": [], "summary_df": None, "export_paths": []}

    logger.info(f"  📊 {dataset_name} - Toplam görüntü: {total_images}")

    processor = BatchProcessor(
        methods=methods,
        data_loader=data_loader,
        batch_size=args.batch_size,
        max_workers=args.workers,
    )

    return processor.run_with_export(
        output_dir=output_dir,
        export_csv=args.export_csv,
        export_visuals=not args.no_visuals,
        parallel=not args.no_parallel,
    )


def main() -> None:
    """Ana işlem akışı."""
    args = parse_args()

    # ─── Banner ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  🌫️  Sisli Görüntü İyileştirme Aracı")
    print("  DCP | CLAHE | Multi-Scale Retinex")
    print("═" * 60)

    # ─── Yöntemleri oluştur ──────────────────────────────────────────────
    if args.param_sweep:
        methods = _build_param_sweep_methods()
        logger.info("  🔬 Parametre sweep modu aktif:")
        for m in methods:
            logger.info(f"     → {m.get_name()}")
    elif "all" in args.methods:
        method_names = list(METHOD_REGISTRY.keys())
        methods = [create_method(n) for n in method_names]
        for m in methods:
            logger.info(f"  📌 Yöntem: {m.get_name()} | Params: {m.get_params()}")
    else:
        methods = [create_method(n) for n in args.methods]
        for m in methods:
            logger.info(f"  📌 Yöntem: {m.get_name()} | Params: {m.get_params()}")

    # ─── Veri seti modu belirleme ────────────────────────────────────────
    all_export_paths = []

    if args.input is not None:
        # Manuel mod: eski DirectoryDataLoader ile çalış
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Girdi bulunamadı: {input_path}")
            sys.exit(1)

        data_loader = DirectoryDataLoader(args.input)
        output = _run_pipeline(data_loader, methods, args, args.output, "Manuel")
        all_export_paths.extend(output["export_paths"])

    else:
        # Metadata CSV modu
        datasets_to_run = []
        if args.dataset in ("indoor", "all"):
            datasets_to_run.append("indoor")
        if args.dataset in ("outdoor", "all"):
            datasets_to_run.append("outdoor")

        for dataset_name in datasets_to_run:
            print(f"\n{'─' * 60}")
            print(f"  📁 Veri Seti: {dataset_name.upper()}")
            print(f"{'─' * 60}")

            data_loader = _create_data_loader(dataset_name, args.hazy_per_image)
            output_dir = str(Path(args.output) / dataset_name)

            output = _run_pipeline(data_loader, methods, args, output_dir, dataset_name.upper())
            all_export_paths.extend(output["export_paths"])

    # ─── Sonuç özeti ─────────────────────────────────────────────────────
    if all_export_paths:
        print("\n📁 Kaydedilen dosyalar:")
        for path in all_export_paths:
            print(f"   → {path}")

    print("\n✅ İşlem tamamlandı.\n")


if __name__ == "__main__":
    main()
