#!/usr/bin/env python3
"""EDA reproducible para MaIA_Scoliosis_Dataset.

Uso:
  python analysis/eda_maia_dataset.py --dataset-root MaIA_Scoliosis_Dataset
  python analysis/eda_maia_dataset.py --dataset-root MaIA_Scoliosis_Dataset --out-json analysis/eda_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_png_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with path.open("rb") as f:
            sig = f.read(8)
            if sig != b"\x89PNG\r\n\x1a\n":
                return None
            length = int.from_bytes(f.read(4), "big")
            chunk_type = f.read(4)
            if chunk_type != b"IHDR" or length < 8:
                return None
            width = int.from_bytes(f.read(4), "big")
            height = int.from_bytes(f.read(4), "big")
            return width, height
    except OSError:
        return None


def parse_jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    try:
        with path.open("rb") as f:
            if f.read(2) != b"\xff\xd8":
                return None
            while True:
                marker_start = f.read(1)
                if not marker_start:
                    return None
                if marker_start != b"\xff":
                    continue

                marker = f.read(1)
                if not marker:
                    return None

                while marker == b"\xff":
                    marker = f.read(1)
                    if not marker:
                        return None

                # Standalone markers without length.
                if marker in {b"\xd8", b"\xd9", b"\x01"} or (0xD0 <= marker[0] <= 0xD7):
                    continue

                seg_len_bytes = f.read(2)
                if len(seg_len_bytes) != 2:
                    return None
                seg_len = int.from_bytes(seg_len_bytes, "big")
                if seg_len < 2:
                    return None

                # SOF markers with dimensions.
                if marker in {
                    b"\xc0",
                    b"\xc1",
                    b"\xc2",
                    b"\xc3",
                    b"\xc5",
                    b"\xc6",
                    b"\xc7",
                    b"\xc9",
                    b"\xca",
                    b"\xcb",
                    b"\xcd",
                    b"\xce",
                    b"\xcf",
                }:
                    data = f.read(seg_len - 2)
                    if len(data) < 5:
                        return None
                    height = int.from_bytes(data[1:3], "big")
                    width = int.from_bytes(data[3:5], "big")
                    return width, height

                f.seek(seg_len - 2, 1)
    except OSError:
        return None


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return parse_png_size(path)
    if suffix in {".jpg", ".jpeg"}:
        return parse_jpeg_size(path)
    return None


def is_standard_vertebra_label(name: str) -> bool:
    if name == "background":
        return True
    if re.fullmatch(r"C[3-7]", name):
        return True
    if re.fullmatch(r"T([1-9]|1[0-2])", name):
        return True
    if re.fullmatch(r"L[1-5]", name):
        return True
    return False


def safe_float(v: str) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = (len(sorted_vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA reproducible para MaIA_Scoliosis_Dataset")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Ruta raíz del dataset")
    parser.add_argument("--out-json", type=Path, default=None, help="Ruta opcional para guardar reporte JSON")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    index_path = root / "dataset_index.csv"
    labels_path = root / "labels_dictionary.json"
    metrics_csv_path = root / "RadiographMetrics" / "radiograph_metrics.csv"

    if not index_path.exists():
        raise FileNotFoundError(f"No existe: {index_path}")

    dataset_rows = read_csv(index_path)
    dataset_cols = list(dataset_rows[0].keys()) if dataset_rows else []

    report: Dict[str, object] = {
        "dataset_root": str(root),
        "dataset_index": {
            "rows": len(dataset_rows),
            "columns": dataset_cols,
        },
    }

    # Folder-level counts.
    folder_counts: Dict[str, int] = {}
    for p in sorted(root.iterdir()):
        if p.is_dir():
            folder_counts[p.name] = sum(1 for _ in p.rglob("*") if _.is_file())
    report["folder_file_counts"] = folder_counts

    # Split distribution.
    split_counts = Counter(r.get("split", "") for r in dataset_rows)
    report["split_distribution"] = dict(split_counts)

    # Path consistency checks.
    path_cols = [
        "radiograph_path",
        "label_binary_path",
        "multiclass_id_png",
        "multiclass_gray_jpg",
        "multiclass_color_jpg",
        "metrics_json",
    ]
    missing_by_col: Dict[str, int] = {}
    for col in path_cols:
        missing = 0
        for r in dataset_rows:
            val = (r.get(col) or "").strip()
            if not val:
                missing += 1
                continue
            if not (root / val).exists():
                missing += 1
        missing_by_col[col] = missing
    report["missing_paths_by_column"] = missing_by_col

    # Patient ID consistency and leakage risk.
    id_to_splits: Dict[str, set] = defaultdict(set)
    id_counts = Counter()
    for r in dataset_rows:
        pid = (r.get("patient_id") or "").strip()
        sp = (r.get("split") or "").strip()
        id_counts[pid] += 1
        id_to_splits[pid].add(sp)

    multi_image_ids = [pid for pid, c in id_counts.items() if c > 1]
    cross_split_ids = [pid for pid, s in id_to_splits.items() if len(s) > 1]
    report["patient_id_checks"] = {
        "unique_patient_ids": len(id_counts),
        "ids_with_more_than_one_image": len(multi_image_ids),
        "max_images_per_patient_id": max(id_counts.values()) if id_counts else 0,
        "ids_appearing_in_multiple_splits": len(cross_split_ids),
        "sample_cross_split_ids": cross_split_ids[:20],
    }

    # Labels analysis.
    if labels_path.exists():
        labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
        multiclass = labels_data.get("multiclass_id_png", {})
        total_multiclass_labels = len(multiclass)
        non_standard = {}
        for k, v in multiclass.items():
            name = str(v)
            if not is_standard_vertebra_label(name):
                non_standard[k] = name

        report["labels"] = {
            "multiclass_label_count": total_multiclass_labels,
            "non_standard_labels_count": len(non_standard),
            "non_standard_labels": non_standard,
        }

    # Metrics CSV analysis.
    if metrics_csv_path.exists():
        metrics_rows = read_csv(metrics_csv_path)
        cobb_vals = [safe_float(r.get("cobb_angle_deg", "")) for r in metrics_rows]
        cobb_vals = [v for v in cobb_vals if v is not None]
        cobb_sorted = sorted(cobb_vals)

        metrics_report: Dict[str, object] = {
            "rows": len(metrics_rows),
            "columns": list(metrics_rows[0].keys()) if metrics_rows else [],
            "cobb_non_null": len(cobb_vals),
        }

        if cobb_vals:
            metrics_report["cobb_stats_deg"] = {
                "min": min(cobb_vals),
                "mean": statistics.fmean(cobb_vals),
                "median": statistics.median(cobb_vals),
                "max": max(cobb_vals),
                "p25": quantile(cobb_sorted, 0.25),
                "p75": quantile(cobb_sorted, 0.75),
            }
            metrics_report["cobb_buckets_deg"] = {
                "lt_10": sum(1 for v in cobb_vals if v < 10),
                "between_10_25": sum(1 for v in cobb_vals if 10 <= v < 25),
                "between_25_40": sum(1 for v in cobb_vals if 25 <= v < 40),
                "ge_40": sum(1 for v in cobb_vals if v >= 40),
            }

        report["radiograph_metrics"] = metrics_report

    # Resolution analysis from radiograph paths in index.
    widths: List[int] = []
    heights: List[int] = []
    ars: List[float] = []
    unique_res = set()
    unreadable_images = 0

    for r in dataset_rows:
        rel = (r.get("radiograph_path") or "").strip()
        if not rel:
            continue
        p = root / rel
        size = image_size(p)
        if size is None:
            unreadable_images += 1
            continue
        w, h = size
        widths.append(w)
        heights.append(h)
        ars.append(w / h if h else 0.0)
        unique_res.add((w, h))

    resolution_report: Dict[str, object] = {
        "readable_images": len(widths),
        "unreadable_images": unreadable_images,
        "unique_resolutions": len(unique_res),
    }

    if widths and heights:
        resolution_report.update(
            {
                "width": {
                    "min": min(widths),
                    "mean": statistics.fmean(widths),
                    "max": max(widths),
                },
                "height": {
                    "min": min(heights),
                    "mean": statistics.fmean(heights),
                    "max": max(heights),
                },
                "aspect_ratio": {
                    "min": min(ars),
                    "mean": statistics.fmean(ars),
                    "max": max(ars),
                },
            }
        )

    report["resolution_analysis"] = resolution_report

    # Console summary.
    print("=== MaIA_Scoliosis_Dataset EDA ===")
    print(f"Root: {root}")
    print(f"Rows in dataset_index.csv: {len(dataset_rows)}")
    print("Split distribution:", dict(split_counts))
    print("Missing paths by column:", missing_by_col)
    print("Patient ID checks:", report["patient_id_checks"])

    if "labels" in report:
        labels_report = report["labels"]
        print(
            "Labels: total multiclass=",
            labels_report["multiclass_label_count"],
            " non-standard=",
            labels_report["non_standard_labels_count"],
        )

    if "radiograph_metrics" in report:
        mrep = report["radiograph_metrics"]
        print("Metrics rows:", mrep["rows"])
        if "cobb_stats_deg" in mrep:
            print("Cobb stats (deg):", mrep["cobb_stats_deg"])
            print("Cobb buckets:", mrep["cobb_buckets_deg"])

    print("Resolution analysis:", report["resolution_analysis"])

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report saved to: {args.out_json}")


if __name__ == "__main__":
    main()
