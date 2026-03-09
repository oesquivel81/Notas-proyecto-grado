#!/usr/bin/env python3
"""Genera graficos EDA para MaIA_Scoliosis_Dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_png_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as f:
        sig = f.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            return None
        _length = int.from_bytes(f.read(4), "big")
        chunk_type = f.read(4)
        if chunk_type != b"IHDR":
            return None
        width = int.from_bytes(f.read(4), "big")
        height = int.from_bytes(f.read(4), "big")
        return width, height


def parse_jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            return None
        while True:
            b = f.read(1)
            if not b:
                return None
            if b != b"\xff":
                continue
            marker = f.read(1)
            if not marker:
                return None
            while marker == b"\xff":
                marker = f.read(1)
                if not marker:
                    return None

            if marker in {b"\xd8", b"\xd9", b"\x01"} or (0xD0 <= marker[0] <= 0xD7):
                continue

            seg_len = int.from_bytes(f.read(2), "big")
            if seg_len < 2:
                return None

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
                h = int.from_bytes(data[1:3], "big")
                w = int.from_bytes(data[3:5], "big")
                return w, h

            f.seek(seg_len - 2, 1)


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    suf = path.suffix.lower()
    if suf == ".png":
        return parse_png_size(path)
    if suf in {".jpg", ".jpeg"}:
        return parse_jpeg_size(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/figures"))
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = read_csv(root / "dataset_index.csv")
    metrics = read_csv(root / "RadiographMetrics" / "radiograph_metrics.csv")

    # 1) Split counts.
    split_counts: Dict[str, int] = {}
    for r in idx:
        s = r["split"]
        split_counts[s] = split_counts.get(s, 0) + 1

    plt.figure(figsize=(6, 4))
    names = sorted(split_counts.keys())
    vals = [split_counts[n] for n in names]
    bars = plt.bar(names, vals)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", va="bottom", fontsize=9)
    plt.title("Distribucion por split")
    plt.ylabel("Numero de imagenes")
    plt.tight_layout()
    plt.savefig(out_dir / "split_distribution.png", dpi=200)
    plt.close()

    # 2) Cobb histogram.
    cobb = [float(r["cobb_angle_deg"]) for r in metrics if r.get("cobb_angle_deg")]
    plt.figure(figsize=(7, 4))
    plt.hist(cobb, bins=20)
    plt.title("Histograma de angulo de Cobb (Scoliosis)")
    plt.xlabel("Cobb (grados)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(out_dir / "cobb_histogram.png", dpi=200)
    plt.close()

    # 3) Cobb boxplot.
    plt.figure(figsize=(4, 5))
    plt.boxplot(cobb, vert=True, tick_labels=["Scoliosis"])
    plt.ylabel("Cobb (grados)")
    plt.title("Boxplot de Cobb")
    plt.tight_layout()
    plt.savefig(out_dir / "cobb_boxplot.png", dpi=200)
    plt.close()

    # 4) Resolution scatter.
    widths: List[int] = []
    heights: List[int] = []
    for r in idx:
        p = root / r["radiograph_path"]
        sz = image_size(p)
        if sz is None:
            continue
        w, h = sz
        widths.append(w)
        heights.append(h)

    plt.figure(figsize=(6, 5))
    plt.scatter(widths, heights, s=14)
    plt.title("Dispersion de resoluciones")
    plt.xlabel("Ancho (px)")
    plt.ylabel("Alto (px)")
    plt.tight_layout()
    plt.savefig(out_dir / "resolution_scatter.png", dpi=200)
    plt.close()

    print(f"Graficos generados en: {out_dir}")


if __name__ == "__main__":
    main()
