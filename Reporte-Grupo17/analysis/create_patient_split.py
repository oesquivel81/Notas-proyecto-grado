#!/usr/bin/env python3
"""Crea split train/val/test estratificado por paciente sin leakage."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def read_csv(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def assign_splits(items: List[str], ratios: Tuple[float, float, float]) -> Dict[str, str]:
    n = len(items)
    n_train = round(n * ratios[0])
    n_val = round(n * ratios[1])
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val

    assignment = {}
    for i, pid in enumerate(items):
        if i < n_train:
            assignment[pid] = "train"
        elif i < n_train + n_val:
            assignment[pid] = "val"
        else:
            assignment[pid] = "test"
    assert sum(1 for _ in assignment) == n
    assert n_test >= 0
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("analysis/dataset_index_patient_split.csv"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("analysis/patient_split_summary.json"),
    )
    args = parser.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-8:
        raise ValueError("Las proporciones train+val+test deben sumar 1.0")

    root = args.dataset_root.resolve()
    rows, fieldnames = read_csv(root / "dataset_index.csv")

    # Agrupacion por paciente y estrato (si tiene alguna imagen scoliosis).
    by_patient: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    strat_scoliosis: List[str] = []
    strat_normal_only: List[str] = []

    for pid, grp in by_patient.items():
        has_scoliosis = any(g["split"].strip().lower() == "scoliosis" for g in grp)
        if has_scoliosis:
            strat_scoliosis.append(pid)
        else:
            strat_normal_only.append(pid)

    rng = random.Random(args.seed)
    rng.shuffle(strat_scoliosis)
    rng.shuffle(strat_normal_only)

    assign = {}
    assign.update(assign_splits(strat_scoliosis, (args.train, args.val, args.test)))
    assign.update(assign_splits(strat_normal_only, (args.train, args.val, args.test)))

    out_rows: List[dict] = []
    for r in rows:
        rr = dict(r)
        rr["split_patient"] = assign[r["patient_id"]]
        out_rows.append(rr)

    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fields = fieldnames + ["split_patient"] if "split_patient" not in fieldnames else fieldnames
    write_csv(out_csv, out_rows, out_fields)

    # Validacion leakage por paciente.
    patient_split_membership: Dict[str, set] = defaultdict(set)
    for r in out_rows:
        patient_split_membership[r["patient_id"]].add(r["split_patient"])
    leakage_patients = [pid for pid, s in patient_split_membership.items() if len(s) > 1]

    # Resumen.
    row_counts = Counter(r["split_patient"] for r in out_rows)
    patient_counts = Counter(assign.values())

    # Proporcion de scoliosis por split_patient.
    scol_by_split = Counter()
    total_by_split = Counter()
    for r in out_rows:
        sp = r["split_patient"]
        total_by_split[sp] += 1
        if r["split"].strip().lower() == "scoliosis":
            scol_by_split[sp] += 1

    scol_rate = {
        k: (scol_by_split[k] / total_by_split[k] if total_by_split[k] else 0.0)
        for k in sorted(total_by_split)
    }

    summary = {
        "seed": args.seed,
        "ratios": {"train": args.train, "val": args.val, "test": args.test},
        "patients_total": len(by_patient),
        "rows_total": len(out_rows),
        "patient_counts_by_split": dict(patient_counts),
        "row_counts_by_split": dict(row_counts),
        "scoliosis_rate_by_split": scol_rate,
        "leakage_patients_count": len(leakage_patients),
        "leakage_patients": leakage_patients[:20],
    }

    out_json = args.out_json.resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"CSV generado: {out_csv}")
    print(f"Resumen JSON: {out_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
