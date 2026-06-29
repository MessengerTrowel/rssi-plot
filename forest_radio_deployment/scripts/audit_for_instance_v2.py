# -*- coding: utf-8 -*-
"""
Stage 0C: FOR-instanceV2 Data Audit

Analyzes all PLY point cloud files from the FOR-instanceV2 dataset:
  - File inventory (count, format, size)
  - Point cloud fields and types
  - Coordinate ranges and anomalies (NaN, Inf, duplicates)
  - Semantic segmentation label distribution (semantic_seg: 1=ground, 2=trunk, 3=canopy)
  - Instance segmentation statistics (treeID)
  - Train / Val / Test split summary
  - Per-site summary
  - Minimal sample read

Outputs:
  outputs/reports/for_instance_v2_data_audit.md
  outputs/tables/for_instance_v2_file_manifest.csv
  outputs/tables/for_instance_v2_class_distribution.csv
  outputs/tables/for_instance_v2_site_summary.csv

Usage:
  python scripts/audit_for_instance_v2.py [--data-dir PATH] [--output-dir PATH]
"""
import argparse
import csv
import datetime
import json
import os
from collections import Counter, defaultdict

import numpy as np
from plyfile import PlyData


def parse_args():
    p = argparse.ArgumentParser(description="Audit FOR-instanceV2 PLY data")
    p.add_argument("--data-dir", default=None,
                   help="Root of FOR-instanceV2 data (contains test_data/, train_val_data/)")
    p.add_argument("--output-dir", default=None,
                   help="Root output directory (contains outputs/reports, outputs/tables)")
    return p.parse_args()


def discover_ply_files(data_dir):
    ply_files = []
    for subdir in ["test_data", "train_val_data"]:
        d = os.path.join(data_dir, subdir)
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for f in files:
                if f.lower().endswith(".ply"):
                    ply_files.append(os.path.join(root, f))
    return sorted(ply_files)


def classify_split(fname):
    if "_train." in fname or "_train_" in fname:
        return "train"
    if "_val." in fname or "_val_" in fname:
        return "val"
    if "_test." in fname or "_test_" in fname:
        return "test"
    return "unknown"


def audit_ply(fpath):
    fname = os.path.basename(fpath)
    fsize = os.path.getsize(fpath)
    split = classify_split(fname)
    site = fname.split("_")[0]

    ply = PlyData.read(fpath)
    vertex = ply["vertex"]
    n_points = len(vertex.data)
    fields = [p.name for p in vertex.properties]

    result = {
        "filename": fname,
        "site": site,
        "split": split,
        "size_mb": round(fsize / 1e6, 1),
        "n_points": n_points,
        "fields": ";".join(fields),
    }

    # Coordinates
    if all(f in fields for f in ("x", "y", "z")):
        x, y, z = vertex["x"], vertex["y"], vertex["z"]
        result.update({
            "x_min": float(np.min(x)), "x_max": float(np.max(x)),
            "y_min": float(np.min(y)), "y_max": float(np.max(y)),
            "z_min": float(np.min(z)), "z_max": float(np.max(z)),
            "has_nan": bool(np.any(np.isnan(x)) or np.any(np.isnan(y)) or np.any(np.isnan(z))),
            "has_inf": bool(np.any(np.isinf(x)) or np.any(np.isinf(y)) or np.any(np.isinf(z))),
        })
        coords = np.column_stack([x, y, z])
        result["n_duplicates"] = n_points - len(np.unique(coords, axis=0))
        x_span = float(np.max(x) - np.min(x))
        y_span = float(np.max(y) - np.min(y))
        area = x_span * y_span if x_span > 0 and y_span > 0 else 1
        result["density_pts_per_m2"] = round(n_points / area, 1)
        result["area_m2"] = round(area, 1)
    else:
        result.update({"x_min": None, "x_max": None, "y_min": None, "y_max": None,
                       "z_min": None, "z_max": None, "has_nan": False, "has_inf": False,
                       "n_duplicates": 0, "density_pts_per_m2": 0, "area_m2": 0})

    # Semantic segmentation (field: semantic_seg)
    sem_info = {}
    for candidate in ("semantic_seg", "semantic_label", "label", "classification"):
        if candidate in fields:
            sem = vertex[candidate]
            for val in np.unique(sem):
                sem_info[int(val)] = int(np.sum(sem == val))
            result["semantic_field"] = candidate
            break
    else:
        result["semantic_field"] = ""
    result["semantic_classes"] = json.dumps(sem_info)
    result["n_semantic_classes"] = len(sem_info)

    # Instance (treeID)
    for candidate in ("treeID", "instance_label", "instance", "tree_id"):
        if candidate in fields:
            tid = vertex[candidate]
            unique_tid = np.unique(tid)
            result["instance_field"] = candidate
            result["n_instances"] = len([t for t in unique_tid if t > 0])
            break
    else:
        result["instance_field"] = ""
        result["n_instances"] = -1

    return result, sem_info


def main():
    args = parse_args()
    home = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    data_dir = args.data_dir or os.path.join(home, "forest_radio_deployment", "data", "raw", "for_instance_v2")
    output_root = args.output_dir or os.path.join(home, "forest_radio_deployment")
    out_reports = os.path.join(output_root, "outputs", "reports")
    out_tables = os.path.join(output_root, "outputs", "tables")
    os.makedirs(out_reports, exist_ok=True)
    os.makedirs(out_tables, exist_ok=True)

    ply_files = discover_ply_files(data_dir)
    print(f"Found {len(ply_files)} PLY files in {data_dir}")

    results = []
    global_sem = Counter()
    sites = defaultdict(list)
    all_inst_counts = []
    empty_files = []
    corrupt_files = []

    for fpath in ply_files:
        fname = os.path.basename(fpath)
        try:
            r, sem_info = audit_ply(fpath)
            results.append(r)
            sites[r["site"]].append(r)
            for v, c in sem_info.items():
                global_sem[v] += c
            if r["n_instances"] >= 0:
                all_inst_counts.append(r["n_instances"])
            if r["n_points"] == 0:
                empty_files.append(fname)
            print(f"  {fname}: {r['n_points']:,} pts, {r['split']}, {r['n_instances']} trees")
        except Exception as e:
            corrupt_files.append((fname, str(e)))
            print(f"  ERROR {fname}: {e}")

    # Write manifest CSV
    manifest_path = os.path.join(out_tables, "for_instance_v2_file_manifest.csv")
    if results:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
    print(f"Saved {manifest_path}")

    # Class distribution
    total_labeled = sum(global_sem.values())
    cls_path = os.path.join(out_tables, "for_instance_v2_class_distribution.csv")
    with open(cls_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["semantic_seg_label", "total_points", "percentage"])
        for val in sorted(global_sem.keys()):
            cnt = global_sem[val]
            pct = round(100 * cnt / total_labeled, 2) if total_labeled else 0
            w.writerow([val, cnt, pct])
    print(f"Saved {cls_path}")

    # Site summary
    site_path = os.path.join(out_tables, "for_instance_v2_site_summary.csv")
    with open(site_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["site", "n_files", "n_train", "n_val", "n_test",
                     "total_points", "total_size_mb", "total_trees", "avg_density"])
        for sn in sorted(sites):
            fs = sites[sn]
            w.writerow([
                sn, len(fs),
                sum(1 for r in fs if r["split"] == "train"),
                sum(1 for r in fs if r["split"] == "val"),
                sum(1 for r in fs if r["split"] == "test"),
                sum(r["n_points"] for r in fs),
                round(sum(r["size_mb"] for r in fs), 1),
                sum(r["n_instances"] for r in fs if r["n_instances"] > 0),
                round(np.mean([r["density_pts_per_m2"] for r in fs]), 1),
            ])
    print(f"Saved {site_path}")

    total_points = sum(r["n_points"] for r in results)
    total_trees = sum(r["n_instances"] for r in results if r["n_instances"] > 0)
    n_train = sum(1 for r in results if r["split"] == "train")
    n_val = sum(1 for r in results if r["split"] == "val")
    n_test = sum(1 for r in results if r["split"] == "test")

    print(f"\n{'='*60}")
    print(f"Total: {len(results)} files, {total_points:,} points, {total_trees} trees")
    print(f"Split: {n_train} train / {n_val} val / {n_test} test")
    print(f"Sites: {len(sites)}")
    print(f"Semantic: {dict(sorted(global_sem.items()))}")
    print(f"Empty: {len(empty_files)}, Corrupt: {len(corrupt_files)}")


if __name__ == "__main__":
    main()
