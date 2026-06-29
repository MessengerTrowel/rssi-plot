"""Convert Baicaowa point cloud data to ForestFormer3D-compatible PLY format.

ForestFormer3D requires binary PLY files with fields:
- x (float32): X coordinate
- y (float32): Y coordinate
- z (float32): Z coordinate
- semantic_seg (int32): Semantic label (1=ground, 2=wood, 3=leaf)
- treeID (int32): Instance label (0=unannotated, >0=tree ID)

If Baicaowa data lacks semantic/instance labels:
- semantic_seg is set to 1 (all points as ground / unannotated)
- treeID is set to 0 (no instance labels)
This enables inference-only mode.

Supported input formats: LAZ, LAS, PLY

Dependencies: numpy, laspy (for LAZ/LAS), plyfile (for PLY input)
No GPU required.
"""

import os
import sys
import struct
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    Path.home() / "forest_radio_deployment"
))


def read_laz(filepath):
    """Read LAZ/LAS file and return coordinates."""
    import laspy
    las = laspy.read(str(filepath))
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float64)

    # Try to read classification if available
    semantic_seg = None
    tree_id = None
    if hasattr(las, "classification"):
        semantic_seg = las.classification.astype(np.int32)
    if hasattr(las, "treeID"):
        tree_id = las.treeID.astype(np.int32)

    return points, semantic_seg, tree_id


def read_ply_input(filepath):
    """Read PLY file and return coordinates + optional labels."""
    from plyfile import PlyData
    ply = PlyData.read(str(filepath))
    v = ply["vertex"]
    points = np.vstack((v["x"], v["y"], v["z"])).T.astype(np.float64)

    semantic_seg = None
    tree_id = None
    if "semantic_seg" in v.data.dtype.names:
        semantic_seg = v["semantic_seg"].astype(np.int32)
    if "treeID" in v.data.dtype.names:
        tree_id = v["treeID"].astype(np.int32)

    return points, semantic_seg, tree_id


def write_binary_ply(filepath, points, semantic_seg, tree_id):
    """Write binary little-endian PLY compatible with ForestFormer3D plyutils.read_ply()."""
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property int semantic_seg\n"
        "property int treeID\n"
        "end_header\n"
    )

    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    z = points[:, 2].astype(np.float32)
    sem = semantic_seg.astype(np.int32)
    tid = tree_id.astype(np.int32)

    # Create structured array
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("semantic_seg", "<i4"), ("treeID", "<i4")
    ])
    data = np.empty(n, dtype=dtype)
    data["x"] = x
    data["y"] = y
    data["z"] = z
    data["semantic_seg"] = sem
    data["treeID"] = tid

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        data.tofile(f)


def convert_file(input_path, output_path, test_mode=True):
    """Convert a single point cloud file to ForestFormer3D PLY format.

    Args:
        input_path: Path to input file (LAZ, LAS, or PLY)
        output_path: Path to output PLY file
        test_mode: If True, set dummy labels for inference-only mode
    """
    suffix = input_path.suffix.lower()

    if suffix in (".laz", ".las"):
        points, semantic_seg, tree_id = read_laz(input_path)
    elif suffix == ".ply":
        points, semantic_seg, tree_id = read_ply_input(input_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    n = len(points)
    print(f"  Points: {n:,}")
    print(f"  X range: {points[:, 0].min():.2f} - {points[:, 0].max():.2f}")
    print(f"  Y range: {points[:, 1].min():.2f} - {points[:, 1].max():.2f}")
    print(f"  Z range: {points[:, 2].min():.2f} - {points[:, 2].max():.2f}")

    # Set default labels if not available or in test mode
    if semantic_seg is None or test_mode:
        semantic_seg = np.ones(n, dtype=np.int32)  # All as ground/unannotated
        print("  Semantic labels: dummy (test mode)")
    else:
        print(f"  Semantic labels: {np.unique(semantic_seg)}")

    if tree_id is None or test_mode:
        tree_id = np.zeros(n, dtype=np.int32)  # No instance labels
        print("  Tree IDs: dummy (test mode)")
    else:
        print(f"  Tree IDs: {len(np.unique(tree_id[tree_id > 0]))} unique")

    write_binary_ply(output_path, points, semantic_seg, tree_id)
    print(f"  Output: {output_path}")


def main():
    """Convert all Baicaowa point cloud files."""
    lidar_dir = PROJECT_ROOT / "data" / "raw" / "baicaowa" / "lidar"
    output_dir = PROJECT_ROOT / "data" / "processed" / "baicaowa_ply"

    if not lidar_dir.exists():
        print(f"Baicaowa LiDAR directory not found: {lidar_dir}")
        print("This script will be run after Baicaowa data is provided.")
        sys.exit(0)

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = (
        list(lidar_dir.glob("*.laz"))
        + list(lidar_dir.glob("*.las"))
        + list(lidar_dir.glob("*.ply"))
    )

    if not input_files:
        print("No point cloud files found.")
        sys.exit(0)

    print(f"Found {len(input_files)} point cloud files")
    for f in sorted(input_files):
        print(f"\nConverting: {f.name}")
        output_name = f.stem + "_baicaowa_test.ply"
        output_path = output_dir / output_name
        try:
            convert_file(f, output_path, test_mode=True)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nConversion complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
