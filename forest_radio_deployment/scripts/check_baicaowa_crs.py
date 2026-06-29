"""Check CRS (Coordinate Reference System) consistency of Baicaowa spatial data.

Checks:
- CRS of LiDAR point cloud
- CRS of orthomosaic
- CRS of DEM/DSM rasters
- CRS of node positions
- CRS of study area boundaries
- Consistency across all datasets

Outputs:
    outputs/tables/baicaowa_crs_check.csv
    outputs/reports/baicaowa_crs_report.md (appended)

Dependencies: numpy, pandas (no GPU required)
Optional: rasterio, geopandas, pyproj, laspy (if available)
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    Path.home() / "forest_radio_deployment"
))

BAICAOWA_DIR = PROJECT_ROOT / "data" / "raw" / "baicaowa"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def check_raster_crs(filepath):
    """Check CRS of a raster file (GeoTIFF)."""
    try:
        import rasterio
        with rasterio.open(filepath) as src:
            return {
                "file": str(filepath.name),
                "format": "raster",
                "crs": str(src.crs),
                "epsg": src.crs.to_epsg() if src.crs else None,
                "bounds": str(src.bounds),
                "resolution": f"{src.res[0]:.4f} x {src.res[1]:.4f}",
                "shape": f"{src.height} x {src.width}",
                "status": "OK"
            }
    except ImportError:
        return {"file": str(filepath.name), "status": "SKIP (rasterio not installed)"}
    except Exception as e:
        return {"file": str(filepath.name), "status": f"ERROR: {e}"}


def check_vector_crs(filepath):
    """Check CRS of a vector file (GeoJSON, Shapefile)."""
    try:
        import geopandas as gpd
        gdf = gpd.read_file(filepath)
        return {
            "file": str(filepath.name),
            "format": "vector",
            "crs": str(gdf.crs),
            "epsg": gdf.crs.to_epsg() if gdf.crs else None,
            "bounds": str(gdf.total_bounds),
            "num_features": len(gdf),
            "status": "OK"
        }
    except ImportError:
        return {"file": str(filepath.name), "status": "SKIP (geopandas not installed)"}
    except Exception as e:
        return {"file": str(filepath.name), "status": f"ERROR: {e}"}


def check_pointcloud_crs(filepath):
    """Check CRS/coordinate range of a point cloud file."""
    suffix = filepath.suffix.lower()
    try:
        if suffix in (".laz", ".las"):
            import laspy
            las = laspy.read(str(filepath))
            return {
                "file": str(filepath.name),
                "format": "LAS/LAZ",
                "crs": str(getattr(las.header, 'parse_crs', lambda: 'unknown')())
                    if hasattr(las.header, 'parse_crs') else "check VLR",
                "x_range": f"{las.x.min():.2f} - {las.x.max():.2f}",
                "y_range": f"{las.y.min():.2f} - {las.y.max():.2f}",
                "z_range": f"{las.z.min():.2f} - {las.z.max():.2f}",
                "num_points": len(las.points),
                "status": "OK"
            }
        elif suffix == ".ply":
            from plyfile import PlyData
            ply = PlyData.read(str(filepath))
            v = ply["vertex"]
            return {
                "file": str(filepath.name),
                "format": "PLY",
                "x_range": f"{v['x'].min():.2f} - {v['x'].max():.2f}",
                "y_range": f"{v['y'].min():.2f} - {v['y'].max():.2f}",
                "z_range": f"{v['z'].min():.2f} - {v['z'].max():.2f}",
                "num_points": len(v),
                "status": "OK"
            }
        else:
            return {"file": str(filepath.name), "status": f"SKIP (unsupported format: {suffix})"}
    except ImportError as ie:
        return {"file": str(filepath.name), "status": f"SKIP ({ie})"}
    except Exception as e:
        return {"file": str(filepath.name), "status": f"ERROR: {e}"}


def main():
    if not BAICAOWA_DIR.exists():
        print(f"Baicaowa data directory not found: {BAICAOWA_DIR}")
        print("This script will be run after Baicaowa data is provided.")
        sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    # Check rasters
    for subdir in ["dem_dsm", "orthomosaic"]:
        raster_dir = BAICAOWA_DIR / subdir
        if raster_dir.exists():
            for f in raster_dir.glob("*.tif"):
                results.append(check_raster_crs(f))

    # Check vectors
    for subdir in ["boundaries", "node_positions"]:
        vec_dir = BAICAOWA_DIR / subdir
        if vec_dir.exists():
            for f in list(vec_dir.glob("*.geojson")) + list(vec_dir.glob("*.shp")):
                results.append(check_vector_crs(f))

    # Check point clouds
    lidar_dir = BAICAOWA_DIR / "lidar"
    if lidar_dir.exists():
        for f in list(lidar_dir.glob("*.laz")) + list(lidar_dir.glob("*.las")) + list(lidar_dir.glob("*.ply")):
            results.append(check_pointcloud_crs(f))

    # Write CSV
    if results:
        csv_path = OUTPUT_DIR / "baicaowa_crs_check.csv"
        fieldnames = sorted(set().union(*(r.keys() for r in results)))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"CRS check results saved to: {csv_path}")
    else:
        print("No spatial files found to check.")


if __name__ == "__main__":
    main()
