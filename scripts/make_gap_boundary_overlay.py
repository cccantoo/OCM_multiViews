import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    maximum_filter,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_coverage(coords_rc: np.ndarray, height: int, width: int, fill_length: int) -> np.ndarray:
    coverage = np.zeros((height, width), dtype=bool)
    radius = max(0, int(fill_length) // 2)
    for r, c in coords_rc.astype(np.int32):
        if r < 0 or r >= height or c < 0 or c >= width:
            continue
        r0, r1 = max(0, r - radius), min(height, r + radius + 1)
        c0, c1 = max(0, c - radius), min(width, c + radius + 1)
        coverage[r0:r1, c0:c1] = True
    return coverage


def extract_gap_centerlines(
    coverage: np.ndarray,
    min_gap_radius: float = 1.0,
    max_gap_radius: float = 12.0,
    dilation_radius: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    filled = binary_fill_holes(coverage)
    internal_gaps = filled & (~coverage)
    if dilation_radius > 0:
        internal_gaps = binary_dilation(
            internal_gaps,
            structure=np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool),
        )
        internal_gaps &= filled
        internal_gaps &= ~coverage

    dist = distance_transform_edt(internal_gaps)
    ridge = dist == maximum_filter(dist, size=3)
    centerlines = internal_gaps & ridge & (dist >= float(min_gap_radius)) & (dist <= float(max_gap_radius))
    return internal_gaps, dist, centerlines


def paint_overlay(base_rgb: np.ndarray, line_mask: np.ndarray, line_width: int = 1) -> np.ndarray:
    out = base_rgb.copy()
    if line_width > 1:
        radius = max(0, line_width // 2)
        line_mask = binary_dilation(
            line_mask,
            structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool),
        )
    out[line_mask] = 0
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract internal gap centerlines and overlay them onto an OCM image.")
    parser.add_argument("--ocm_dir", required=True, help="Existing OCM view directory")
    parser.add_argument("--image_name", default="ocm_image.png", help="Base OCM image filename")
    parser.add_argument("--hint_name", default="gap_boundary_hint.png", help="Output hint mask filename")
    parser.add_argument(
        "--overlay_name",
        default="ocm_image_gap_boundary_overlay.png",
        help="Output overlay image filename",
    )
    parser.add_argument("--line_width", type=int, default=1, help="Overlay line width in pixels")
    parser.add_argument("--min_gap_radius", type=float, default=1.0, help="Minimum gap half-width to keep")
    parser.add_argument("--max_gap_radius", type=float, default=10.0, help="Maximum gap half-width to keep")
    parser.add_argument("--gap_dilation", type=int, default=0, help="Optional dilation on internal gaps before centerline extraction")
    args = parser.parse_args()

    ocm_dir = Path(args.ocm_dir)
    meta = json.loads((ocm_dir / "ocm_metadata.json").read_text(encoding="utf-8"))
    coords_rc = np.load(ocm_dir / "coords_rc.npy")
    image_path = ocm_dir / args.image_name
    base_rgb = np.array(Image.open(image_path).convert("RGB"))

    height = int(meta["image_height"])
    width = int(meta["image_width"])
    fill_length = int(meta["fill_length_FL"])
    coverage = build_coverage(coords_rc, height, width, fill_length)
    internal_gaps, dist, centerlines = extract_gap_centerlines(
        coverage,
        min_gap_radius=args.min_gap_radius,
        max_gap_radius=args.max_gap_radius,
        dilation_radius=args.gap_dilation,
    )

    hint = np.zeros((height, width), dtype=np.uint8)
    hint[centerlines] = 255
    overlay = paint_overlay(base_rgb, centerlines, line_width=args.line_width)

    hint_path = ocm_dir / args.hint_name
    overlay_path = ocm_dir / args.overlay_name
    meta_path = ocm_dir / "gap_boundary_metadata.json"
    Image.fromarray(hint).save(hint_path)
    Image.fromarray(overlay).save(overlay_path)

    out_meta = {
        "source_ocm_dir": str(ocm_dir),
        "base_image": str(image_path),
        "hint_image": str(hint_path),
        "overlay_image": str(overlay_path),
        "fill_length_FL": fill_length,
        "line_width": int(args.line_width),
        "min_gap_radius": float(args.min_gap_radius),
        "max_gap_radius": float(args.max_gap_radius),
        "gap_dilation": int(args.gap_dilation),
        "internal_gap_pixels": int(internal_gaps.sum()),
        "centerline_pixels": int(centerlines.sum()),
        "gap_radius_stats": {
            "p50": float(np.percentile(dist[internal_gaps], 50)) if np.any(internal_gaps) else 0.0,
            "p90": float(np.percentile(dist[internal_gaps], 90)) if np.any(internal_gaps) else 0.0,
            "max": float(dist[internal_gaps].max()) if np.any(internal_gaps) else 0.0,
        },
    }
    meta_path.write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved hint image: {hint_path}")
    print(f"saved overlay image: {overlay_path}")
    print(f"internal_gap_pixels={int(internal_gaps.sum())}, centerline_pixels={int(centerlines.sum())}")


if __name__ == "__main__":
    main()
