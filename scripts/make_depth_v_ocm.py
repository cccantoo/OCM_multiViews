import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocm_rock.ocm_mapping import fill_ocm_image, save_image


def robust_normalize(values: np.ndarray, low_q: float, high_q: float) -> np.ndarray:
    lo, hi = np.percentile(values, [low_q, high_q])
    if hi <= lo + 1e-12:
        return np.full_like(values, 0.5, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a depth-aware OCM variant by writing view-depth into HSV V."
    )
    parser.add_argument("--ocm_dir", required=True, help="Existing OCM view directory")
    parser.add_argument("--out_name", default="ocm_image_depth_v.png", help="Output image filename")
    parser.add_argument("--aux_name", default="depth_aux.png", help="Output depth auxiliary filename")
    parser.add_argument("--near_bright", action="store_true", help="Make near points brighter instead of darker")
    parser.add_argument("--low_q", type=float, default=2.0, help="Low percentile for robust depth normalization")
    parser.add_argument("--high_q", type=float, default=98.0, help="High percentile for robust depth normalization")
    parser.add_argument("--v_min", type=float, default=0.45, help="Minimum brightness multiplier")
    parser.add_argument("--v_max", type=float, default=1.0, help="Maximum brightness multiplier")
    parser.add_argument("--target_void_ratio", type=float, default=None, help="Override fill target void ratio")
    parser.add_argument("--max_fill_length", type=int, default=None, help="Override max fill length")
    args = parser.parse_args()

    ocm_dir = Path(args.ocm_dir)
    points_view = np.load(ocm_dir / "points_view.npy")
    coords_rc = np.load(ocm_dir / "coords_rc.npy")
    ocm_colors = np.load(ocm_dir / "ocm_colors.npy")
    meta = json.loads((ocm_dir / "ocm_metadata.json").read_text(encoding="utf-8"))

    if len(points_view) != len(coords_rc) or len(points_view) != len(ocm_colors):
        raise ValueError("points_view, coords_rc, and ocm_colors must have the same length")

    H = int(meta["image_height"])
    W = int(meta["image_width"])
    target_void_ratio = (
        float(args.target_void_ratio)
        if args.target_void_ratio is not None
        else float(meta["config"]["target_void_ratio"])
    )
    max_fill_length = (
        int(args.max_fill_length)
        if args.max_fill_length is not None
        else int(meta["config"]["max_fill_length"])
    )

    depth = points_view[:, 1].astype(np.float64)
    depth_norm = robust_normalize(depth, args.low_q, args.high_q)
    if args.near_bright:
        depth_norm = 1.0 - depth_norm

    v_scale = args.v_min + depth_norm * (args.v_max - args.v_min)
    v_scale = np.clip(v_scale, 0.0, 1.0)

    depth_colors = np.clip(ocm_colors * v_scale[:, None], 0.0, 1.0)
    depth_img, fl, ratio = fill_ocm_image(
        coords_rc,
        depth_colors,
        None,
        H,
        W,
        target_void_ratio=target_void_ratio,
        max_fill_length=max_fill_length,
    )

    depth_gray = np.round(depth_norm * 255.0).astype(np.uint8)
    depth_aux_colors = np.repeat((depth_gray[:, None] / 255.0), 3, axis=1)
    depth_aux_img, _, _ = fill_ocm_image(
        coords_rc,
        depth_aux_colors,
        None,
        H,
        W,
        target_void_ratio=target_void_ratio,
        max_fill_length=max_fill_length,
    )

    out_path = ocm_dir / args.out_name
    aux_path = ocm_dir / args.aux_name
    meta_path = ocm_dir / "depth_v_metadata.json"
    save_image(str(out_path), depth_img)
    save_image(str(aux_path), depth_aux_img)

    out_meta = {
        "source_ocm_dir": str(ocm_dir),
        "out_image": str(out_path),
        "aux_image": str(aux_path),
        "fill_length_FL": int(fl),
        "void_ratio": float(ratio),
        "near_bright": bool(args.near_bright),
        "depth_percentiles": {
            "low_q": float(args.low_q),
            "high_q": float(args.high_q),
            "depth_at_low_q": float(np.percentile(depth, args.low_q)),
            "depth_at_high_q": float(np.percentile(depth, args.high_q)),
        },
        "v_range": {
            "v_min": float(args.v_min),
            "v_max": float(args.v_max),
        },
    }
    meta_path.write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved depth-aware OCM: {out_path}")
    print(f"saved depth auxiliary: {aux_path}")
    print(f"fill_length_FL={fl}, void_ratio={ratio:.6f}")


if __name__ == "__main__":
    main()
