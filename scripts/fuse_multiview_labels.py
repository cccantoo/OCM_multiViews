import argparse
from pathlib import Path

import numpy as np

from ocm_rock.config import OCMConfig
from ocm_rock.io_utils import write_json
from ocm_rock.plane_analysis import analyze_planes, write_planes_csv
from ocm_rock.visualize import colorize_by_labels, plot_stereonet


def main():
    parser = argparse.ArgumentParser(description="Quickly fuse labels3d.npy from multiple OCM inference views")
    parser.add_argument("--inputs", nargs="+", required=True, help="Inference directories containing points.npy and labels3d.npy")
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_plane_points", type=int, default=None)
    args = parser.parse_args()

    cfg = OCMConfig()
    min_points = args.min_plane_points or cfg.min_plane_points
    input_dirs = [Path(p) for p in args.inputs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points = np.load(input_dirs[0] / "points.npy")
    fused = np.zeros(len(points), dtype=np.int32)
    source_map = {}
    summary = []

    for view_idx, d in enumerate(input_dirs):
        labels = np.load(d / "labels3d.npy")
        if labels.shape != fused.shape:
            raise ValueError(f"labels shape mismatch: {d} {labels.shape} != {fused.shape}")
        current_labeled = int((fused > 0).sum())
        added_points = 0
        added_instances = 0

        for lab in sorted(int(x) for x in np.unique(labels) if x > 0):
            add_mask = (fused == 0) & (labels == lab)
            count = int(add_mask.sum())
            if count < min_points:
                continue
            new_id = int(fused.max()) + 1
            fused[add_mask] = new_id
            source_map[str(new_id)] = {
                "source_dir": str(d),
                "source_order": view_idx,
                "source_label": lab,
                "added_points": count,
            }
            added_points += count
            added_instances += 1

        summary.append({
            "source_dir": str(d),
            "source_order": view_idx,
            "source_instances": int(labels.max()),
            "source_labeled_points": int((labels > 0).sum()),
            "fused_points_before": current_labeled,
            "added_instances": added_instances,
            "added_points": added_points,
            "fused_points_after": int((fused > 0).sum()),
        })

    np.save(out / "points.npy", points)
    np.save(out / "fused_labels3d.npy", fused)
    planes = analyze_planes(points, fused, min_points)
    write_json(str(out / "fused_planes.json"), planes)
    write_planes_csv(str(out / "fused_planes.csv"), planes)
    write_json(str(out / "fused_source_map.json"), source_map)
    write_json(str(out / "fused_summary.json"), summary)
    colorize_by_labels(points, fused, str(out / "fused_colored_planes.ply"))
    plot_stereonet(planes, str(out / "fused_stereonet.png"))

    print(f"fused instances={int(fused.max())}, planes={len(planes)}, labeled_points={int((fused > 0).sum())}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
