import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ocm_rock.config import OCMConfig
from ocm_rock.io_utils import write_json
from ocm_rock.plane_analysis import approximate_trace_length, normal_to_orientation
from ocm_rock.preprocess import global_plane_normal
from ocm_rock.visualize import colorize_by_labels, plot_stereonet


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def plane_record(points: np.ndarray, ids: np.ndarray, plane_id: int) -> Dict:
    pts = points[ids]
    normal = global_plane_normal(pts)
    center = pts.mean(axis=0)
    signed = (pts - center) @ normal
    dd, da = normal_to_orientation(normal)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    return {
        "plane_id": int(plane_id),
        "num_points": int(len(ids)),
        "normal": [float(x) for x in normal],
        "center": [float(x) for x in center],
        "dip_direction_DD": float(dd),
        "dip_angle_DA": float(da),
        "trace_length": float(approximate_trace_length(pts)),
        "fit_residual_rmse": float(np.sqrt(np.mean(signed * signed))),
        "fit_residual_mean_abs": float(np.mean(np.abs(signed))),
        "bbox_min": [float(x) for x in bbox_min],
        "bbox_max": [float(x) for x in bbox_max],
        "extent": [float(x) for x in (bbox_max - bbox_min)],
    }


def write_records_csv(path: Path, records: List[Dict]) -> None:
    keys = [
        "plane_id",
        "num_points",
        "dip_direction_DD",
        "dip_angle_DA",
        "trace_length",
        "fit_residual_rmse",
        "fit_residual_mean_abs",
        "normal",
        "center",
        "bbox_min",
        "bbox_max",
        "extent",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            row = {k: record.get(k) for k in keys}
            for k, v in row.items():
                if isinstance(v, list):
                    row[k] = str(v)
            writer.writerow(row)


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.clip(abs(np.dot(a, b)), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def bbox_gap(a: Dict, b: Dict) -> float:
    amin = np.asarray(a["bbox_min"], dtype=float)
    amax = np.asarray(a["bbox_max"], dtype=float)
    bmin = np.asarray(b["bbox_min"], dtype=float)
    bmax = np.asarray(b["bbox_max"], dtype=float)
    gap = np.maximum(0.0, np.maximum(amin - bmax, bmin - amax))
    return float(np.linalg.norm(gap))


def pair_metrics(a: Dict, b: Dict, a_ids: np.ndarray, b_ids: np.ndarray) -> Dict:
    na = np.asarray(a["normal"], dtype=float)
    nb = np.asarray(b["normal"], dtype=float)
    ca = np.asarray(a["center"], dtype=float)
    cb = np.asarray(b["center"], dtype=float)
    intersection = int(np.intersect1d(a_ids, b_ids, assume_unique=True).size)
    min_count = max(1, min(len(a_ids), len(b_ids)))
    return {
        "angle_deg": angle_deg(na, nb),
        "center_dist": float(np.linalg.norm(ca - cb)),
        "plane_dist": float(max(abs(np.dot(na, cb - ca)), abs(np.dot(nb, ca - cb)))),
        "overlap_count": intersection,
        "overlap_ratio": float(intersection / min_count),
        "bbox_gap": bbox_gap(a, b),
    }


def is_match(metrics: Dict, args) -> bool:
    spatial_close = (
        metrics["center_dist"] <= args.center_thresh
        or metrics["overlap_ratio"] >= args.overlap_thresh
        or metrics["bbox_gap"] <= args.center_thresh
    )
    return (
        metrics["angle_deg"] <= args.angle_thresh
        and metrics["plane_dist"] <= args.plane_dist_thresh
        and spatial_close
    )


def load_inputs(input_dirs: List[Path]) -> Tuple[np.ndarray, List[np.ndarray]]:
    points = np.load(input_dirs[0] / "points.npy")
    labels_list = []
    for d in input_dirs:
        labels_path = d / "labels3d.npy"
        if not labels_path.exists():
            raise FileNotFoundError(f"missing labels3d.npy: {labels_path}")
        labels = np.load(labels_path)
        if labels.shape[0] != points.shape[0]:
            raise ValueError(f"labels shape mismatch: {d} {labels.shape} != ({points.shape[0]},)")
        labels_list.append(labels)

        points_path = d / "points.npy"
        if points_path.exists() and d != input_dirs[0]:
            other_points = np.load(points_path, mmap_mode="r")
            if other_points.shape != points.shape:
                raise ValueError(f"points shape mismatch: {d} {other_points.shape} != {points.shape}")
    return points, labels_list


def build_candidates(
    points: np.ndarray,
    labels_list: List[np.ndarray],
    input_dirs: List[Path],
    min_points: int,
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    candidates = []
    candidate_indices = {}
    for source_order, (d, labels) in enumerate(zip(input_dirs, labels_list)):
        for source_label in sorted(int(x) for x in np.unique(labels) if x > 0):
            ids = np.flatnonzero(labels == source_label)
            if len(ids) < min_points:
                continue
            candidate_id = len(candidates)
            key = f"candidate_{candidate_id:04d}"
            record = plane_record(points, ids, candidate_id)
            record.update(
                {
                    "candidate_id": candidate_id,
                    "source_dir": str(d),
                    "source_view": d.name,
                    "source_order": int(source_order),
                    "source_label": int(source_label),
                    "indices_key": key,
                }
            )
            candidates.append(record)
            candidate_indices[key] = ids.astype(np.int64)
    return candidates, candidate_indices


def match_candidates(
    candidates: List[Dict],
    candidate_indices: Dict[str, np.ndarray],
    args,
) -> Tuple[UnionFind, List[Dict]]:
    uf = UnionFind(len(candidates))
    matches = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if (
                not args.allow_same_view_match
                and candidates[i]["source_order"] == candidates[j]["source_order"]
            ):
                continue
            metrics = pair_metrics(
                candidates[i],
                candidates[j],
                candidate_indices[candidates[i]["indices_key"]],
                candidate_indices[candidates[j]["indices_key"]],
            )
            matched = is_match(metrics, args)
            if matched:
                uf.union(i, j)
            if matched or args.save_rejected_matches:
                matches.append(
                    {
                        "candidate_a": int(i),
                        "candidate_b": int(j),
                        "matched": bool(matched),
                        **metrics,
                    }
                )
    return uf, matches


def build_groups(
    points: np.ndarray,
    candidates: List[Dict],
    candidate_indices: Dict[str, np.ndarray],
    uf: UnionFind,
    min_points: int,
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    by_root: Dict[int, List[int]] = {}
    for i in range(len(candidates)):
        by_root.setdefault(uf.find(i), []).append(i)

    groups = []
    group_indices = {}
    for members in sorted(by_root.values(), key=lambda xs: min(xs)):
        ids = np.unique(np.concatenate([candidate_indices[candidates[i]["indices_key"]] for i in members]))
        if len(ids) < min_points:
            continue
        group_id = len(groups)
        key = f"group_{group_id:04d}"
        record = plane_record(points, ids, group_id + 1)
        source_views = sorted({candidates[i]["source_view"] for i in members})
        record.update(
            {
                "group_id": int(group_id),
                "output_label": int(group_id + 1),
                "indices_key": key,
                "candidate_ids": [int(i) for i in members],
                "source_views": source_views,
                "source_view_count": int(len(source_views)),
                "candidate_count": int(len(members)),
                "raw_union_points": int(len(ids)),
            }
        )
        groups.append(record)
        group_indices[key] = ids.astype(np.int64)
    return groups, group_indices


def assign_fused_labels(points: np.ndarray, groups: List[Dict], group_indices: Dict[str, np.ndarray]) -> np.ndarray:
    labels = np.zeros(len(points), dtype=np.int32)
    best_dist = np.full(len(points), np.inf, dtype=np.float64)
    best_view_count = np.zeros(len(points), dtype=np.int16)
    best_point_count = np.zeros(len(points), dtype=np.int32)
    eps = 1e-9

    for group in groups:
        label = int(group["output_label"])
        ids = group_indices[group["indices_key"]]
        normal = np.asarray(group["normal"], dtype=float)
        center = np.asarray(group["center"], dtype=float)
        dist = np.abs((points[ids] - center) @ normal)

        current_dist = best_dist[ids]
        current_views = best_view_count[ids]
        current_counts = best_point_count[ids]
        view_count = int(group["source_view_count"])
        point_count = int(group["raw_union_points"])
        update = (dist < current_dist - eps) | (
            np.abs(dist - current_dist) <= eps
        ) & (
            (view_count > current_views)
            | ((view_count == current_views) & (point_count > current_counts))
        )

        update_ids = ids[update]
        labels[update_ids] = label
        best_dist[update_ids] = dist[update]
        best_view_count[update_ids] = view_count
        best_point_count[update_ids] = point_count

    return labels


def remap_and_analyze(points: np.ndarray, labels: np.ndarray, min_points: int) -> Tuple[np.ndarray, List[Dict], Dict[int, int]]:
    remapped = np.zeros_like(labels, dtype=np.int32)
    planes = []
    label_map = {}
    next_label = 1
    for old_label in sorted(int(x) for x in np.unique(labels) if x > 0):
        ids = np.flatnonzero(labels == old_label)
        if len(ids) < min_points:
            continue
        label_map[int(old_label)] = int(next_label)
        remapped[ids] = next_label
        planes.append(plane_record(points, ids, next_label))
        next_label += 1
    return remapped, planes, label_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Geometrically fuse labels3d.npy instances from multiple OCM inference views."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="Inference dirs containing points.npy and labels3d.npy")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--angle_thresh", type=float, default=10.0, help="Maximum acute normal angle in degrees")
    parser.add_argument("--center_thresh", type=float, default=3.0, help="Maximum center distance, in point-cloud units")
    parser.add_argument("--plane_dist_thresh", type=float, default=0.5, help="Maximum symmetric point-to-plane distance")
    parser.add_argument("--overlap_thresh", type=float, default=0.05, help="Minimum shared point ratio over the smaller candidate")
    parser.add_argument("--min_points", type=int, default=None, help="Minimum points for candidate and fused planes")
    parser.add_argument("--min_plane_points", type=int, default=None, help="Alias for --min_points")
    parser.add_argument("--allow_same_view_match", action="store_true", help="Allow merging candidates from the same view")
    parser.add_argument(
        "--save_rejected_matches",
        action="store_true",
        help="Save non-matching pair metrics too. This can make fusion_matches.json larger.",
    )
    parser.add_argument("--skip_ply", action="store_true", help="Skip fused_colored_planes.ply")
    parser.add_argument("--skip_stereonet", action="store_true", help="Skip fused_stereonet.png")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OCMConfig()
    min_points = args.min_points or args.min_plane_points or cfg.min_plane_points
    input_dirs = [Path(p) for p in args.inputs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    points, labels_list = load_inputs(input_dirs)
    candidates, candidate_indices = build_candidates(points, labels_list, input_dirs, min_points)
    if not candidates:
        raise RuntimeError("no valid plane candidates found")

    uf, matches = match_candidates(candidates, candidate_indices, args)
    groups, group_indices = build_groups(points, candidates, candidate_indices, uf, min_points)
    fused_raw = assign_fused_labels(points, groups, group_indices)
    fused, planes, label_map = remap_and_analyze(points, fused_raw, min_points)

    for group in groups:
        old_label = int(group["output_label"])
        group["final_output_label"] = label_map.get(old_label)
        group["assigned_points_after_conflict_resolution"] = int((fused_raw == old_label).sum())

    per_view = []
    for source_order, (d, labels) in enumerate(zip(input_dirs, labels_list)):
        per_view.append(
            {
                "source_dir": str(d),
                "source_order": int(source_order),
                "source_instances": int(len([x for x in np.unique(labels) if x > 0])),
                "source_labeled_points": int((labels > 0).sum()),
            }
        )

    summary = {
        "inputs": [str(d) for d in input_dirs],
        "parameters": {
            "angle_thresh": float(args.angle_thresh),
            "center_thresh": float(args.center_thresh),
            "plane_dist_thresh": float(args.plane_dist_thresh),
            "overlap_thresh": float(args.overlap_thresh),
            "min_points": int(min_points),
            "allow_same_view_match": bool(args.allow_same_view_match),
        },
        "per_view": per_view,
        "candidate_total": int(len(candidates)),
        "matched_edges": int(sum(1 for m in matches if m["matched"])),
        "fusion_groups_before_final_filter": int(len(groups)),
        "fused_planes": int(len(planes)),
        "labeled_points_before_union": int(sum((labels > 0).sum() for labels in labels_list)),
        "unique_labeled_points_across_inputs": int(np.logical_or.reduce([labels > 0 for labels in labels_list]).sum()),
        "fused_labeled_points": int((fused > 0).sum()),
    }

    np.save(out / "points.npy", points)
    np.save(out / "fused_labels3d.npy", fused)
    np.savez_compressed(out / "fusion_candidate_indices.npz", **candidate_indices)
    np.savez_compressed(out / "fusion_group_indices.npz", **group_indices)
    write_json(str(out / "fusion_candidates.json"), candidates)
    write_json(str(out / "fusion_groups.json"), groups)
    write_json(str(out / "fusion_matches.json"), matches)
    write_json(str(out / "fusion_summary.json"), summary)
    write_json(str(out / "fused_planes.json"), planes)
    write_records_csv(out / "fused_planes.csv", planes)

    if not args.skip_ply:
        colorize_by_labels(points, fused, str(out / "fused_colored_planes.ply"))
    if not args.skip_stereonet:
        plot_stereonet(planes, str(out / "fused_stereonet.png"))

    print(
        "fusion completed: "
        f"candidates={len(candidates)}, matched_edges={summary['matched_edges']}, "
        f"groups={len(groups)}, fused_planes={len(planes)}, "
        f"labeled_points={summary['fused_labeled_points']}, out={out}"
    )


if __name__ == "__main__":
    main()
