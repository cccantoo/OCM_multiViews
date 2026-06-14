"""生成一个合成岩体点云，便于无真实数据时测试 OCM 图像生成是否可运行。"""
from pathlib import Path
import argparse
import numpy as np


def sample_plane(normal, center, width=5.0, height=3.0, n=8000, noise=0.015):
    normal = np.array(normal, dtype=float); normal /= np.linalg.norm(normal)
    tmp = np.array([0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(normal, tmp); u /= np.linalg.norm(u)
    v = np.cross(normal, u); v /= np.linalg.norm(v)
    a = (np.random.random(n)-0.5)*width
    b = (np.random.random(n)-0.5)*height
    pts = np.array(center) + a[:,None]*u + b[:,None]*v + np.random.normal(scale=noise, size=(n,3))
    return pts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/pointclouds/synthetic_rock.txt")
    args = parser.parse_args()
    np.random.seed(0)
    planes = [
        ([0.2, -0.8, 0.55], [0, 0, 0], 5, 3, 9000),
        ([-0.6, -0.5, 0.62], [1.5, 0.2, 0.4], 4, 2.5, 7000),
        ([0.75, -0.25, 0.62], [-1.8, -0.1, -0.2], 3.5, 2.2, 6500),
        ([0.1, -0.95, 0.28], [0, 0.1, 1.2], 5, 1.4, 4000),
    ]
    pts = np.vstack([sample_plane(*p) for p in planes])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, pts, fmt="%.6f")
    print(args.out)


if __name__ == "__main__":
    main()
