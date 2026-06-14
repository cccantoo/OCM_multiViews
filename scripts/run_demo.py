"""无需真实数据的一键演示：合成点云 -> 生成 OCM 图像。"""
import subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
pc = root / "data/pointclouds/synthetic_rock.txt"
out = root / "outputs/demo_synthetic"
subprocess.check_call([sys.executable, str(root / "scripts/make_synthetic_demo.py"), "--out", str(pc)])
subprocess.check_call([sys.executable, str(root / "scripts/run_generate_ocm.py"), "--point_cloud", str(pc), "--out", str(out)])
print("Demo 完成：", out)
