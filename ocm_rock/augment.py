from pathlib import Path
import json
import random
import cv2
import numpy as np
from PIL import Image


def hue_shift_rgb(img: np.ndarray, shift: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + shift * 180.0) % 180.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def augment_labelme_dataset(src_root: str, dst_root: str, h_values=None, seed: int = 42) -> None:
    """按论文 Step 4.1.3：HSV Hue 变换 + 仿射 + 水平/垂直翻转。

    注意：这里为了保持 LabelMe polygon 的精确变换，采用 raster mask 方式生成增强后的 mask png；
    训练时如需使用增强数据，可进一步转 COCO，或直接改 Dataset 读取 mask png。
    """
    random.seed(seed)
    np.random.seed(seed)
    h_values = h_values or [i / 10 for i in range(10)]
    src = Path(src_root)
    dst = Path(dst_root)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    for img_path in sorted((src / "images").glob("*.png")):
        img = np.array(Image.open(img_path).convert("RGB"))
        ann = json.load(open(src / "annotations" / f"{img_path.stem}.json", "r", encoding="utf-8"))
        mask = labelme_to_instance_mask(ann, img.shape[:2])
        for h in h_values:
            imgh = hue_shift_rgb(img, h)
            angle = random.uniform(-90, 90)
            shear = random.uniform(-15, 15)
            tx = random.uniform(-img.shape[1] / 2, img.shape[1] / 2)
            ty = random.uniform(-img.shape[0] / 2, img.shape[0] / 2)
            aug_img, aug_mask = affine_pair(imgh, mask, angle, shear, tx, ty)
            if random.random() < 0.5:
                aug_img = np.flip(aug_img, axis=1).copy(); aug_mask = np.flip(aug_mask, axis=1).copy()
            if random.random() < 0.5:
                aug_img = np.flip(aug_img, axis=0).copy(); aug_mask = np.flip(aug_mask, axis=0).copy()
            name = f"{img_path.stem}_h{int(h*10):02d}_{random.randint(0,999999):06d}"
            Image.fromarray(aug_img).save(dst / "images" / f"{name}.png")
            Image.fromarray(aug_mask.astype(np.uint16)).save(dst / "masks" / f"{name}.png")


def labelme_to_instance_mask(ann, hw):
    from PIL import ImageDraw
    H, W = hw
    mask = np.zeros((H, W), dtype=np.uint16)
    idx = 1
    for shape in ann.get("shapes", []):
        pts = [tuple(p) for p in shape.get("points", [])]
        if len(pts) < 3:
            continue
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).polygon(pts, outline=1, fill=1)
        mask[np.array(m) > 0] = idx
        idx += 1
    return mask


def affine_pair(img, mask, angle, shear, tx, ty):
    H, W = img.shape[:2]
    center = (W / 2, H / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    Sh = np.array([[1, np.tan(np.deg2rad(shear)), 0], [0, 1, 0]], dtype=np.float32)
    M3 = np.vstack([M, [0, 0, 1]])
    Sh3 = np.vstack([Sh, [0, 0, 1]])
    A = (Sh3 @ M3)[:2]
    A[:, 2] += [tx, ty]
    img2 = cv2.warpAffine(img, A, (W, H), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    mask2 = cv2.warpAffine(mask, A, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    return img2, mask2
