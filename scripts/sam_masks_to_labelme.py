import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Convert one-mask-per-instance SAM PNGs to one LabelMe JSON file")
    parser.add_argument("--image", required=True, help="Source image that the masks were created from")
    parser.add_argument("--mask_dir", required=True, help="Directory containing *_mask.png files")
    parser.add_argument("--out", required=True, help="Output LabelMe JSON path")
    parser.add_argument("--label", default="discontinuity")
    parser.add_argument("--pattern", default="*_mask.png")
    parser.add_argument("--min_area", type=float, default=50.0)
    parser.add_argument("--epsilon_ratio", type=float, default=0.002, help="Polygon simplification ratio of contour perimeter")
    parser.add_argument("--embed_image", action="store_true", help="Embed base64 imageData in JSON")
    args = parser.parse_args()

    image_path = Path(args.image)
    mask_dir = Path(args.mask_dir)
    out_path = Path(args.out)

    with Image.open(image_path) as img:
        width, height = img.size

    shapes = []
    skipped = []
    for mask_path in sorted(mask_dir.glob(args.pattern)):
        mask = np.array(Image.open(mask_path).convert("L"))
        if mask.shape[:2] != (height, width):
            skipped.append({"mask": str(mask_path), "reason": f"size {mask.shape[1]}x{mask.shape[0]} != image {width}x{height}"})
            continue
        binary = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= args.min_area]
        if not contours:
            skipped.append({"mask": str(mask_path), "reason": "no contour above min_area"})
            continue

        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, args.epsilon_ratio * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            skipped.append({"mask": str(mask_path), "reason": "polygon has fewer than 3 points"})
            continue

        points = [[float(x), float(y)] for x, y in approx]
        shapes.append({
            "label": args.label,
            "points": points,
            "group_id": None,
            "description": mask_path.stem,
            "shape_type": "polygon",
            "flags": {},
            "mask": None,
        })

    image_data = None
    if args.embed_image:
        image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    data = {
        "version": "6.2.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": image_data,
        "imageHeight": height,
        "imageWidth": width,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"image_size={width}x{height}, masks={len(list(mask_dir.glob(args.pattern)))}, shapes={len(shapes)}, skipped={len(skipped)}")
    for item in skipped[:20]:
        print("skipped:", item)


if __name__ == "__main__":
    main()
