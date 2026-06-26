import os
import cv2
import json
import torch
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


## 可套选

# =========================
# 1. 路径配置
# =========================
IMAGE_PATH = r"C:\jgmsb\segment-anything-main\sam_rock_demo_jgm\images\rockbench015nosk.jpg"
# IMAGE_PATH = r"C:\jgmsb\segment-anything-main\sam_rock_demo_jgm\images\DJI_0520.JPG"
CHECKPOINT_PATH = r"C:\jgmsb\segment-anything-main\sam_rock_demo_jgm\checkpoints\sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
OUTPUT_DIR = r"C:\jgmsb\segment-anything-main\sam_rock_demo_jgm\rockbenchResults"


# =========================
# 2. 全局变量
# =========================
foreground_points = []
background_points = []

box_prompt = None
box_temp = None
drawing_box = False
box_start = None

lasso_points = []
lasso_mask = None
drawing_lasso = False

current_mask = None
current_score = None

image_bgr = None
image_rgb = None
predictor = None
display_scale = 1.0

# point / box / lasso
interaction_mode = "point"


# =========================
# 3. 工具函数
# =========================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_next_save_index():
    """
    自动查找 results 文件夹中已经保存的结构面编号，
    返回下一个可用编号。
    例如已有 rock_J001_mask.png、rock_J002_mask.png，
    下一次返回 3。
    """
    ensure_dir(OUTPUT_DIR)

    image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    max_index = 0

    for filename in os.listdir(OUTPUT_DIR):
        if filename.startswith(f"{image_name}_J") and filename.endswith("_mask.png"):
            try:
                index_part = filename.replace(f"{image_name}_J", "").replace("_mask.png", "")
                index_num = int(index_part)
                max_index = max(max_index, index_num)
            except ValueError:
                pass

    return max_index + 1


def resize_for_display(img, max_width=1280, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def to_original_xy(x, y):
    ox = int(x / display_scale)
    oy = int(y / display_scale)

    h, w = image_bgr.shape[:2]
    ox = max(0, min(ox, w - 1))
    oy = max(0, min(oy, h - 1))

    return ox, oy


def build_point_prompts():
    points = []
    labels = []

    for p in foreground_points:
        points.append(p)
        labels.append(1)

    for p in background_points:
        points.append(p)
        labels.append(0)

    if len(points) == 0:
        return None, None

    return np.array(points), np.array(labels)


def get_lasso_box():
    """
    根据套索点生成外接框。
    SAM 本身不直接接收任意多边形作为 prompt，
    所以这里用多边形外接框作为 SAM 的 box prompt，
    再把 SAM 结果限制在套索范围内。
    """
    if len(lasso_points) < 3:
        return None

    pts = np.array(lasso_points)
    x1 = int(np.min(pts[:, 0]))
    y1 = int(np.min(pts[:, 1]))
    x2 = int(np.max(pts[:, 0]))
    y2 = int(np.max(pts[:, 1]))

    return [x1, y1, x2, y2]


def build_lasso_mask():
    """
    把套索点转为二值 mask。
    """
    if len(lasso_points) < 3:
        return None

    h, w = image_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    pts = np.array(lasso_points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 1)

    return mask.astype(bool)


def predict_mask():
    """
    根据当前点提示、框提示、套索范围调用 SAM 预测 mask。
    """
    global current_mask, current_score, lasso_mask

    point_coords, point_labels = build_point_prompts()

    final_box = None

    if interaction_mode == "lasso" and len(lasso_points) >= 3:
        final_box = get_lasso_box()
        lasso_mask = build_lasso_mask()
    elif box_prompt is not None:
        final_box = box_prompt

    if point_coords is None and final_box is None:
        current_mask = None
        current_score = None
        return

    masks, scores, logits = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        box=np.array(final_box) if final_box is not None else None,
        multimask_output=True
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]
    score = float(scores[best_idx])

    # 如果是套索模式，把 SAM 输出限制在你圈选的范围内
    if interaction_mode == "lasso" and lasso_mask is not None:
        mask = np.logical_and(mask, lasso_mask)

    current_mask = mask
    current_score = score


def draw_overlay():
    vis = image_bgr.copy()

    # 当前 mask 蓝色叠加
    if current_mask is not None:
        mask_uint8 = current_mask.astype(np.uint8)

        blue_layer = np.zeros_like(vis)
        blue_layer[:, :, 0] = 255

        alpha = 0.45
        vis = np.where(
            mask_uint8[:, :, None] == 1,
            (vis * (1 - alpha) + blue_layer * alpha).astype(np.uint8),
            vis
        )

        contours, _ = cv2.findContours(
            mask_uint8 * 255,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    # 前景点：绿色
    for x, y in foreground_points:
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.circle(vis, (x, y), 9, (255, 255, 255), 2)

    # 背景点：红色
    for x, y in background_points:
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.circle(vis, (x, y), 9, (255, 255, 255), 2)

    # 已确认框
    if box_prompt is not None:
        x1, y1, x2, y2 = box_prompt
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)

    # 正在拖动的框
    if box_temp is not None:
        x1, y1, x2, y2 = box_temp
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # 套索范围
    if len(lasso_points) >= 2:
        pts = np.array(lasso_points, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=False, color=(0, 165, 255), thickness=2)

        if not drawing_lasso and len(lasso_points) >= 3:
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 165, 255), thickness=3)

    mode_text = {
        "point": "POINT MODE",
        "box": "BOX MODE",
        "lasso": "LASSO MODE"
    }.get(interaction_mode, "POINT MODE")

    lines = [
        f"Mode: {mode_text}",
        "P: point mode | B: box mode | L: lasso mode",
        "Point mode: Left=foreground, Right=background",
        "Box mode: drag left mouse to draw box",
        "Lasso mode: hold left mouse and draw boundary",
        "S: save | R: reset current | Q: quit"
    ]

    if current_score is not None:
        lines.append(f"SAM score: {current_score:.4f}")

    y = 30
    for line in lines:
        cv2.putText(
            vis,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )
        y += 28

    display_img, _ = resize_for_display(vis)
    cv2.imshow("SAM Rock Discontinuity Segmentation", display_img)


# =========================
# 4. 鼠标交互
# =========================
def mouse_callback(event, x, y, flags, param):
    global drawing_box, box_start, box_temp, box_prompt
    global drawing_lasso, lasso_points, lasso_mask

    ox, oy = to_original_xy(x, y)

    # 点选模式
    if interaction_mode == "point":
        if event == cv2.EVENT_LBUTTONDOWN:
            foreground_points.append([ox, oy])
            print(f"添加前景点: ({ox}, {oy})")
            predict_mask()
            draw_overlay()

        elif event == cv2.EVENT_RBUTTONDOWN:
            background_points.append([ox, oy])
            print(f"添加背景点: ({ox}, {oy})")
            predict_mask()
            draw_overlay()

    # 框选模式
    elif interaction_mode == "box":
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_box = True
            box_start = (ox, oy)
            box_temp = None

        elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
            x1, y1 = box_start
            box_temp = [min(x1, ox), min(y1, oy), max(x1, ox), max(y1, oy)]
            draw_overlay()

        elif event == cv2.EVENT_LBUTTONUP and drawing_box:
            drawing_box = False

            x1, y1 = box_start
            box_prompt = [min(x1, ox), min(y1, oy), max(x1, ox), max(y1, oy)]
            box_temp = None

            print(f"添加框提示: {box_prompt}")
            predict_mask()
            draw_overlay()

        elif event == cv2.EVENT_RBUTTONDOWN:
            box_prompt = None
            box_temp = None
            print("已清除框提示。")
            predict_mask()
            draw_overlay()

    # 套索圈选模式
    elif interaction_mode == "lasso":
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_lasso = True
            lasso_points = [[ox, oy]]
            lasso_mask = None
            print("开始套索圈选。")

        elif event == cv2.EVENT_MOUSEMOVE and drawing_lasso:
            # 控制点密度，避免鼠标轻微移动产生太多点
            if len(lasso_points) == 0:
                lasso_points.append([ox, oy])
            else:
                last_x, last_y = lasso_points[-1]
                dist = ((ox - last_x) ** 2 + (oy - last_y) ** 2) ** 0.5
                if dist >= 5:
                    lasso_points.append([ox, oy])
                    draw_overlay()

        elif event == cv2.EVENT_LBUTTONUP and drawing_lasso:
            drawing_lasso = False

            if len(lasso_points) >= 3:
                print(f"套索圈选完成，共 {len(lasso_points)} 个边界点。")
                predict_mask()
            else:
                print("套索点太少，已忽略。")

            draw_overlay()

        elif event == cv2.EVENT_RBUTTONDOWN:
            lasso_points = []
            lasso_mask = None
            print("已清除套索范围。")
            predict_mask()
            draw_overlay()


# =========================
# 5. 保存与重置
# =========================
def save_result():
    if current_mask is None:
        print("当前没有 mask，无法保存。")
        return

    ensure_dir(OUTPUT_DIR)

    image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    save_index = get_next_save_index()
    discontinuity_id = f"J{save_index:03d}"

    file_prefix = f"{image_name}_{discontinuity_id}"

    mask_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_mask.png")
    overlay_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_overlay.png")
    points_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_prompt_info.json")

    mask_img = current_mask.astype(np.uint8) * 255
    cv2.imwrite(mask_path, mask_img)

    vis = image_bgr.copy()

    blue_layer = np.zeros_like(vis)
    blue_layer[:, :, 0] = 255
    alpha = 0.45

    vis = np.where(
        current_mask[:, :, None] == 1,
        (vis * (1 - alpha) + blue_layer * alpha).astype(np.uint8),
        vis
    )

    contours, _ = cv2.findContours(
        mask_img,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)

    for x, y in foreground_points:
        cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
        cv2.circle(vis, (x, y), 9, (255, 255, 255), 2)

    for x, y in background_points:
        cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
        cv2.circle(vis, (x, y), 9, (255, 255, 255), 2)

    if box_prompt is not None:
        x1, y1, x2, y2 = box_prompt
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)

    if len(lasso_points) >= 3:
        pts = np.array(lasso_points, dtype=np.int32)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 165, 255), thickness=3)

    cv2.imwrite(overlay_path, vis)

    prompt_info = {
        "image_path": IMAGE_PATH,
        "checkpoint_path": CHECKPOINT_PATH,
        "model_type": MODEL_TYPE,
        "discontinuity_id": discontinuity_id,
        "interaction_mode": interaction_mode,
        "foreground_points": foreground_points,
        "background_points": background_points,
        "box_prompt": box_prompt,
        "lasso_points": lasso_points,
        "sam_score": current_score,
        "mask_path": mask_path,
        "overlay_path": overlay_path
    }

    with open(points_path, "w", encoding="utf-8") as f:
        json.dump(prompt_info, f, ensure_ascii=False, indent=2)

    print("保存完成：")
    print(f"结构面编号：{discontinuity_id}")
    print(mask_path)
    print(overlay_path)
    print(points_path)


def reset_current():
    global foreground_points, background_points
    global box_prompt, box_temp, drawing_box, box_start
    global lasso_points, lasso_mask, drawing_lasso
    global current_mask, current_score

    foreground_points = []
    background_points = []

    box_prompt = None
    box_temp = None
    drawing_box = False
    box_start = None

    lasso_points = []
    lasso_mask = None
    drawing_lasso = False

    current_mask = None
    current_score = None

    print("已清空当前结构面的所有提示。")
    draw_overlay()


def set_mode(mode):
    global interaction_mode
    interaction_mode = mode
    print(f"当前模式切换为：{interaction_mode}")
    draw_overlay()


# =========================
# 6. 主程序
# =========================
def main():
    global image_bgr, image_rgb, predictor, display_scale

    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"找不到图片：{IMAGE_PATH}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"找不到模型权重：{CHECKPOINT_PATH}")

    print("正在读取图片...")
    image_bgr = cv2.imread(IMAGE_PATH)

    if image_bgr is None:
        raise ValueError("图片读取失败，请检查图片格式。")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _, display_scale = resize_for_display(image_bgr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前设备: {device}")

    print("正在加载 SAM ViT-H 模型...")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    print("正在提取图像 embedding，第一次会稍慢...")
    predictor.set_image(image_rgb)

    print("启动交互窗口。")
    print("P：点选模式。")
    print("B：框选模式。")
    print("L：套索/范围圈选模式。")
    print("S：保存当前结构面，自动编号。")
    print("R：清空当前结构面提示。")
    print("Q：退出。")

    cv2.namedWindow("SAM Rock Discontinuity Segmentation", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM Rock Discontinuity Segmentation", mouse_callback)

    draw_overlay()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == ord("p"):
            set_mode("point")

        elif key == ord("b"):
            set_mode("box")

        elif key == ord("l"):
            set_mode("lasso")

        elif key == ord("s"):
            save_result()

        elif key == ord("r"):
            reset_current()

        elif key == ord("q"):
            print("退出程序。")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()