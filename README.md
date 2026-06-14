# OCM 岩体结构面识别复刻工程

本工程按论文 **OCM: an intelligent recognition method of rock discontinuity based on optimal color mapping of 3D point cloud via deep learning** 的 5 步流程实现：
1. PCA 法向量估计 + 半球化 + sharp point 检测 + NPW-OC 骨架收缩；
2. 法向量最优色彩映射 OCM；
3. OCM 图像方向校准、尺寸归一、空洞填充；
4. LabelMe 标注数据转 Mask R-CNN 训练样本，并支持 HSV/仿射/翻转增强；
5. Mask R-CNN 识别 2D mask，再反投影到 3D 点云，输出结构面姿态、极射赤平投影、3D 结果。

> 说明：论文没有公开训练好的 Mask R-CNN 权重和完整 43 个标注数据，因此本工程提供严格按论文公式实现的可训练/可推理代码。要复现论文数值结果，需要准备与论文一致或相近的数据集与 LabelMe 标注。

## 快速开始

```bash
conda create -n ocm_rock python=3.10 -y
conda activate ocm_rock
pip install -r requirements.txt
```

### 1. 用本地点云生成 OCM 图像

```bash
python scripts/run_generate_ocm.py --point_cloud data/pointclouds/your_cloud.txt --out outputs/case01
```

输入点云支持：`.txt/.xyz/.csv` 的 `x y z [r g b]` 或 Open3D 支持的 `.ply/.pcd/.xyz`。

### 2. LabelMe 标注

```bash
labelme outputs/case01/ocm_image.png
```

每个结构面用 polygon 框出，label 可统一写 `discontinuity`，也可以写 `plane_001`。

### 3. 训练 Mask R-CNN

将图片与 json 放入如下结构：

```text
data/ocm_dataset/
  train/images/*.png
  train/annotations/*.json
  val/images/*.png
  val/annotations/*.json
```

运行：

```bash
python scripts/train_maskrcnn.py --dataset data/ocm_dataset --out outputs/train_run --epochs 260 --batch_size 4 --lr 1e-5
```

### 4. 推理并反投影到三维

```bash
python scripts/run_infer.py --point_cloud data/pointclouds/your_cloud.txt --weights outputs/train_run/model_last.pth --out outputs/infer_case01
```

## 主要输出

- `ocm_image.png`：论文 Step 3 的 OCM 图像；
- `sharp_skeleton.npy`：NPW-OC 后的交线骨架点；
- `mask_pred.png`：Mask R-CNN 实例分割结果；
- `planes.json/csv`：每个结构面的点数、法向量、倾向 DD、倾角 DA、三维迹长等；
- `stereonet.png`：极射赤平投影；
- `colored_planes.ply`：按识别结构面着色后的点云。
