import torch
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def build_maskrcnn(
    num_classes: int = 2,
    score_thresh: float = 0.05,
    nms_thresh: float = 0.7,
    detections_per_img: int = 512,
    pretrained: bool = True,
    min_size: int = 800,
    max_size: int = 1333,
):
    """构建论文使用的 Mask R-CNN 实例分割模型。

    num_classes=2：背景 + discontinuity。
    """
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=None,
        min_size=min_size,
        max_size=max_size,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = nms_thresh
    model.roi_heads.detections_per_img = detections_per_img
    return model


def load_model(weights_path: str, device: str = None, **kwargs):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_maskrcnn(pretrained=False, **kwargs)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device
