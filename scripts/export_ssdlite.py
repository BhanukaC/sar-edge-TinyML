# sar-edge/scripts/export_ssdlite.py   run from sar-edge/: python scripts/export_ssdlite.py
# Exports SSDLite backbone + head (no postprocessing) and saves the anchors.
import torch, numpy as np
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.image_list import ImageList

full = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
).eval()


class SSDRaw(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head

    def forward(self, x):
        feats = list(self.backbone(x).values())
        out = self.head(feats)
        return out["bbox_regression"], out["cls_logits"]


net = SSDRaw(full).eval()
x = torch.zeros(1, 3, 320, 320)

# anchors depend only on input size and feature map sizes, so save them once
with torch.no_grad():
    feats = list(full.backbone(x).values())
    anchors = full.anchor_generator(ImageList(x, [(320, 320)]), feats)[0]
np.save("models/ssdlite_anchors_320.npy", anchors.numpy())

torch.onnx.export(
    net,
    x,
    "models/ssdlite_p00_fp32_320.onnx",
    opset_version=12,
    input_names=["images"],
    output_names=["bbox_regression", "cls_logits"],
    do_constant_folding=True,
    dynamo=False,
)  # classic exporter: supports opset 12 for TensorRT 8.2
