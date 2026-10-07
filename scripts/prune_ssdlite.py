# sar-edge/scripts/prune_ssdlite.py   run from sar-edge/: python scripts/prune_ssdlite.py
# Rebuilds the same network as export_ssdlite.py, then makes the p30 and p50 versions.
import copy, torch, torch_pruning as tp
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)


class SSDRaw(torch.nn.Module):  # same wrapper as in export_ssdlite.py
    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head

    def forward(self, x):
        out = self.head(list(self.backbone(x).values()))
        return out["bbox_regression"], out["cls_logits"]


net = SSDRaw(
    ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    ).eval()
).eval()


example = torch.randn(1, 3, 320, 320)
for ratio, tag in [(0.3, "p30"), (0.5, "p50")]:
    m = copy.deepcopy(net)
    base_macs, base_params = tp.utils.count_ops_and_params(m, example)
    pruner = tp.pruner.MetaPruner(
        m,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=[m.head],  # keep output channels = anchors x classes
        round_to=8,
    )  # channel counts in multiples of 8 suit TensorRT
    pruner.step()
    macs, params = tp.utils.count_ops_and_params(m, example)
    print(tag, params / base_params, macs / base_macs)
    torch.onnx.export(
        m.eval(),
        example,
        "models/ssdlite_%s_fp32_320.onnx" % tag,
        opset_version=12,
        input_names=["images"],
        output_names=["bbox_regression", "cls_logits"],
        dynamo=False,
    )  # classic exporter, opset 12
