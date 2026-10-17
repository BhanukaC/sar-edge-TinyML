# sar-edge/scripts/export_ssdlite_sard.py   run from sar-edge/: python scripts/export_ssdlite_sard.py
#
# Exports the SARD-trained SSDLite models (p00, p30, p50) to ONNX, network only (backbone +
# head, no torchvision post-processing), with the same settings as the smoke test: 320x320,
# opset 12 for TensorRT 8.2, classic exporter. Also saves the SARD anchors, which the boards
# need for decoding (common.post_ssd). Pruning removes channels, not grid positions, so all
# three models share the same anchors; the script checks that.
import numpy as np, torch, onnx, onnxsim
from torchvision.models.detection.image_list import ImageList

ANCHORS = "models/ssdlite-sard_anchors_320.npy"


class SSDRaw(torch.nn.Module):
    """Backbone + head only: raw box offsets and class logits, decoded later by post_ssd."""

    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head

    def forward(self, x):
        out = self.head(list(self.backbone(x).values()))
        return out["bbox_regression"], out["cls_logits"]


saved = None
for tag in ["p00", "p30", "p50"]:
    full = (
        torch.load("models/ssdlite-sard_%s.pt" % tag, weights_only=False).float().eval()
    )
    net = SSDRaw(full).eval()
    x = torch.zeros(1, 3, 320, 320)

    # the anchors come from the trained model itself (its SARD-sized anchor generator)
    with torch.no_grad():
        feats = list(full.backbone(x).values())
        anchors = full.anchor_generator(ImageList(x, [(320, 320)]), feats)[0].numpy()
    if saved is None:
        np.save(ANCHORS, anchors)
        saved = anchors
        print(
            "saved %s: %d anchors, smallest %.1f px wide"
            % (ANCHORS, len(anchors), (anchors[:, 2] - anchors[:, 0]).min())
        )
    else:
        assert np.allclose(anchors, saved), "%s has different anchors" % tag

    out = "models/ssdlite-sard_%s_fp32_320.onnx" % tag
    torch.onnx.export(
        net,
        x,
        out,
        opset_version=12,
        input_names=["images"],
        output_names=["bbox_regression", "cls_logits"],
        do_constant_folding=True,
        dynamo=False,
    )
    model, ok = onnxsim.simplify(onnx.load(out))  # same as running python -m onnxsim
    onnx.save(model, out)
    print("exported %s (simplified: %s)" % (out, ok))
