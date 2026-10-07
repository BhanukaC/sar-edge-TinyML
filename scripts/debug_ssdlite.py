# sar-edge/scripts/debug_ssdlite.py   run from sar-edge/: python scripts/debug_ssdlite.py
# Compares our ONNX + post_ssd pipeline with torchvision's full SSDLite model.
import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root
import cv2, numpy as np, onnxruntime as ort, torch
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from ultralytics.utils import ASSETS
from common import prep_ssd, post_ssd, CONF, PERSON_TV

anchors = np.load("models/ssdlite_anchors_320.npy")
s = ort.InferenceSession(
    "models/ssdlite_p00_fp32_320.onnx", providers=["CPUExecutionProvider"]
)
print([(o.name, o.shape) for o in s.get_outputs()], "anchors:", anchors.shape)

full = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
).eval()

for path in [str(ASSETS / "bus.jpg"), "data/smoke/crowd.jpg"]:
    img = cv2.imread(path)

    # our pipeline: ONNX network + our own decoding and NMS
    x, scale = prep_ssd(img)
    reg, logits = s.run(None, {"images": x})
    ours = post_ssd(reg, logits, anchors, scale)

    # torchvision reference: full model with its own resize, normalisation and NMS
    t = (
        torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        / 255
    )
    with torch.no_grad():
        o = full([t])[0]
    keep = (o["labels"] == PERSON_TV) & (o["scores"] >= CONF)

    print(
        os.path.basename(path),
        "| ours:",
        len(ours),
        "people | torchvision:",
        int(keep.sum()),
        "people",
    )
    if len(ours):
        print("  our top box:", ours[ours[:, 4].argmax()].round(1))
    if keep.any():
        i = o["scores"][keep].argmax()
        print(
            "  tv top box: ",
            o["boxes"][keep][i].numpy().round(1),
            round(float(o["scores"][keep][i]), 3),
        )
