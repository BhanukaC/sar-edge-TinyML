# sar-edge/scripts/verify_yolo_sard.py   run from sar-edge/: python scripts/verify_yolo_sard.py
# Checks the three SARD-trained YOLO26n models (p00, p30, p50):
#   1. they load, and their parameter counts show the pruning (p30 < p00, p50 < p30)
#   2. their accuracy on the 197 test images (mAP50, mAP50-95, precision, recall)
#   3. pictures with detections on 6 test images, to check by eye
import os, sys, glob

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # scripts/, for yolo_prune_utils
from yolo_prune_utils import SplitC2f  # needed to load the pruned models
from ultralytics import YOLO

images = sorted(glob.glob("data/SARD/test/images/*.jpg"))[
    :6
]  # same 6 images for every model

print(
    "%-4s %12s %8s %9s %10s %7s"
    % ("", "parameters", "mAP50", "mAP50-95", "precision", "recall")
)
for tag in ["p00", "p30", "p50"]:
    model = YOLO("models/yolo26n-sard_%s.pt" % tag)
    n = sum(p.numel() for p in model.model.parameters())

    # accuracy on the test set (verbose=False keeps the output short)
    m = model.val(
        data="data/sard.yaml", split="test", imgsz=640, verbose=False, plots=False
    )

    # detections drawn on a few test images, saved to results/yolo_verify/<tag>/
    model.predict(
        images,
        imgsz=640,
        conf=0.25,
        save=True,
        verbose=False,
        project="results/yolo_verify",
        name=tag,
        exist_ok=True,
    )

    print(
        "%-4s %12d %8.3f %9.3f %10.3f %7.3f"
        % (tag, n, m.box.map50, m.box.map, m.box.mp, m.box.mr)
    )

print("pictures in results/yolo_verify/p00, p30, p50")
