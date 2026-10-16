# sar-edge/scripts/export_yolo_sard.py   run from sar-edge/: python scripts/export_yolo_sard.py
# Exports the three SARD-trained YOLO26n models (p00, p30, p50) to ONNX for the boards,
# with the same settings as the smoke test (640x640, opset 12 for TensorRT 8.2, batch 1).
import os, sys, shutil

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # scripts/, for yolo_prune_utils
from yolo_prune_utils import SplitC2f  # needed to load the pruned models (p30, p50)
from ultralytics import YOLO

for tag in ["p00", "p30", "p50"]:
    path = YOLO("models/yolo26n-sard_%s.pt" % tag).export(
        format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False, batch=1
    )
    out = "models/yolo26n-sard_%s_fp32_640.onnx" % tag
    shutil.move(path, out)
    print("exported", out)
