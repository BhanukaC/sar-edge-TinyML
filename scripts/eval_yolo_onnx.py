# sar-edge/scripts/eval_yolo_onnx.py   run from sar-edge/: python scripts/eval_yolo_onnx.py
# Measures mAP on the 197 SARD test images for every YOLO ONNX file (FP32 and INT8),
# using ONNX Runtime, the same runtime as on the Pi.
from ultralytics import YOLO

for tag in ["p00", "p30", "p50"]:
    for prec in ["fp32", "int8"]:
        f = "models/yolo26n-sard_%s_%s_640.onnx" % (tag, prec)
        m = YOLO(f, task="detect").val(
            data="data/sard.yaml", split="test", imgsz=640, batch=1, verbose=False
        )
        print(
            "%s %s  mAP50 %.3f  mAP50-95 %.3f  precision %.3f  recall %.3f"
            % (tag, prec, m.box.map50, m.box.map, m.box.mp, m.box.mr)
        )
