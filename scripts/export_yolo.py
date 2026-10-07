# sar-edge/scripts/export_yolo.py   run from sar-edge/: python scripts/export_yolo.py
# Exports COCO-pretrained YOLO26n to models/yolo26n_p00_fp32_640.onnx
from ultralytics import YOLO
import shutil

m = YOLO("yolo26n.pt")  # COCO-pretrained, downloads automatically
path = m.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True,
    dynamic=False,
    batch=1,
    half=False,
)
shutil.move(path, "models/yolo26n_p00_fp32_640.onnx")
