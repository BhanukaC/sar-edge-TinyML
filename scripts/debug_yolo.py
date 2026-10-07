# sar-edge/scripts/debug_yolo.py   run from sar-edge/: python scripts/debug_yolo.py
# Compares our ONNX pipeline with Ultralytics' own prediction on two images.
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import ASSETS
from common import prep_yolo, post_yolo_raw, PERSON_YOLO

s = ort.InferenceSession(
    "models/yolo26n_p00_fp32_640.onnx", providers=["CPUExecutionProvider"]
)
ref = YOLO("yolo26n.pt")

for path in [str(ASSETS / "bus.jpg"), "data/smoke/crowd.jpg"]:
    img = cv2.imread(path)
    x, r, pad = prep_yolo(img)
    out = s.run(None, {"images": x})[0]
    ours = post_yolo_raw(out, r, pad)
    theirs = ref.predict(path, conf=0.25, classes=[0], verbose=False)[0].boxes
    print(os.path.basename(path), img.shape[:2])
    print(
        "  max person score in ONNX output:",
        round(float(out[0, 4 + PERSON_YOLO].max()), 3),
    )
    print("  our pipeline:", len(ours), "people | Ultralytics:", len(theirs), "people")
