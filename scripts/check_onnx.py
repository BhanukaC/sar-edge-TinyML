# sar-edge/scripts/check_onnx.py   run from sar-edge/: python scripts/check_onnx.py
# Checks an exported ONNX file: valid graph, input/output shapes, one inference.
import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root, so `import common` works
import onnx, onnxruntime as ort, cv2
from common import prep_yolo

onnx.checker.check_model("models/yolo26n_p00_fp32_640.onnx")
s = ort.InferenceSession(
    "models/yolo26n_p00_fp32_640.onnx", providers=["CPUExecutionProvider"]
)
print([(i.name, i.shape) for i in s.get_inputs()])
print([(o.name, o.shape) for o in s.get_outputs()])
img = cv2.imread("data/smoke/crowd.jpg")
assert img is not None, "data/smoke/crowd.jpg missing: run scripts/pick_crowd.py first"
x, r, pad = prep_yolo(img)
print(s.run(None, {s.get_inputs()[0].name: x})[0].shape)


# decode to person boxes and save a picture to look at
from common import post_yolo_raw

out = s.run(None, {s.get_inputs()[0].name: x})[0]
det = post_yolo_raw(out, r, pad)
print(len(det), "people found")
for x1, y1, x2, y2, score in det:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
cv2.imwrite("results/check_crowd_yolo.jpg", img)
