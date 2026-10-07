# sar-edge/scripts/compare_int8.py   run from sar-edge/: python scripts/compare_int8.py
# Compares each INT8 model with its FP32 version on data/smoke/: people found on
# bus.jpg, and how similar the raw outputs are (cosine, 1.0 = identical).
# Boxes and scores are compared separately: box values are large (pixels), so a
# combined cosine can look perfect while every score is broken.
import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root
import glob
import cv2, numpy as np, onnxruntime as ort
import common as C

anchors = np.load("models/ssdlite_anchors_320.npy")
images = sorted(glob.glob("data/smoke/*.jpg"))


def outputs(path):
    """Return {image: (person boxes, raw box values, raw score values)} for one model."""
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    res = {}
    for f in images:
        img = cv2.imread(f)
        key = os.path.basename(f)
        if "ssdlite" in path:
            x, scale = C.prep_ssd(img)
            reg, logits = s.run(None, {"images": x})
            res[key] = (
                C.post_ssd(reg, logits, anchors, scale),
                reg.ravel(),
                logits.ravel(),
            )
        else:
            x, r, pad = C.prep_yolo(img)
            out = s.run(None, {s.get_inputs()[0].name: x})[0]
            res[key] = (
                C.post_yolo_raw(out, r, pad),
                out[0, :4].ravel(),
                out[0, 4:].ravel(),
            )
    return res


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


for q in sorted(glob.glob("models/*_int8_*.onnx")):
    f = q.replace("_int8_", "_fp32_")
    a, b = outputs(f), outputs(q)
    cos_box = np.mean([cosine(a[k][1], b[k][1]) for k in a])
    cos_score = np.mean([cosine(a[k][2], b[k][2]) for k in a])
    print(
        "%-22s bus.jpg: fp32 %d -> int8 %d people | cosine boxes %.4f, scores %.4f"
        % (
            os.path.basename(q)[:-5],
            len(a["bus.jpg"][0]),
            len(b["bus.jpg"][0]),
            cos_box,
            cos_score,
        )
    )
