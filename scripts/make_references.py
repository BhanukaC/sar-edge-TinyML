# sar-edge/scripts/make_references.py   run from sar-edge/: python scripts/make_references.py
# Runs every FP32 model on data/smoke/ and saves, per model:
#   references/reference_<model>.json  person boxes [x1, y1, x2, y2, score] per image
#   references/raw_<model>.npz         raw output tensors per image (for pruned models)
import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root, so `import common` works
import glob, json
import cv2, numpy as np, onnxruntime as ort
import common as C

os.makedirs("references", exist_ok=True)
images = sorted(glob.glob("data/smoke/*.jpg"))
assert (
    images
), "no images in data/smoke/: run make_image_sets.py and pick_crowd.py first"


def run(model_path):
    s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = os.path.basename(model_path).replace(".onnx", "")
    anchors = np.load(C.ssd_anchor_file(name))      # SARD models use their own anchors
    boxes, raw = {}, {}
    for f in images:
        img = cv2.imread(f)
        key = os.path.basename(f)
        if name.startswith("ssdlite"):
            x, scale = C.prep_ssd(img)
            reg, logits = s.run(None, {"images": x})
            det = C.post_ssd(reg, logits, anchors, scale)
            raw[key + "_reg"], raw[key + "_cls"] = reg, logits
        else:
            x, r, pad = C.prep_yolo(img)
            out = s.run(None, {s.get_inputs()[0].name: x})[0]
            det = C.post_yolo_raw(out, r, pad)  # our export is [1, 84, 8400], see 1.5
            raw[key] = out
        boxes[key] = det.round(2).tolist()

    with open("references/reference_%s.json" % name, "w") as fh:
        json.dump(boxes, fh, indent=1)
    np.savez_compressed("references/raw_%s.npz" % name, **raw)

    # one-line summary: bus.jpg proves the pipeline for unpruned COCO models
    total = sum(len(v) for v in boxes.values())
    print(
        "%-28s bus.jpg: %s people | all %d images: %d boxes"
        % (name, len(boxes.get("bus.jpg", [])), len(images), total)
    )


for m in sorted(glob.glob("models/*_fp32_*.onnx")):
    run(m)
