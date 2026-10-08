# sar-edge/scripts/jetson_check.py   run on the Jetson from the repo folder: python3 scripts/jetson_check.py
# Python 3.6 compatible. For each engine, runs bus.jpg and crowd.jpg through trtexec
# (input from a file, output to JSON) and compares with the laptop FP32 references.
import os, sys, glob, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import cv2
import numpy as np
import common as C

TRT = "/usr/src/tensorrt/bin/trtexec"
anchors = np.load("models/ssdlite_anchors_320.npy")
os.makedirs("tmp", exist_ok=True)


def run_engine(engine, x):
    """Run one input through the engine with trtexec; return {output name: array}."""
    x.astype(np.float32).tofile("tmp/in.bin")
    subprocess.run([TRT, "--loadEngine=" + engine, "--loadInputs=images:tmp/in.bin",
                    "--exportOutput=tmp/out.json", "--iterations=1", "--warmUp=0",
                    "--duration=0"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with open("tmp/out.json") as fh:
        outs = json.load(fh)
    return {o["name"]: np.array(o["values"], dtype=np.float32).reshape(
        [int(d) for d in o["dimensions"].split("x")]) for o in outs}


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


for engine in sorted(glob.glob("engines/*.engine")):
    name = os.path.basename(engine)[:-7]
    ref = name.replace("_fp16_", "_fp32_")
    raw = np.load("references/raw_%s.npz" % ref)
    with open("references/reference_%s.json" % ref) as fh:
        ref_boxes = json.load(fh)
    parts = []
    for img_name in ["bus.jpg", "crowd.jpg"]:
        img = cv2.imread("data/smoke/" + img_name)
        if name.startswith("ssdlite"):
            x, scale = C.prep_ssd(img)
            o = run_engine(engine, x)
            det = C.post_ssd(o["bbox_regression"], o["cls_logits"], anchors, scale)
            cos = cosine(o["cls_logits"].ravel(), raw[img_name + "_cls"].ravel())
        else:
            x, r, pad = C.prep_yolo(img)
            out = list(run_engine(engine, x).values())[0]
            det = C.post_yolo_raw(out, r, pad)
            cos = cosine(out[0, 4:].ravel(), raw[img_name][0, 4:].ravel())
        parts.append("%s %d/%d cos %.4f" % (img_name, len(det), len(ref_boxes[img_name]), cos))
    print("%-24s %s" % (name, " | ".join(parts)))
