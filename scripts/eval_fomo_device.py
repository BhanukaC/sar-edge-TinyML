# sar-edge/scripts/eval_fomo_device.py   run on the Pi (venv ~/sar) or Jetson (venv ~/sar38)
# from the repo folder: python scripts/eval_fomo_device.py [pattern]
# Runs each FOMO .eim on the SARD test images and saves its detections as centres in
# original-image pixels: results/<board>/fomo_test/<model>.json
import os, sys, glob, json
import cv2
from edge_impulse_linux.image import ImageImpulseRunner

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "models/fomo-*-sard_p00_*_320.eim"
ON_JETSON = os.path.exists("/etc/nv_tegra_release")
BOARD = "jetson" if ON_JETSON else "pi"
IMAGES = sorted(glob.glob("data/SARD/test/images/*.jpg"))
INPUT = 320  # FOMO input size (Fit longest axis)
OUT = os.path.join("results", BOARD, "fomo_test")
os.makedirs(OUT, exist_ok=True)
assert IMAGES, "no test images in data/SARD/test/images: git pull first"


def features_for(runner, img_rgb):
    try:
        return runner.get_features_from_image_auto_studio_settings(img_rgb)[0]
    except AttributeError:
        return runner.get_features_from_image(img_rgb)[0]


for path in sorted(glob.glob(PATTERN)):
    if ("_jetson_" in path) != ON_JETSON:  # each board only runs its own builds
        continue
    name = os.path.basename(path)[:-4]
    runner = ImageImpulseRunner(os.path.abspath(path))
    results = {}
    try:
        runner.init()
        for f in IMAGES:
            bgr = cv2.imread(f)
            h, w = bgr.shape[:2]
            feats = features_for(runner, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            boxes = runner.classify(feats)["result"].get("bounding_boxes", [])
            # map centres from the 320x320 input back to the original frame
            # (Fit longest axis: scale the longer side to 320, image centred, padding split evenly)
            s = INPUT / float(max(w, h))
            pad_x, pad_y = (INPUT - w * s) / 2.0, (INPUT - h * s) / 2.0
            dets = []
            for b in boxes:
                cx = b["x"] + b["width"] / 2.0
                cy = b["y"] + b["height"] / 2.0
                dets.append(
                    [
                        round((cx - pad_x) / s, 1),
                        round((cy - pad_y) / s, 1),
                        round(b["value"], 3),
                    ]
                )
            results[os.path.basename(f)] = dets
    finally:
        runner.stop()
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(results, fh)
    print(
        "%-32s %d images, %d detections"
        % (name, len(results), sum(len(v) for v in results.values()))
    )
