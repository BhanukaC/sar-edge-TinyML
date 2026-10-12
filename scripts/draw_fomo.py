# sar-edge/scripts/draw_fomo.py   run on the Pi (venv ~/sar) or Jetson (venv ~/sar38)
# from the repo folder: python scripts/draw_fomo.py [pattern]
# Runs each FOMO .eim on the smoke images and saves pictures with the detections drawn.
# FOMO finds object centres on an 8x8-pixel grid, so each detection is drawn as a dot
# (the centre) inside its grid cell, with its score. Drawn on the 320x320 image FOMO saw.
import os, sys, glob
import cv2
from edge_impulse_linux.image import ImageImpulseRunner

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "models/fomo-*_p00_*_320.eim"
ON_JETSON = os.path.exists("/etc/nv_tegra_release")
BOARD = "jetson" if ON_JETSON else "pi"
IMAGES = sorted(glob.glob("data/smoke/*.jpg"))
SCALE = 2  # show the 320x320 input at 640x640


def features_for(runner, img_rgb):
    # same resize mode as set in Studio (Fit longest axis); older SDKs lack this
    try:
        return runner.get_features_from_image_auto_studio_settings(img_rgb)
    except AttributeError:
        return runner.get_features_from_image(img_rgb)


for path in sorted(glob.glob(PATTERN)):
    is_jetson_build = "_jetson_" in path
    if is_jetson_build != ON_JETSON:  # each board only runs its own builds
        continue
    name = os.path.basename(path)[:-4]
    out_dir = os.path.join("results", BOARD, "fomo_vis", name)
    os.makedirs(out_dir, exist_ok=True)
    runner = ImageImpulseRunner(os.path.abspath(path))
    try:
        runner.init()
        for f in IMAGES:
            img = cv2.cvtColor(
                cv2.imread(f), cv2.COLOR_BGR2RGB
            )  # Edge Impulse expects RGB
            features, seen = features_for(runner, img)  # seen = the image FOMO got
            boxes = runner.classify(features)["result"].get("bounding_boxes", [])
            vis = cv2.cvtColor(seen, cv2.COLOR_RGB2BGR)
            vis = cv2.resize(
                vis, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST
            )
            for b in boxes:
                x, y = int(b["x"] * SCALE), int(b["y"] * SCALE)
                w, h = int(b["width"] * SCALE), int(b["height"] * SCALE)
                cx, cy = x + w // 2, y + h // 2
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 1)  # grid cell
                cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)  # object centre
                cv2.putText(
                    vis,
                    "%.2f" % b["value"],
                    (x, max(y - 3, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 255),
                    1,
                )
            cv2.putText(
                vis,
                "%s  %d found" % (name, len(boxes)),
                (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            cv2.imwrite(os.path.join(out_dir, os.path.basename(f)), vis)
        print("%-30s -> %s" % (name, out_dir))
    finally:
        runner.stop()
