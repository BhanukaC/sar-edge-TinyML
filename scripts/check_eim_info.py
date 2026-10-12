# sar-edge/scripts/check_eim_info.py   run on the Pi from the repo folder: python scripts/check_eim_info.py [pattern]
# Prints what each FOMO .eim reports about itself (project, deployment, impulse, threshold),
# to check that every file came from the intended impulse.
import os, sys, glob
from edge_impulse_linux.image import ImageImpulseRunner

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "models/fomo-*-sard_p00_*_320.eim"

for path in sorted(glob.glob(PATTERN)):
    if "_jetson_" in path:  # Jetson builds don't run on the Pi
        continue
    runner = ImageImpulseRunner(os.path.abspath(path))
    try:
        info = runner.init()
    finally:
        runner.stop()
    params = info.get("model_parameters", {})
    print(path)
    print("  project:  ", info.get("project"))
    print("  impulse:  ", info.get("impulse", "(not reported)"))
    print("  threshold:", params.get("threshold"))
    print(
        "  input:    ",
        params.get("image_input_width"),
        "x",
        params.get("image_input_height"),
    )
