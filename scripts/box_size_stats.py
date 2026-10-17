# sar-edge/scripts/box_size_stats.py   run from sar-edge/: python scripts/box_size_stats.py
# How big SARD's people are, in the frame and in each model's input, compared with
# SSDLite's smallest default anchor (about 20% of 320 = 64 px).
import glob
import numpy as np

w, h = [], []
for f in glob.glob("data/SARD/train/labels/*.txt"):
    for line in open(f):
        p = line.split()
        if len(p) == 5:
            w.append(float(p[3]))  # normalised width (0 to 1)
            h.append(float(p[4]))  # normalised height
w, h = np.array(w), np.array(h)


def show(name, width_px, height_px):
    pw, ph = w * width_px, h * height_px
    print(
        "%-28s width median %5.1f px (10%%: %5.1f, 90%%: %5.1f)   height median %5.1f px (10%%: %5.1f, 90%%: %5.1f)"
        % (
            name,
            np.median(pw),
            np.percentile(pw, 10),
            np.percentile(pw, 90),
            np.median(ph),
            np.percentile(ph, 10),
            np.percentile(ph, 90),
        )
    )


print("%d people in the training set" % len(w))
show("Original frame 1920 x 1080", 1920, 1080)
show("YOLO input (640 x 360 image)", 640, 360)
show("SSDLite input 320 x 320", 320, 320)
print("SSDLite's smallest default anchor: about 64 x 64 px")
