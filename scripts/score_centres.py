# sar-edge/scripts/score_centres.py   run on the laptop from sar-edge/:
#   python scripts/score_centres.py <tolerance px> results/pi/fomo_test/*.json results/jetson/fomo_test/*.json
# Scores detection centres against the SARD test labels with one rule for every model:
# a detection is correct if its centre lies inside a labelled person box, widened by
# <tolerance> pixels on each side, that has not been matched yet (highest scores first).
# Tolerance 0 = strict; 24 = half a FOMO grid cell in the frame (FOMO's own precision).
import os, sys, json
import cv2

TOL = float(sys.argv[1])
FILES = sys.argv[2:]


def gt_boxes(image_name):
    """Labelled person boxes in pixels (x1, y1, x2, y2) for one test image."""
    img = cv2.imread(os.path.join("data/SARD/test/images", image_name))
    h, w = img.shape[:2]
    label = os.path.join(
        "data/SARD/test/labels", os.path.splitext(image_name)[0] + ".txt"
    )
    boxes = []
    if os.path.exists(label):
        for line in open(label):
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = (
                    float(p[1]) * w,
                    float(p[2]) * h,
                    float(p[3]) * w,
                    float(p[4]) * h,
                )
                boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes


gt_cache = {}
for path in FILES:
    with open(path) as fh:
        dets = json.load(fh)
    tp = fp = fn = 0
    for image_name, centres in dets.items():
        if image_name not in gt_cache:
            gt_cache[image_name] = gt_boxes(image_name)
        gts, used = gt_cache[image_name], set()
        for x, y, score in sorted(centres, key=lambda d: -d[2]):
            hit = next(
                (
                    i
                    for i, (x1, y1, x2, y2) in enumerate(gts)
                    if i not in used
                    and x1 - TOL <= x <= x2 + TOL
                    and y1 - TOL <= y <= y2 + TOL
                ),
                None,
            )
            if hit is None:
                fp += 1
            else:
                used.add(hit)
                tp += 1
        fn += len(gts) - len(used)
    p = tp / float(tp + fp) if tp + fp else 0.0
    r = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(
        "%-60s P %.3f  R %.3f  F1 %.3f  (TP %d, FP %d, FN %d)"
        % (path, p, r, f1, tp, fp, fn)
    )
