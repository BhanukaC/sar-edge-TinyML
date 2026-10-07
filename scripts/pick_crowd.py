# sar-edge/scripts/pick_crowd.py   run from sar-edge/: python scripts/pick_crowd.py
# Copies the validation image with the most people to data/smoke/crowd.jpg.
import glob, os, shutil


def count_people(label_file):
    # one non-empty line per labelled person in YOLO label files
    with open(label_file) as f:
        return sum(1 for line in f if line.strip())


labels = glob.glob("data/SARD/valid/labels/*.txt")
best = max(labels, key=count_people)
name = os.path.splitext(os.path.basename(best))[0]
img = glob.glob("data/SARD/valid/images/" + name + ".*")[0]  # .jpg or .png
shutil.copy(img, "data/smoke/crowd.jpg")
print("crowd.jpg =", os.path.basename(img), "with", count_people(best), "people")
