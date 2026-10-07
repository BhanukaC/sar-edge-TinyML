# scripts/make_image_sets.py   run: python scripts/make_image_sets.py
import random, shutil, glob, os

random.seed(42)
val = sorted(glob.glob("data/SARD/valid/images/*.jpg"))
train = sorted(glob.glob("data/SARD/train/images/*.jpg"))
print(len(val), "val images,", len(train), "train images")  # both must be > 0
os.makedirs("data/smoke", exist_ok=True)
os.makedirs("data/calib", exist_ok=True)
for f in random.sample(val, 20):
    shutil.copy(f, "data/smoke/")
for f in random.sample(train, 100):
    shutil.copy(f, "data/calib/")
