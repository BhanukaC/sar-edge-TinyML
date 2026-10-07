# sar-edge/scripts/make_fomo_upload.py   run from sar-edge/: python scripts/make_fomo_upload.py
# Copies 300 SARD training images and 50 validation images, with their YOLO labels,
# into data/fomo_upload/ for uploading to Edge Impulse.
import glob, os, random, shutil

random.seed(42)


def copy_set(images, out):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        label = os.path.join(
            os.path.dirname(os.path.dirname(img)), "labels", name + ".txt"
        )
        shutil.copy(img, os.path.join(out, "images"))
        if os.path.exists(label):
            shutil.copy(label, os.path.join(out, "labels"))


train = sorted(glob.glob("data/SARD/train/images/*.jpg"))
valid = sorted(glob.glob("data/SARD/valid/images/*.jpg"))
copy_set(random.sample(train, 300), "data/fomo_upload/train")
copy_set(random.sample(valid, 50), "data/fomo_upload/test")

# Edge Impulse maps class numbers to names from data.yaml or classes.txt,
# which must be inside each folder we upload
for part in ("train", "test"):
    shutil.copy("data/SARD/data.yaml", "data/fomo_upload/%s/data.yaml" % part)
    with open("data/fomo_upload/%s/classes.txt" % part, "w") as fh:
        fh.write("human\n")
print("done: 300 train, 50 test images in data/fomo_upload/")
