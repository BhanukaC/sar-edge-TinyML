# sar-edge/scripts/make_fomo_upload_full.py   run from sar-edge/: python scripts/make_fomo_upload_full.py
# Copies the full SARD v16 train/valid/test splits, with their YOLO labels and class list,
# into data/fomo_full/ for uploading to Edge Impulse.
import os, shutil

for part in ("train", "valid", "test"):
    src, dst = "data/SARD/%s" % part, "data/fomo_full/%s" % part
    shutil.copytree(
        os.path.join(src, "images"), os.path.join(dst, "images"), dirs_exist_ok=True
    )
    shutil.copytree(
        os.path.join(src, "labels"), os.path.join(dst, "labels"), dirs_exist_ok=True
    )
    shutil.copy("data/SARD/data.yaml", os.path.join(dst, "data.yaml"))
    with open(os.path.join(dst, "classes.txt"), "w") as fh:
        fh.write("human\n")
    print(part, len(os.listdir(os.path.join(dst, "images"))), "images")
