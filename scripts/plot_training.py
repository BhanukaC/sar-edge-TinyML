# sar-edge/scripts/plot_training.py   run from sar-edge/: python scripts/plot_training.py
# Draws validation mAP50 and mAP50-95 per epoch for p00 (full training) and p30/p50
# (retraining after pruning), from Ultralytics' results.csv files, into one figure.
import csv
import matplotlib.pyplot as plt

RUNS = {
    "p00 (full, 100 epochs)": "runs/detect/runs/yolo26n_sard/results.csv",
    "p30 (retrained after pruning)": "runs/detect/runs/yolo26n-sard_p30_ft/results.csv",
    "p50 (retrained after pruning)": "runs/detect/runs/yolo26n-sard_p50_ft/results.csv",
}


def read(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    # column names can carry stray spaces in some Ultralytics versions
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    epoch = [int(r["epoch"]) for r in rows]
    map50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    map5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    return epoch, map50, map5095


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for label, path in RUNS.items():
    epoch, map50, map5095 = read(path)
    ax1.plot(epoch, map50, label=label)
    ax2.plot(epoch, map5095, label=label)
ax1.set_title("Validation mAP50")
ax2.set_title("Validation mAP50-95")
for ax in (ax1, ax2):
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
ax1.set_ylabel("mAP")
ax1.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/yolo_train/training_curves.png", dpi=200)
print("saved results/yolo_train/training_curves.png")
