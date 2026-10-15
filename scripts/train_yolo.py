# sar-edge/scripts/train_yolo.py   run from sar-edge/: python scripts/train_yolo.py [epochs]
# Fine-tunes COCO-pretrained YOLO26n on SARD (one class: human) and evaluates it.
# Result: runs/yolo26n_sard/weights/best.pt, copied to models/yolo26n_sard_p00.pt
import sys, shutil, torch
from ultralytics import YOLO

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100

if torch.cuda.is_available():
    device = 0  # Colab / NVIDIA GPU
elif torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon Mac
else:
    device = "cpu"
print("training on", device)

model = YOLO("yolo26n.pt")  # start from COCO weights (transfer learning)
model.train(
    data="data/sard.yaml",
    epochs=EPOCHS,
    imgsz=640,  # same input size as the smoke test
    batch=16,
    device=device,
    seed=42,  # reproducible
    patience=20,  # stop early if validation mAP stops improving for 20 epochs
    project="runs",
    name="yolo26n_sard",
    exist_ok=True,
)

save_dir = model.trainer.save_dir  # where Ultralytics actually saved this run
best_path = "%s/weights/best.pt" % save_dir
print("results in", save_dir)

best = YOLO(best_path)
metrics = best.val(data="data/sard.yaml", split="test", imgsz=640, device=device)
print(
    "test mAP50 %.3f  mAP50-95 %.3f  precision %.3f  recall %.3f"
    % (metrics.box.map50, metrics.box.map, metrics.box.mp, metrics.box.mr)
)
shutil.copy(best_path, "models/yolo26n-sard_p00.pt")
