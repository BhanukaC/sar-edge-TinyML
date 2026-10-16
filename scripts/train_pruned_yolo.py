# sar-edge/scripts/train_pruned_yolo.py   run from sar-edge/: python scripts/train_pruned_yolo.py p30 [epochs]
#
# Retrains a pruned YOLO26n (made by prune_yolo.py) on SARD, so it recovers the accuracy
# that pruning took away, then tests it and saves it as models/yolo26n-sard_<tag>.pt.
#
# The problem this solves: Ultralytics' normal training always builds a fresh model from
# its config file (yolo26n.yaml) and copies matching weights into it. For a pruned model
# that would silently bring back the full-size network and throw the pruning away.
# So we subclass Ultralytics' trainer and make it use our pruned network directly.
import os, sys, shutil

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # scripts/, so yolo_prune_utils can be imported
import torch
from yolo_prune_utils import (
    SplitC2f,
)  # not used directly, but must be imported so torch.load

# can rebuild the pruned SplitC2f blocks
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

TAG = sys.argv[1]  # which pruned model: p30 or p50
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 50
# 50 epochs instead of 100: the pruned model starts from already-trained weights and
# mostly needs to readjust; patience=20 below stops earlier if it stops improving.

# Use the fastest available device: NVIDIA GPU (e.g. Colab), Apple GPU (mps), or CPU.
if torch.cuda.is_available():
    device = 0
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("training on", device)

# Load the whole pruned network. weights_only=False is needed because the file contains
# a full model object, not just a list of weights (newer PyTorch refuses this by default).
pruned = torch.load("models/yolo26n-sard_%s_pruned.pt" % TAG, weights_only=False)
for p in pruned.parameters():
    p.requires_grad_(True)  # all layers are trained, none frozen
n_before = sum(p.numel() for p in pruned.parameters())
print(
    "pruned model: %d parameters" % n_before
)  # remembered to check the pruning survives training


class PrunedTrainer(DetectionTrainer):
    """Ultralytics' detection trainer, but it trains our pruned network.

    Normally get_model() builds a new YOLO26n from yolo26n.yaml. Returning the pruned
    model instead keeps everything else of Ultralytics' training as usual: data
    augmentation, loss, optimizer, learning rate schedule, validation and saving.
    """

    def get_model(self, cfg=None, weights=None, verbose=True):
        return pruned


trainer = PrunedTrainer(
    overrides=dict(
        model="yolo26n.yaml",  # required to start the trainer; get_model() replaces it
        data="data/sard.yaml",  # same SARD v16 dataset and splits as the full model
        epochs=EPOCHS,
        imgsz=640,  # same input size as everywhere else
        batch=16,
        device=device,
        seed=42,  # same seed as the full model's training, for reproducibility
        patience=20,  # stop if validation mAP has not improved for 20 epochs
        project="runs",
        name="yolo26n-sard_%s_ft" % TAG,
        exist_ok=True,
    )
)
trainer.train()

# Ultralytics saves the best epoch (highest validation mAP) as best.pt. Ask the trainer
# where it saved, instead of assuming a path (this version adds "runs/detect/" in front).
best_path = "%s/weights/best.pt" % trainer.save_dir
best = YOLO(best_path)

# The key check: if the trainer had rebuilt the full model, the parameter count would
# jump back to about 2.4 million. A difference under 2% means the pruning was kept.
n_after = sum(p.numel() for p in best.model.parameters())
print(
    "parameters before training %d, after %d  (%s)"
    % (
        n_before,
        n_after,
        (
            "pruning kept"
            if abs(n_after - n_before) < 0.02 * n_before
            else "WARNING: model size changed, pruning may be lost"
        ),
    )
)

# Final accuracy on the 197 test images, never seen during training or validation.
m = best.val(data="data/sard.yaml", split="test", imgsz=640, device=device)
print(
    "test mAP50 %.3f  mAP50-95 %.3f  precision %.3f  recall %.3f"
    % (m.box.map50, m.box.map, m.box.mp, m.box.mr)
)

# Copy the retrained model next to the others, with the project's naming scheme.
shutil.copy(best_path, "models/yolo26n-sard_%s.pt" % TAG)
