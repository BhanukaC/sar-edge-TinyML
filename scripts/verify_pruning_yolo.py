# sar-edge/scripts/verify_pruning_yolo.py   run from sar-edge/: python scripts/verify_pruning_yolo.py [ratio]
# Checks Torch-Pruning's wiring: physically pruning channels must give the same output
# as keeping them but forcing them to zero. Same output = wiring correct.
import os, sys, copy

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))  # repo root, for common.py
sys.path.insert(0, here)  # scripts/, for prune_yolo.py
import cv2, numpy as np, torch, torch.nn as nn, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.utils import ASSETS
from prune_yolo import SplitC2f, first_tensor
from common import prep_yolo, PERSON_YOLO

RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
img = cv2.imread(str(ASSETS / "bus.jpg"))
x, r, pad = prep_yolo(img)
xt = torch.from_numpy(x)


def run(model):
    model.eval()
    with torch.no_grad():
        return first_tensor(model(xt)).numpy()


model = YOLO("yolo26n.pt").model.float().eval()
for i, m in enumerate(model.model):
    if isinstance(m, C2f):
        model.model[i] = SplitC2f(m)
masked = copy.deepcopy(model)  # same model; channels will be zeroed, not removed
name_of = {m: n for n, m in model.named_modules()}
masked_mods = dict(masked.named_modules())

for p in model.parameters():
    p.requires_grad_(True)
ignored = [model.model[-1]]
ignored += [
    m for m in model.modules() if type(m).__name__ in ("C2PSA", "PSABlock", "Attention")
]
pruner = tp.pruner.MetaPruner(
    model,
    torch.randn(1, 3, 640, 640),
    importance=tp.importance.MagnitudeImportance(p=2),
    pruning_ratio=RATIO,
    ignored_layers=ignored,
    round_to=None,
)

n_zeroed = 0
for group in pruner.step(interactive=True):
    # before removing, zero the same channels in the copy (BN weight and bias = 0 -> output 0)
    for dep, idxs in group:
        mod = dep.target.module
        if isinstance(mod, nn.BatchNorm2d):
            bn = masked_mods[name_of[mod]]
            ids = [getattr(i, "idx", i) for i in idxs]
            with torch.no_grad():
                bn.weight[ids] = 0
                bn.bias[ids] = 0
            n_zeroed += len(ids)
    group.prune()

out_p, out_m = run(model), run(masked)
print("ratio %.2f, %d channels zeroed in the copy" % (RATIO, n_zeroed))
print("pruned model : top person score %.3f" % out_p[0, 4 + PERSON_YOLO].max())
print("masked model : top person score %.3f" % out_m[0, 4 + PERSON_YOLO].max())
print("max output difference pruned vs masked: %.2e" % np.abs(out_p - out_m).max())
