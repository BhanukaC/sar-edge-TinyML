# sar-edge/scripts/debug_prune_groups.py   run from sar-edge/: python scripts/debug_prune_groups.py
# Prunes YOLO26n one dependency group at a time (ratio 0.05) and stops at the
# first group that breaks the person score on bus.jpg, then prints that group.
import os, sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))  # repo root, for common.py
sys.path.insert(0, here)  # scripts/, for prune_yolo.py
import cv2, torch, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.utils import ASSETS
from prune_yolo import SplitC2f, first_tensor
from common import prep_yolo, PERSON_YOLO

img = cv2.imread(str(ASSETS / "bus.jpg"))
x, r, pad = prep_yolo(img)
xt = torch.from_numpy(x)


def score(model):
    model.eval()
    with torch.no_grad():
        out = first_tensor(model(xt)).numpy()
    return float(out[0, 4 + PERSON_YOLO].max())


model = YOLO("yolo26n.pt").model.float().eval()
for i, m in enumerate(model.model):
    if isinstance(m, C2f):
        model.model[i] = SplitC2f(m)
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
    pruning_ratio=0.05,
    ignored_layers=ignored,
    round_to=8,
)
print("before pruning: score %.3f" % score(model), flush=True)

for k, group in enumerate(pruner.step(interactive=True)):
    group.prune()
    s = score(model)
    root = group[0].dep.target.name  # the layer this group starts from
    print("group %2d (%s): score %.3f" % (k, root, s), flush=True)
    if s < 0.5:
        print("\nScore broke at this group. Full group:\n")
        print(group)
        break
