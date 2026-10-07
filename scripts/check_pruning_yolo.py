# sar-edge/scripts/check_pruning_yolo.py   run from sar-edge/: python scripts/check_pruning_yolo.py [graph size]
# Prunes YOLO26n at increasing ratios and prints the top person score on bus.jpg.
# Smooth decline = pruning works but needs fine-tuning. A cliff at small ratios = bug.
import os, sys, time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))  # repo root, for common.py
sys.path.insert(0, here)  # scripts/, for prune_yolo.py
import cv2, numpy as np, torch, torch_pruning as tp, onnxruntime as ort
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.utils import ASSETS
from prune_yolo import SplitC2f, first_tensor
from common import prep_yolo, post_yolo_raw, post_yolo_e2e, PERSON_YOLO

GRAPH = (
    int(sys.argv[1]) if len(sys.argv) > 1 else 64
)  # image size used to trace the graph
print("tracing graph at %dx%d" % (GRAPH, GRAPH), flush=True)

img = cv2.imread(str(ASSETS / "bus.jpg"))
x, r, pad = prep_yolo(img)
xt = torch.from_numpy(x)


def person_result(out):
    """Top person score and number of people, for either YOLO output layout."""
    if out.shape[1] == 84:  # raw head: (1, 84, 8400)
        return out[0, 4 + PERSON_YOLO].max(), len(post_yolo_raw(out, r, pad))
    d = out[0]  # end-to-end head: (1, 300, 6)
    s = d[d[:, 5] == PERSON_YOLO, 4]
    return (s.max() if len(s) else 0.0), len(post_yolo_e2e(out, r, pad))


for ratio in [0.0, 0.05, 0.1, 0.2, 0.3]:
    t0 = time.time()
    model = YOLO("yolo26n.pt").model.float().eval()
    for i, m in enumerate(model.model):
        if isinstance(m, C2f):
            model.model[i] = SplitC2f(m)
    if ratio > 0:
        for p in model.parameters():
            p.requires_grad_(True)
        ignored = [model.model[-1]]
        ignored += [
            m
            for m in model.modules()
            if type(m).__name__ in ("C2PSA", "PSABlock", "Attention")
        ]
        tp.pruner.MetaPruner(
            model,
            torch.randn(1, 3, GRAPH, GRAPH),
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=ratio,
            ignored_layers=ignored,
            round_to=None,
        ).step()
    model.eval()
    with torch.no_grad():
        out = first_tensor(model(xt)).numpy()
    score, n = person_result(out)
    print(
        "ratio %.2f: top person score %.3f, people found %d  (%.1fs)"
        % (ratio, score, n, time.time() - t0),
        flush=True,
    )
