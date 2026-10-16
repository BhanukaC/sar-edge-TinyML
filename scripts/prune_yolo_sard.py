# sar-edge/scripts/prune_yolo_sard.py   run from sar-edge/: python scripts/prune_yolo_sard.py
#
# Prunes the SARD-trained YOLO26n to two smaller versions (p30 and p50) and saves them
# as models/yolo26n-sard_<tag>_pruned.pt, ready to be retrained by train_pruned_yolo.py.
# (prune_yolo.py is the smoke-test version, which pruned the COCO model and exported ONNX.)
#
# Order of the project: train the full model on SARD -> prune it here -> retrain the
# pruned versions. Pruning removes channels, which hurts accuracy until the remaining
# layers have been retrained, so these files are not exported to ONNX yet.
#
# All settings were worked out and verified in the smoke test:
#   - C3k2 blocks rewritten with SplitC2f, so Torch-Pruning can track their channels
#   - detection head and attention blocks left unpruned (the head's outputs must keep
#     their size, and attention splits channels into heads that must stay equal)
#   - graph traced at the real input size, 640x640
#   - round_to=None, so "30%" means 30% in every layer, whatever its size
import os, sys, time

sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)  # scripts/, so yolo_prune_utils can be imported
import torch, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f  # C3k2 is a subclass of C2f
from yolo_prune_utils import SplitC2f, first_tensor

SOURCE = "models/yolo26n-sard_p00.pt"  # the full model fine-tuned on SARD

t0 = time.time()


def log(msg):
    # print with the time since start, so slow steps are easy to spot
    print("[%5.1fs] %s" % (time.time() - t0, msg), flush=True)


for ratio, tag in [(0.3, "p30"), (0.5, "p50")]:
    # Each prune level starts again from a fresh copy of the trained model,
    # so p50 is not "p30 pruned further".
    log("%s: loading %s" % (tag, SOURCE))
    model = YOLO(
        SOURCE
    ).model.float()  # float32 for pruning (checkpoints may be stored in float16)

    # 1. Rewrite C2f / C3k2 blocks, then check that the model still computes the same.
    #    A random input is enough for this check: if the rewrite were wrong, any input
    #    would show a difference.
    x = torch.randn(1, 3, 640, 640)
    model.eval()  # evaluation mode: batch norm uses its stored statistics
    with torch.no_grad():
        before = first_tensor(model(x)).clone()
    n = 0
    for i, m in enumerate(model.model):
        if isinstance(m, C2f):
            model.model[i] = SplitC2f(m)
            n += 1
    with torch.no_grad():
        after = first_tensor(model(x))
    log(
        "%s: rewrote %d C2f/C3k2 blocks, max output difference %.2e"
        % (tag, n, (before - after).abs().max())
    )  # expect about 0

    # 2. Prune.
    #    Torch-Pruning needs gradients enabled to trace the model's structure.
    for p in model.parameters():
        p.requires_grad_(True)
    base_macs, base_params = tp.utils.count_ops_and_params(
        model, x
    )  # size before pruning, for the report

    # Layers whose output channels must not be removed:
    #   - model.model[-1] is the detection head: its outputs are the boxes and scores
    #   - C2PSA / PSABlock / Attention: attention reshapes channels into equal "heads",
    #     and removing channels unevenly would break that reshape
    ignored = [model.model[-1]]
    ignored += [
        m
        for m in model.modules()
        if type(m).__name__ in ("C2PSA", "PSABlock", "Attention")
    ]
    log("%s: ignoring head + %d attention modules" % (tag, len(ignored) - 1))

    pruner = tp.pruner.MetaPruner(
        model,
        x,  # x shows Torch-Pruning how data flows through the model
        importance=tp.importance.MagnitudeImportance(
            p=2
        ),  # remove channels with the smallest weights (L2 norm)
        pruning_ratio=ratio,  # share of channels removed in each prunable layer
        ignored_layers=ignored,
        round_to=None,
    )  # exact ratio, no rounding of channel counts
    pruner.step()  # actually removes the channels and rewires the layers

    macs, params = tp.utils.count_ops_and_params(model, x)
    log(
        "%s: %d parameters (%.1f%% of %d), MACs %.1f%%"
        % (tag, params, 100 * params / base_params, base_params, 100 * macs / base_macs)
    )

    # 3. Save the whole pruned network (structure and weights).
    #    torch.save(model) keeps the new, smaller layer shapes; saving only the weights
    #    would not work, because they no longer fit the original YOLO26n structure.
    out = "models/yolo26n-sard_%s_pruned.pt" % tag
    torch.save(model, out)
    log("%s: saved %s" % (tag, out))
