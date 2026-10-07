# sar-edge/scripts/prune_yolo.py   run from sar-edge/: python scripts/prune_yolo.py
# Makes the p30 and p50 versions of YOLO26n and exports them to ONNX.
# Also imported by check_pruning_yolo.py and verify_pruning_yolo.py (SplitC2f,
# first_tensor), so the pruning itself only runs under if __name__ == "__main__".
# Fixes needed for Torch-Pruning 1.6.1 on YOLO26 (7-8 October):
#   1. C2f/C3k2 blocks split one conv's output with chunk(2), which Torch-Pruning
#      cannot track (IndexError). SplitC2f rewrites it as two convs, same maths.
#   2. The attention blocks (C2PSA) reshape channels into heads; they are left
#      unpruned. With them excluded, tracing at full 640x640 takes under a second.
#   3. round_to=None, so the pruning ratio means the same for every layer size.
import time, shutil, copy
import torch, torch.nn as nn, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f

t0 = time.time()


def log(msg):
    print("[%5.1fs] %s" % (time.time() - t0, msg), flush=True)


class SplitC2f(nn.Module):
    """Same maths as Ultralytics C2f / C3k2, but 'cv1 then chunk(2)' becomes two convs."""

    def __init__(self, old):
        super().__init__()
        c = old.c  # channels in each half
        self.cv0 = copy.deepcopy(old.cv1)
        self.cv1 = copy.deepcopy(old.cv1)
        for new, part in ((self.cv0, slice(0, c)), (self.cv1, slice(c, 2 * c))):
            conv, bn = new.conv, new.bn
            conv.weight = nn.Parameter(conv.weight.data[part].clone())
            conv.out_channels = c
            bn.weight = nn.Parameter(bn.weight.data[part].clone())
            bn.bias = nn.Parameter(bn.bias.data[part].clone())
            bn.running_mean = bn.running_mean[part].clone()
            bn.running_var = bn.running_var[part].clone()
            bn.num_features = c
        self.m, self.cv2 = old.m, old.cv2
        for k in ("f", "i", "type", "np"):  # Ultralytics routing info
            if hasattr(old, k):
                setattr(self, k, getattr(old, k))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def first_tensor(o):
    # model output can be a tuple, list or dict depending on the version
    while isinstance(o, (list, tuple)):
        o = o[0]
    if isinstance(o, dict):
        o = next(iter(o.values()))
    return o


if __name__ == "__main__":
    for ratio, tag in [(0.3, "p30"), (0.5, "p50")]:
        log("%s: loading model" % tag)
        y = YOLO("yolo26n.pt")
        model = y.model.float()

        # 1. rewrite C2f / C3k2 blocks and check the outputs did not change
        x = torch.randn(1, 3, 640, 640)
        model.eval()
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
        )

        # 2. prune
        for p in model.parameters():
            p.requires_grad_(True)
        example = torch.randn(1, 3, 640, 640)
        base_macs, base_params = tp.utils.count_ops_and_params(model, example)

        ignored = [model.model[-1]]  # detection head
        ignored += [
            m
            for m in model.modules()
            if type(m).__name__ in ("C2PSA", "PSABlock", "Attention")
        ]
        log("%s: ignoring head + %d attention modules" % (tag, len(ignored) - 1))

        pruner = tp.pruner.MetaPruner(
            model,
            example,
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=ratio,
            ignored_layers=ignored,
            round_to=None,
        )
        log("%s: pruning" % tag)
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example)
        log(
            "%s: params %.3f, MACs %.3f of baseline"
            % (tag, params / base_params, macs / base_macs)
        )

        # 3. export
        log("%s: exporting ONNX" % tag)
        y.model = model
        path = y.export(
            format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False, batch=1
        )
        shutil.move(path, "models/yolo26n_%s_fp32_640.onnx" % tag)
        log("%s: done" % tag)
