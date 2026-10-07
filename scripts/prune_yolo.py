# sar-edge/scripts/prune_yolo.py   run from sar-edge/: python scripts/prune_yolo.py
# Makes the p30 and p50 versions of YOLO26n and exports them to ONNX.
import time, shutil
from ultralytics import YOLO
import torch, torch_pruning as tp

t0 = time.time()


def log(msg):
    print("[%5.1fs] %s" % (time.time() - t0, msg), flush=True)


for ratio, tag in [(0.3, "p30"), (0.5, "p50")]:
    log("%s: loading model" % tag)
    y = YOLO("yolo26n.pt")
    model = y.model.float()
    for p in model.parameters():
        p.requires_grad_(True)
    example = torch.randn(1, 3, 640, 640)
    base_macs, base_params = tp.utils.count_ops_and_params(model, example)

    log("%s: building dependency graph" % tag)
    pruner = tp.pruner.MetaPruner(
        model,
        example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=[model.model[-1]],  # never prune the detection head
        round_to=8,
    )

    log("%s: pruning" % tag)
    pruner.step()
    macs, params = tp.utils.count_ops_and_params(model, example)
    log(
        "%s: params %.3f, MACs %.3f of baseline"
        % (tag, params / base_params, macs / base_macs)
    )

    log("%s: exporting ONNX" % tag)
    y.model = model
    path = y.export(
        format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False, batch=1
    )
    shutil.move(path, "models/yolo26n_%s_fp32_640.onnx" % tag)
    log("%s: done" % tag)
