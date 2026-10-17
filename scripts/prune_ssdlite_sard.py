# sar-edge/scripts/prune_ssdlite_sard.py   run from sar-edge/: python scripts/prune_ssdlite_sard.py
#
# Prunes the SARD-trained SSDLite (models/ssdlite-sard_p00.pt) to p30 and p50 and saves
# the pruned models as models/ssdlite-sard_<tag>_pruned.pt, ready for retraining with
# train_ssdlite.py. Same settings as the smoke test and as YOLO: L2 magnitude importance,
# round_to=None, the detection head left unpruned (its outputs are anchors x classes
# and anchors x 4 box values, so their sizes must stay).
import torch, torch_pruning as tp

SOURCE = "models/ssdlite-sard_p00.pt"


class SSDRaw(torch.nn.Module):
    """Backbone + head only, without torchvision's resizing and post-processing.

    Torch-Pruning traces this to find which layers depend on each other. It shares its
    modules with the full model, so pruning it prunes the full model in place.
    """

    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head

    def forward(self, x):
        out = self.head(list(self.backbone(x).values()))
        return out["bbox_regression"], out["cls_logits"]


for ratio, tag in [(0.3, "p30"), (0.5, "p50")]:
    full = (
        torch.load(SOURCE, weights_only=False).float().eval()
    )  # fresh copy for each level
    net = SSDRaw(full).eval()
    x = torch.randn(1, 3, 320, 320)
    base_macs, base_params = tp.utils.count_ops_and_params(net, x)

    for p in net.parameters():
        p.requires_grad_(True)  # needed to trace the model
    pruner = tp.pruner.MetaPruner(
        net,
        x,
        importance=tp.importance.MagnitudeImportance(
            p=2
        ),  # remove filters with the smallest weights
        pruning_ratio=ratio,
        ignored_layers=[net.head],  # keep the output sizes of the head
        round_to=None,
    )
    pruner.step()
    macs, params = tp.utils.count_ops_and_params(net, x)

    # the full model (with torchvision's own resizing and post-processing) must still run
    full.eval()
    with torch.no_grad():
        full([torch.rand(3, 320, 320)])

    out = "models/ssdlite-sard_%s_pruned.pt" % tag
    torch.save(full, out)
    print(
        "%s: %d parameters (%.1f%% of %d), MACs %.1f%%  -> %s"
        % (
            tag,
            params,
            100.0 * params / base_params,
            base_params,
            100.0 * macs / base_macs,
            out,
        )
    )
