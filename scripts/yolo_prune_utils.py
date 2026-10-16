# sar-edge/scripts/yolo_prune_utils.py
#
# Shared helpers for pruning YOLO26n, imported by prune_yolo.py, train_pruned_yolo.py,
# check_pruning_yolo.py and verify_pruning_yolo.py.
#
# Why a separate file: when PyTorch saves a whole model with torch.save(), it stores
# the name of the file each class came from. If SplitC2f were defined inside a script
# that is run directly, PyTorch would record it as coming from "__main__", and loading
# the saved model later from another script would fail. Keeping it here gives it a
# stable home ("yolo_prune_utils.SplitC2f") that every script can import.
import copy
import torch, torch.nn as nn


class SplitC2f(nn.Module):
    """Same maths as Ultralytics' C2f / C3k2 block, written so Torch-Pruning can handle it.

    The original block runs one convolution (cv1) and then splits its output into two
    halves with chunk(2). Torch-Pruning loses track of which channel is which across
    that split, which caused an IndexError in the smoke test. Here cv1's weights are cut
    in half and each half becomes its own convolution (cv0 and cv1). The result is
    mathematically identical (checked: output difference 0), but every channel now has
    one clear producer, so the pruning tool can follow it.
    """

    def __init__(self, old):
        super().__init__()
        c = old.c  # number of channels in each half

        # Start from two copies of the original cv1 (conv + batch norm + activation) ...
        self.cv0 = copy.deepcopy(old.cv1)
        self.cv1 = copy.deepcopy(old.cv1)

        # ... and keep only the first half of the channels in cv0, the second half in cv1.
        # Every per-channel part has to be cut the same way: the conv weights, and the
        # batch norm's scale, shift and running statistics.
        for new, part in ((self.cv0, slice(0, c)), (self.cv1, slice(c, 2 * c))):
            conv, bn = new.conv, new.bn
            conv.weight = nn.Parameter(conv.weight.data[part].clone())
            conv.out_channels = c
            bn.weight = nn.Parameter(bn.weight.data[part].clone())
            bn.bias = nn.Parameter(bn.bias.data[part].clone())
            bn.running_mean = bn.running_mean[part].clone()
            bn.running_var = bn.running_var[part].clone()
            bn.num_features = c

        # The rest of the block (the inner bottleneck layers and the final conv) is reused as is.
        self.m, self.cv2 = old.m, old.cv2

        # Ultralytics stores routing information on each top-level layer: where its input
        # comes from (f), its index (i), its type name and parameter count. The model's
        # forward pass relies on these, so they must be copied over.
        for k in ("f", "i", "type", "np"):
            if hasattr(old, k):
                setattr(self, k, getattr(old, k))

    def forward(self, x):
        # Same data flow as the original block: two halves, then each inner layer works
        # on the output of the previous one, and everything is concatenated for cv2.
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def first_tensor(o):
    """Return the main output tensor of a YOLO model.

    Depending on the Ultralytics version and on training or evaluation mode, the model
    returns a tensor, a tuple, a list or a dictionary. For comparing outputs we only need
    the first tensor, so this unwraps whatever structure comes back.
    """
    while isinstance(o, (list, tuple)):
        o = o[0]
    if isinstance(o, dict):
        o = next(iter(o.values()))
    return o
