# sar-edge/scripts/train_ssdlite.py   run from sar-edge/: python scripts/train_ssdlite.py [epochs]
#
# Fine-tunes COCO-pretrained SSDLite-MobileNetV3 on SARD (classes: background, human),
# keeps the epoch with the best validation mAP50, tests it on the 197 test images, and
# saves the whole model as models/ssdlite-sard_p00.pt (for pruning and export later).
#
# Choices made so results compare with YOLO:
#   - same SARD v16 splits (train / valid / test)
#   - images stretched to 320x320, exactly like prep_ssd() in common.py on the boards
#   - mAP computed with Ultralytics' own box_iou and ap_per_class, using the same
#     matching rule as its validator, so mAP50 and mAP50-95 mean the same as for YOLO
import os, sys, glob, copy, random, functools, time
import numpy as np, cv2, torch, torch.nn as nn
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large,
    SSDLite320_MobileNet_V3_Large_Weights,
)
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.models.detection import _utils as det_utils
from ultralytics.utils.metrics import ap_per_class, box_iou

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SIZE = 320  # SSDLite's native input size, as on the boards
BATCH = 32
PATIENCE = 20  # stop if validation mAP50 has not improved for 20 epochs

device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
print("training on", device)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)  # reproducible, same seed as YOLO


def load_split(split):
    """Load a whole split into memory once, stretched to 320x320, with boxes in 320x320 pixels.

    Decoding 1920x1080 JPEGs every epoch would be the slowest part of training, so each
    image is decoded and resized only once. Stretching (not letterboxing) matches both
    torchvision's own SSDLite transform and prep_ssd() on the boards.
    """
    imgs, targets = [], []
    for f in sorted(glob.glob("data/SARD/%s/images/*.jpg" % split)):
        img = cv2.cvtColor(cv2.resize(cv2.imread(f), (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
        boxes = []
        label = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(label):
            for line in open(label):
                p = line.split()
                if len(p) == 5:
                    # YOLO labels are normalised (0 to 1), so multiplying by 320 gives
                    # coordinates in the stretched 320x320 image directly
                    cx, cy, bw, bh = [float(v) * SIZE for v in p[1:]]
                    x1, y1 = max(cx - bw / 2, 0.0), max(cy - bh / 2, 0.0)
                    x2, y2 = min(cx + bw / 2, SIZE), min(cy + bh / 2, SIZE)
                    if x2 > x1 and y2 > y1:  # torchvision rejects empty boxes
                        boxes.append([x1, y1, x2, y2])
        imgs.append(
            torch.from_numpy(img).permute(2, 0, 1).contiguous()
        )  # uint8, 3x320x320
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        targets.append(
            {"boxes": b, "labels": torch.ones(len(b), dtype=torch.int64)}
        )  # 1 = human
    print(
        "%s: %d images, %d people"
        % (split, len(imgs), sum(len(t["boxes"]) for t in targets))
    )
    return imgs, targets


def build_model():
    """COCO-pretrained SSDLite with a new 2-class score head (background, human).

    The backbone and the box-regression head keep their COCO weights (transfer
    learning); only the classification head is replaced, since it had 91 COCO classes.
    """
    model = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
    )
    in_channels = det_utils.retrieve_out_channels(model.backbone, (SIZE, SIZE))
    n_anchors = model.anchor_generator.num_anchors_per_location()
    norm = functools.partial(
        nn.BatchNorm2d, eps=0.001, momentum=0.03
    )  # as in torchvision's SSDLite
    model.head.classification_head = SSDLiteClassificationHead(
        in_channels, n_anchors, 2, norm
    )
    # SARD's people are only a few pixels wide at 320x320 (median 6 x 15 px), far smaller
    # than SSDLite's COCO anchors (smallest about 64 px). Anchors from 3% to 30% of the
    # image (about 10 to 96 px) fit them. Same number of anchors per position (6), so the
    # pretrained heads keep their shapes; the trained model saves its own anchors for export.
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

    model.anchor_generator = DefaultBoxGenerator(
        [[2, 3] for _ in range(6)], min_ratio=0.03, max_ratio=0.3
    )
    return model


def batch_to_device(imgs, targets, idx, flip):
    """Turn a list of indices into model input, with random horizontal flips as augmentation."""
    ims, tgs = [], []
    for i in idx:
        im, t = imgs[i].float() / 255.0, dict(targets[i])
        if flip and random.random() < 0.5:
            im = im.flip(-1)  # mirror the image left-right ...
            b = t["boxes"].clone()
            b[:, [0, 2]] = SIZE - t["boxes"][:, [2, 0]]  # ... and the boxes with it
            t["boxes"] = b
        ims.append(im.to(device))
        tgs.append({k: v.to(device) for k, v in t.items()})
    return ims, tgs


@torch.no_grad()
def evaluate(model, imgs, targets):
    """Precision, recall, mAP50 and mAP50-95, computed like Ultralytics' validator.

    Low score threshold (0.001) and up to 300 detections per image, as for mAP in
    Ultralytics. IoU does not change when an image is stretched, so computing it in
    the 320x320 space gives the same result as in the original frame.
    """
    model.eval()
    model.score_thresh, model.detections_per_img = 0.001, 300
    iouv = torch.linspace(0.5, 0.95, 10)  # the 10 IoU thresholds of mAP50-95
    stats = []
    for s in range(0, len(imgs), BATCH):
        idx = list(range(s, min(s + BATCH, len(imgs))))
        ims, _ = batch_to_device(imgs, targets, idx, flip=False)
        for o, i in zip(model(ims), idx):
            pb, ps, gb = o["boxes"].cpu(), o["scores"].cpu(), targets[i]["boxes"]
            tp = np.zeros((len(pb), len(iouv)), dtype=bool)
            if len(pb) and len(gb):
                iou = box_iou(gb, pb)  # (people, detections)
                for j, thr in enumerate(iouv):
                    # each detection matches at most one person and vice versa, best IoU first
                    x = torch.nonzero(iou >= thr)
                    if len(x):
                        m = torch.cat(
                            [x.float(), iou[x[:, 0], x[:, 1]].unsqueeze(1)], 1
                        ).numpy()
                        m = m[m[:, 2].argsort()[::-1]]
                        m = m[np.unique(m[:, 1], return_index=True)[1]]
                        m = m[np.unique(m[:, 0], return_index=True)[1]]
                        tp[m[:, 1].astype(int), j] = True
            stats.append((tp, ps.numpy(), np.zeros(len(pb)), np.zeros(len(gb))))
    model.train()
    tp, conf, pred_cls, target_cls = [np.concatenate(x, 0) for x in zip(*stats)]
    if len(tp) == 0:
        return 0.0, 0.0, 0.0, 0.0
    r = ap_per_class(tp, conf, pred_cls, target_cls)  # Ultralytics' AP computation
    p, rec, ap = r[2], r[3], r[5]
    return float(p.mean()), float(rec.mean()), float(ap[:, 0].mean()), float(ap.mean())


train_imgs, train_targets = load_split("train")
val_imgs, val_targets = load_split("valid")

model = build_model().to(device)
opt = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.01,
    momentum=0.9,
    weight_decay=4e-5,
)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=EPOCHS
)  # smooth decay to ~0

os.makedirs("results/ssdlite_train", exist_ok=True)
log = open("results/ssdlite_train/results.csv", "w")
log.write("epoch,time,train_loss,precision,recall,mAP50,mAP50-95\n")
best, best_epoch, t0 = -1.0, 0, time.time()
model.train()
for epoch in range(1, EPOCHS + 1):
    order = list(range(len(train_imgs)))
    random.shuffle(order)
    total = 0.0
    for s in range(0, len(order), BATCH):
        ims, tgs = batch_to_device(
            train_imgs, train_targets, order[s : s + BATCH], flip=True
        )
        losses = model(ims, tgs)  # box loss + classification loss
        loss = sum(losses.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    sched.step()
    p, r, m50, m5095 = evaluate(model, val_imgs, val_targets)
    log.write(
        "%d,%.1f,%.4f,%.4f,%.4f,%.4f,%.4f\n"
        % (epoch, time.time() - t0, total, p, r, m50, m5095)
    )
    log.flush()
    print(
        "epoch %3d  loss %8.2f  val P %.3f R %.3f mAP50 %.3f mAP50-95 %.3f  (%.0f s)"
        % (epoch, total, p, r, m50, m5095, time.time() - t0),
        flush=True,
    )
    if m50 > best:  # keep the best epoch, like Ultralytics' best.pt
        best, best_epoch = m50, epoch
        torch.save(copy.deepcopy(model).cpu(), "models/ssdlite-sard_p00.pt")
    elif epoch - best_epoch >= PATIENCE:
        print("no improvement for %d epochs, stopping" % PATIENCE)
        break

print("best epoch %d, validation mAP50 %.3f" % (best_epoch, best))
test_imgs, test_targets = load_split("test")
model = torch.load("models/ssdlite-sard_p00.pt", weights_only=False).to(device)
p, r, m50, m5095 = evaluate(model, test_imgs, test_targets)
print(
    "test mAP50 %.3f  mAP50-95 %.3f  precision %.3f  recall %.3f" % (m50, m5095, p, r)
)
