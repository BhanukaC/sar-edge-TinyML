# sar-edge/scripts/train_pruned_ssdlite.py   run from sar-edge/: python scripts/train_pruned_ssdlite.py p30 [epochs]
#
# Retrains a pruned SSDLite (made by prune_ssdlite_sard.py) on SARD to recover the accuracy
# that pruning took away, keeps the epoch with the best validation mAP50, tests it on the
# 197 test images, and saves it as models/ssdlite-sard_<tag>.pt.
#
# Same data, augmentation, optimizer and evaluation as train_ssdlite.py (used for p00), so
# p00, p30 and p50 are trained and measured the same way. The only difference: the model is
# loaded from the pruned file instead of being built from COCO weights, so it keeps its
# smaller structure and its SARD anchors.
import os, sys, glob, copy, random, time
import numpy as np, cv2, torch
from ultralytics.utils.metrics import ap_per_class, box_iou

TAG = sys.argv[1]  # which pruned model: p30 or p50
EPOCHS = (
    int(sys.argv[2]) if len(sys.argv) > 2 else 50
)  # 50, as for YOLO's pruned models
START = "models/ssdlite-sard_%s_pruned.pt" % TAG
OUT = "models/ssdlite-sard_%s.pt" % TAG
SIZE = 320  # SSDLite's input size, as on the boards
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
torch.manual_seed(42)  # same seed as all other models


def load_split(split):
    """Load a whole split into memory once, stretched to 320x320, with boxes in 320x320 pixels.
    (Identical to train_ssdlite.py.)"""
    imgs, targets = [], []
    for f in sorted(glob.glob("data/SARD/%s/images/*.jpg" % split)):
        img = cv2.cvtColor(cv2.resize(cv2.imread(f), (SIZE, SIZE)), cv2.COLOR_BGR2RGB)
        boxes = []
        label = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(label):
            for line in open(label):
                p = line.split()
                if len(p) == 5:
                    cx, cy, bw, bh = [
                        float(v) * SIZE for v in p[1:]
                    ]  # normalised -> 320x320 pixels
                    x1, y1 = max(cx - bw / 2, 0.0), max(cy - bh / 2, 0.0)
                    x2, y2 = min(cx + bw / 2, SIZE), min(cy + bh / 2, SIZE)
                    if x2 > x1 and y2 > y1:  # torchvision rejects empty boxes
                        boxes.append([x1, y1, x2, y2])
        imgs.append(torch.from_numpy(img).permute(2, 0, 1).contiguous())
        b = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        targets.append(
            {"boxes": b, "labels": torch.ones(len(b), dtype=torch.int64)}
        )  # 1 = human
    print(
        "%s: %d images, %d people"
        % (split, len(imgs), sum(len(t["boxes"]) for t in targets))
    )
    return imgs, targets


def batch_to_device(imgs, targets, idx, flip):
    """Model input for a list of indices, with random horizontal flips as augmentation.
    (Identical to train_ssdlite.py.)"""
    ims, tgs = [], []
    for i in idx:
        im, t = imgs[i].float() / 255.0, dict(targets[i])
        if flip and random.random() < 0.5:
            im = im.flip(-1)
            b = t["boxes"].clone()
            b[:, [0, 2]] = SIZE - t["boxes"][:, [2, 0]]
            t["boxes"] = b
        ims.append(im.to(device))
        tgs.append({k: v.to(device) for k, v in t.items()})
    return ims, tgs


@torch.no_grad()
def evaluate(model, imgs, targets):
    """Precision, recall, mAP50 and mAP50-95, computed like Ultralytics' validator.
    (Identical to train_ssdlite.py, so all SSDLite and YOLO numbers are comparable.)"""
    model.eval()
    model.score_thresh, model.detections_per_img = 0.001, 300
    iouv = torch.linspace(0.5, 0.95, 10)
    stats = []
    for s in range(0, len(imgs), BATCH):
        idx = list(range(s, min(s + BATCH, len(imgs))))
        ims, _ = batch_to_device(imgs, targets, idx, flip=False)
        for o, i in zip(model(ims), idx):
            pb, ps, gb = o["boxes"].cpu(), o["scores"].cpu(), targets[i]["boxes"]
            tp = np.zeros((len(pb), len(iouv)), dtype=bool)
            if len(pb) and len(gb):
                iou = box_iou(gb, pb)
                for j, thr in enumerate(iouv):
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
    r = ap_per_class(tp, conf, pred_cls, target_cls)
    p, rec, ap = r[2], r[3], r[5]
    return float(p.mean()), float(rec.mean()), float(ap[:, 0].mean()), float(ap.mean())


train_imgs, train_targets = load_split("train")
val_imgs, val_targets = load_split("valid")

# Load the pruned model: its smaller layers, its SARD anchors and its weights are kept.
model = torch.load(START, weights_only=False).to(device)
n_before = sum(p.numel() for p in model.parameters())
print("pruned model %s: %d parameters" % (START, n_before))

# Measure how much pruning hurt before any retraining, for the report.
p, r, m50, m5095 = evaluate(model, val_imgs, val_targets)
print("before retraining: val mAP50 %.3f  mAP50-95 %.3f" % (m50, m5095))

opt = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.01,
    momentum=0.9,
    weight_decay=4e-5,
)  # same as for p00
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

os.makedirs("results/ssdlite_train", exist_ok=True)
log = open("results/ssdlite_train/ssdlite-sard_%s_results.csv" % TAG, "w")
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
        loss = sum(model(ims, tgs).values())  # box loss + classification loss
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
    if m50 > best:
        best, best_epoch = m50, epoch
        torch.save(copy.deepcopy(model).cpu(), OUT)
    elif epoch - best_epoch >= PATIENCE:
        print("no improvement for %d epochs, stopping" % PATIENCE)
        break

print("best epoch %d, validation mAP50 %.3f" % (best_epoch, best))

# Final check and test: reload the saved best model, confirm it is still the pruned size.
model = torch.load(OUT, weights_only=False).to(device)
n_after = sum(p.numel() for p in model.parameters())
print(
    "parameters before training %d, after %d  (%s)"
    % (
        n_before,
        n_after,
        "pruning kept" if n_after == n_before else "WARNING: size changed",
    )
)
test_imgs, test_targets = load_split("test")
p, r, m50, m5095 = evaluate(model, test_imgs, test_targets)
print(
    "test mAP50 %.3f  mAP50-95 %.3f  precision %.3f  recall %.3f" % (m50, m5095, p, r)
)
