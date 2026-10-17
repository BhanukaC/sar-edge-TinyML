# sar-edge/scripts/eval_ssdlite_onnx.py   run from sar-edge/: python scripts/eval_ssdlite_onnx.py
#
# Measures mAP on the 197 SARD test images for the SARD SSDLite ONNX files (FP32 and INT8),
# with ONNX Runtime and common.py's preprocessing and decoding, exactly as on the Pi.
# Matching rule and AP computation are the same as in train_ssdlite.py (Ultralytics'
# box_iou and ap_per_class), so the numbers compare with YOLO's and with training.
import os, sys, glob

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root, for common.py
import numpy as np, cv2, torch, onnxruntime as ort
import common as C
from ultralytics.utils.metrics import ap_per_class, box_iou

anchors = np.load(C.ssd_anchor_file("ssdlite-sard"))  # the SARD anchors
images = sorted(glob.glob("data/SARD/test/images/*.jpg"))
iouv = torch.linspace(0.5, 0.95, 10)  # the 10 IoU thresholds of mAP50-95


def gt_boxes(f, w, h):
    """Labelled people of one image, in original pixels (x1, y1, x2, y2)."""
    label = f.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    boxes = []
    if os.path.exists(label):
        for line in open(label):
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = (
                    float(p[1]) * w,
                    float(p[2]) * h,
                    float(p[3]) * w,
                    float(p[4]) * h,
                )
                boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)


def match(pred, gt):
    """Correct detections at each IoU threshold, same rule as Ultralytics' validator."""
    tp = np.zeros((len(pred), len(iouv)), dtype=bool)
    if len(pred) and len(gt):
        iou = box_iou(gt, torch.from_numpy(pred[:, :4]).float())
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
    return tp


for tag in ["p00", "p30", "p50"]:
    for prec in ["fp32", "int8"]:
        path = "models/ssdlite-sard_%s_%s_320.onnx" % (tag, prec)
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        stats = []
        for f in images:
            img = cv2.imread(f)
            h, w = img.shape[:2]
            x, scale = C.prep_ssd(img)
            reg, logits = sess.run(None, {inp: x})
            # very low threshold and up to 300 detections, as for mAP in training;
            # NMS at 0.55, torchvision's own SSD setting
            d = C.post_ssd(reg, logits, anchors, scale, conf=0.001, iou=0.55)
            d = d[np.argsort(-d[:, 4])[:300]]
            gt = gt_boxes(f, w, h)
            stats.append((match(d, gt), d[:, 4], np.zeros(len(d)), np.zeros(len(gt))))
        tp, conf, pred_cls, target_cls = [np.concatenate(s, 0) for s in zip(*stats)]
        r = ap_per_class(tp, conf, pred_cls, target_cls)
        p, rec, ap = r[2], r[3], r[5]
        print(
            "%s %s  mAP50 %.3f  mAP50-95 %.3f  precision %.3f  recall %.3f"
            % (tag, prec, ap[:, 0].mean(), ap.mean(), p.mean(), rec.mean())
        )
