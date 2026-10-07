# sar-edge/common.py   (repo root, NOT inside scripts/; copied to both boards)
# Shared preprocessing and postprocessing for all models and all machines.
# The laptop, the Raspberry Pi and the Jetson import this same file, so every
# model sees exactly the same input and its output is decoded the same way.
# Must stay Python 3.6 compatible (Jetson) and only depend on numpy and cv2.

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Settings used everywhere. Change them here only, and record them in the manifest.
# ---------------------------------------------------------------------------
CONF = 0.25  # minimum score for a detection to count as a person
IOU = 0.45  # boxes overlapping more than this are merged by NMS
PERSON_YOLO = 0  # person index in Ultralytics COCO (and human in our SARD model)
PERSON_TV = 1  # person index in torchvision COCO (index 0 is background)


# ---------------------------------------------------------------------------
# Preprocessing: turn a camera image (OpenCV BGR, any size) into model input.
# ---------------------------------------------------------------------------


def letterbox(img, size):
    """Resize to fit inside size x size without changing proportions.

    The free space is filled with grey (114), the same padding Ultralytics
    uses in training. Returns the padded image, the scale factor r and the
    padding offsets, which are needed later to map boxes back.
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)  # scale so the longer side fits
    nh, nw = int(round(h * r)), int(round(w * r))  # new height and width
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2  # centre the image
    out[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh))
    return out, r, (left, top)


def prep_yolo(img_bgr, size=640):
    """Input for YOLO: letterboxed, RGB, values 0 to 1, shape (1, 3, size, size)."""
    lb, r, pad = letterbox(img_bgr, size)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], r, pad  # HWC -> CHW, add batch axis


def prep_ssd(img_bgr, size=320):
    """Input for SSDLite: stretched to size x size, RGB, values -1 to 1.

    torchvision's SSDLite was trained on stretched (not letterboxed) images
    normalised with mean 0.5 and std 0.5. Returns the scale per axis so boxes
    can be mapped back to the original image.
    """
    h, w = img_bgr.shape[:2]
    x = (
        cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2RGB).astype(
            np.float32
        )
        / 255.0
    )
    x = (x - 0.5) / 0.5  # torchvision SSDLite normalisation
    return x.transpose(2, 0, 1)[None], (w / size, h / size)


# ---------------------------------------------------------------------------
# Non-maximum suppression: keep the best box among overlapping duplicates.
# ---------------------------------------------------------------------------


def nms(boxes, scores, iou=IOU):
    """Return indices of the boxes to keep.

    boxes: (N, 4) as x1, y1, x2, y2. scores: (N,).
    Repeatedly keep the highest-scoring box and drop every remaining box
    that overlaps it by more than iou.
    """
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]  # highest score first
    keep = []
    while order.size:
        i = order[0]  # best remaining box
        keep.append(i)
        rest = order[1:]
        # width and height of the intersection with every other box
        iw = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        ih = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = iw * ih
        # keep only boxes whose overlap (IoU) with box i is small enough
        order = rest[inter / (areas[i] + areas[rest] - inter + 1e-9) < iou]
    return np.array(keep, dtype=int)


def undo_letterbox(b, r, pad):
    """Map boxes from the 640x640 letterboxed image back to original pixels."""
    b = b.copy()
    b[:, [0, 2]] -= pad[0]  # remove left padding from x1, x2
    b[:, [1, 3]] -= pad[1]  # remove top padding from y1, y2
    return b / r  # undo the scaling


# ---------------------------------------------------------------------------
# Postprocessing: raw model output -> person boxes [x1, y1, x2, y2, score]
# in original image pixels. Every function returns an array of shape (K, 5).
# ---------------------------------------------------------------------------


def post_yolo_e2e(out, r, pad):
    """YOLO26 end-to-end output, shape (1, N, 6): x1, y1, x2, y2, score, class.

    YOLO26 already removes duplicates inside the model, so no NMS is needed;
    we only filter by score and class.
    """
    d = out[0]
    d = d[(d[:, 4] >= CONF) & (d[:, 5] == PERSON_YOLO)]
    return np.hstack([undo_letterbox(d[:, :4], r, pad), d[:, 4:5]])


def post_yolo_raw(out, r, pad):
    """YOLO11 / YOLOv8 output, shape (1, 4 + classes, 8400).

    Each of the 8400 columns is one candidate: centre x, centre y, width,
    height, then one score per class. These models need NMS afterwards.
    """
    p = out[0].T  # -> (8400, 4 + classes)
    s = p[:, 4 + PERSON_YOLO]  # person score of each candidate
    p, s = p[s >= CONF], s[s >= CONF]  # drop weak candidates first (faster NMS)
    # convert centre/size to corner coordinates x1, y1, x2, y2
    b = np.stack(
        [
            p[:, 0] - p[:, 2] / 2,
            p[:, 1] - p[:, 3] / 2,
            p[:, 0] + p[:, 2] / 2,
            p[:, 1] + p[:, 3] / 2,
        ],
        1,
    )
    k = nms(b, s)
    return np.hstack([undo_letterbox(b[k], r, pad), s[k, None]])


def post_ssd(reg, logits, anchors, scale):
    """SSDLite raw output -> person boxes.

    reg: (1, N, 4) box offsets relative to each anchor.
    logits: (1, N, C) class scores before softmax (C = 91 for COCO).
    anchors: (N, 4) default boxes x1, y1, x2, y2 in 320 px, from
             ssdlite_anchors_320.npy.
    This reproduces torchvision's own decoding, which we left out of the ONNX
    export because TensorRT 8.2 cannot run it.
    """
    # anchor width, height and centre
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + 0.5 * aw
    ay = anchors[:, 1] + 0.5 * ah
    d = reg[0]
    clip = np.log(1000.0 / 16)  # torchvision's limit, stops exp() exploding
    # shift and scale each anchor by the predicted offsets
    cx = d[:, 0] / 10.0 * aw + ax  # box coder weights (10, 10, 5, 5)
    cy = d[:, 1] / 10.0 * ah + ay
    w = np.exp(np.minimum(d[:, 2] / 5.0, clip)) * aw
    h = np.exp(np.minimum(d[:, 3] / 5.0, clip)) * ah
    boxes = np.clip(
        np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), 0, 320
    )
    # softmax over classes, then take the person probability
    e = np.exp(
        logits[0] - logits[0].max(1, keepdims=True)
    )  # subtract max for stability
    s = (e / e.sum(1, keepdims=True))[:, PERSON_TV]
    boxes, s = boxes[s >= CONF], s[s >= CONF]
    k = nms(boxes, s)
    sx, sy = scale
    return np.hstack(
        [boxes[k] * [sx, sy, sx, sy], s[k, None]]
    )  # back to original pixels
