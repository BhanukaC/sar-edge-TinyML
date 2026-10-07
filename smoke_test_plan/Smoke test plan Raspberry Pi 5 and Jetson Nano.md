# Smoke test plan: Raspberry Pi 5 and Jetson Nano

Oct 7, 2026 · @Bhanuka Uyanage

## Purpose

Before Phase 3 starts on 9 October, we confirm that every model file in our experiment table builds, runs and gives sensible detections on its board, and we record exactly which file, format, runtime and precision that is.

We smoke test with COCO-pretrained weights, since COCO already has a person class. The fine-tuned SARD models later go through the same pipeline under the same file names.

The smoke test is done when all 32 rows in the manifest (12 YOLO, 12 SSDLite, 8 FOMO) are marked pass, or pass with a documented fallback.

## Setup status (7 October)

Both boards are installed, online and reachable over SSH; the remaining work is Step 1 on the laptop and a few board checks.

- [x] Jetson Nano: JetPack 4.6.1 flashed, first-boot setup done, SSH enabled, internet via the Mac, versions recorded with jetson\_release
- [x] Raspberry Pi 5: Debian 13 (Trixie) 64-bit, SSH working, venv `~/sar` with onnxruntime 1.30.0
- [ ] Jetson: `sudo apt update && sudo apt upgrade -y`, 4GB swap file, `trtexec --help` check
- [ ] Jetson: SSH from the Mac confirmed with `ssh nano@nano-desktop.local`, then run headless
- [ ] Pi: `vcgencmd get_throttled` shows `0x0` with the official 27W supply and active cooler
- [ ] Get a 5V 4A barrel-jack supply before any 10W (MAXN) Jetson runs
- [ ] Step 1 test kit on the laptop (below)
- [ ] Pull the Ultralytics JetPack 4 Docker image on a fast network (several GB, not over a hotspot)

## Boards and runtimes

Our Jetson is the original Jetson Nano Developer Kit 4GB (module P3448-0000; Maxwell GPU, quad-core Cortex-A57, 4GB LPDDR4), flashed with JetPack 4.6.1. Jetson versions below are recorded from jetson\_release on 7 October 2026.

|  | Raspberry Pi 5 | Jetson Nano 4GB |
| --- | --- | --- |
| OS | Raspberry Pi OS 64-bit, Debian 13 (Trixie), kernel 6.18.50 | JetPack 4.6.1 (L4T 32.7.1), Ubuntu 18.04, kernel 4.9.253-tegra |
| Python | 3.13.5, in a venv (system-wide pip is blocked) | 3.6 (too old for current Ultralytics and PyTorch) |
| GPU libraries | None | CUDA 10.2.300, cuDNN 8.2.1.32, VPI 1.2.3 |
| OpenCV | Installed with pip | 4.1.1, without CUDA (CPU preprocessing only, which is all we need) |
| YOLO and SSDLite runtime | ONNX Runtime 1.30.0 on CPU | TensorRT 8.2.1.8 |
| Precisions tested | FP32, INT8 (static QDQ) | FP32, FP16 |
| FOMO runtime | Edge Impulse Linux runner (.eim), float32 and int8 | Edge Impulse Jetson GPU build |
| Power reading | `vcgencmd pmic_read_adc`, USB-C meter as cross-check | `tegrastats` (POM\_5V\_IN, GPU, CPU rails), jtop 4.3.2 |
| Power supply | Official 27W USB-C | 5V 4A barrel jack, J48 jumper fitted |
| Power modes | Default | Mode 0 = 10W (MAXN), mode 1 = 5W; currently set to 5W |
| Login | `ssh pi@pi.local` | `ssh nano@nano-desktop.local` |

INT8 is not tested on the Jetson Nano because its Maxwell GPU has no INT8 acceleration. The 4GB memory is shared between CPU and GPU, so TensorRT builds need swap and a small workspace.

Open question: do we benchmark the Jetson at 10W, 5W or both? 5W is arguably closer to a solar UAV budget.

## Connecting to the boards

Both boards need internet for installs, so keep their Ethernet in the router and reach them from the laptop over the same network or over USB. Eduroam usually blocks device-to-device traffic, so use a home router or phone hotspot.

### Raspberry Pi 5

Our Pi is already set up: hostname `pi`, user `pi`, SSH enabled. Connect with `ssh pi@pi.local` from a laptop on the same network. The steps below are for setting up a fresh board.

1. Flash Raspberry Pi OS 64-bit with Raspberry Pi Imager. In Edit settings, set a hostname, username and password, and enable SSH under Services. Current releases have no default `pi` user.
2. From a Mac or Linux laptop, use the `.local` suffix:

```bash
ssh <username>@<hostname>.local
```

3. If the name does not resolve, list SSH devices on the network with `dns-sd -B _ssh._tcp` (Mac), or find the Pi's IP in the router's client list and use `ssh <username>@<ip>`.

### Jetson Nano

Our Nano is already set up: user `nano`, hostname `nano-desktop`, SSH enabled at boot, internet shared from the Mac over Ethernet (address in 192.168.2.x). Connect with `ssh nano@nano-desktop.local`, or `ssh nano@192.168.55.1` over the micro-USB cable. The steps below are for setting up a fresh board.

Flashing a fresh SD card (64GB or larger) from a Mac:

1. Download the Jetson Nano Developer Kit SD Card Image for JetPack 4.6.1 (`jetson-nano-jp461-sd-card-image.zip`, about 6GB) from NVIDIA's JetPack archive. It is the last SD image for the original Nano. Download on a normal network, not a hotspot.
2. Unzip in the terminal, since Archive Utility struggles with large files: `ditto -x -k jetson-nano-jp461-sd-card-image.zip .`
3. Flash the `.img` with Raspberry Pi Imager: Choose OS, Use custom, select the image, choose the SD card, and answer No to OS customisation. Recent balenaEtcher versions on macOS fail with a `requestMetadata is not a function` error, so we use Imager instead.
4. Eject when macOS says the disk is not readable. That is normal for a Linux card.
5. There is no default username or password; the first-boot wizard creates them. Choose MAXN if powered by the barrel jack, 5W if powered by micro-USB. The mode can be changed later with `nvpmodel`.

1) First boot needs a monitor, keyboard and mouse to finish the Ubuntu setup wizard (choose MAXN as power mode). Skip this if the TA gives us a board that is already set up, and ask for the username and password.
2) Enable SSH and note the IP:

```bash
sudo systemctl enable --now ssh
hostname -I
```

3. Recommended setup: Ethernet from the Nano to the router (for internet), power from the barrel jack, and a micro-USB data cable from the Nano to the laptop. The Nano then appears at a fixed address over USB:

```bash
ssh <username>@192.168.55.1
```

4. Without a micro-USB data cable, connect the laptop to the same router and SSH to the Nano's router IP.
5. If the Nano must sit next to the laptop with a direct Ethernet cable, share the laptop's Wi-Fi internet over Ethernet (macOS: System Settings, General, Sharing, Internet Sharing; Windows: Wi-Fi adapter Properties, Sharing tab). Without sharing, `apt` and `pip` will fail.
6. Check internet on the Nano before installing anything: `ping -c 3 google.com`.

### Working on the boards

- Copy files with `scp -r models/ <username>@<address>:~/sar-edge/`.
- Run long benchmarks inside `tmux` so they survive a dropped SSH session.
- VS Code Remote-SSH will probably not work on the Jetson, because Ubuntu 18.04's glibc is too old for recent VS Code. Use plain SSH and scp.

### Copying files by drag and drop

On a Mac, Cyberduck (free) gives a file browser for each board over the same SSH login:

1. Open Connection, and set the protocol dropdown to SFTP (SSH File Transfer Protocol). The default is FTP, which gives a "Connection refused" error.
2. Server `pi.local` (Pi) or `nano-desktop.local` (Jetson), port 22, with the board's username and password.
3. Save each connection as a bookmark, then drag files in and out.

For large folders such as SARD images, zip them first or use `scp -r`, which is faster than copying thousands of small files.

## Step 1: Prepare the test kit on a laptop or Colab

All exporting, pruning and INT8 calibration happens off the boards; the boards only build TensorRT engines and run inference. Step 1 ends with one shared folder holding every ONNX and EIM file, the test images, the shared Python code and the reference detections.

### 1.1 Environment, folder layout and input sizes

Use Python 3.10 or newer on a laptop or Colab. A GPU is not needed for the smoke test.

```bash
python -m venv .venv && source .venv/bin/activate
pip install ultralytics torch torchvision onnx onnxsim onnxruntime==1.30.0 torch-pruning opencv-python numpy roboflow python-dotenv onnxscript
```

Keep one repo with this layout, so every teammate and both boards use the same paths:

```
sar-edge/
  common.py          # preprocessing and postprocessing, used everywhere
  scripts/           # export, prune, quantize, reference scripts
  data/smoke/        # 20 test images + crowd.jpg
  data/calib/        # 100 calibration images
  models/            # all .onnx and .eim files
  references/        # reference_{model}.json
  engines/           # Jetson only: TensorRT engines, not committed
  results/pi/        # Pi outputs and logs
  results/jetson/    # Jetson outputs and logs
```

Every Python block in this doc starts with a comment giving its file path and how to run it. Always run scripts from the repo root (`cd sar-edge`, then `python scripts/<name>.py`), because all paths in them (`models/`, `data/`, `references/`) are relative to the root. Scripts that use `common.py` add the repo root to `sys.path` in their first lines, since `common.py` sits one folder above `scripts/`.

| File | Section | Makes |
| --- | --- | --- |
| `common.py` | 1.4 | Shared pre- and postprocessing (imported, not run) |
| `scripts/download_sard.py` | 1.2 | `data/SARD/` |
| `scripts/make_image_sets.py` | 1.2 | `data/smoke/`, `data/calib/` |
| `scripts/export_yolo.py` | 1.5 | `models/yolo26n_p00_fp32_640.onnx` |
| `scripts/check_onnx.py` | 1.5 | Printed checks only |
| `scripts/export_ssdlite.py` | 1.6 | `models/ssdlite_p00_fp32_320.onnx`, `models/ssdlite_anchors_320.npy` |
| `scripts/prune_ssdlite.py` | 1.7 | `models/ssdlite_{p30,p50}_fp32_320.onnx` |
| `scripts/prune_yolo.py` | 1.7 | `models/yolo26n_{p30,p50}_fp32_640.onnx` |
| `scripts/quantize_int8.py` | 1.8 | `models/*_int8_*.onnx` |
| `scripts/make_references.py` | 1.9 | `references/reference_*.json`, `references/raw_*.npz` |

Run them in the order of the table.

Fix the input sizes now and keep them for the whole project, since changing them later means redoing every row:

| Model | Input size | Reason |
| --- | --- | --- |
| YOLO | 640x640 | Ultralytics default |
| SSDLite-MobileNetV3 | 320x320 | Native size of ssdlite320 |
| FOMO | 320x320 | Small people in aerial images need resolution; FOMO output is 1/8 of input |

Open question: SARD people are small in high-resolution aerial images. If 320 is too coarse for SSDLite and FOMO after fine-tuning, we change it for all precisions at once and note it.

### 1.2 Test images and calibration images

Both scripts run once, on the laptop or Colab, from the repo root `sar-edge/`. Save them as `scripts/download_sard.py` and `scripts/make_image_sets.py`. The boards never download the dataset; they get `data/smoke/` through the repo.

Step A, download SARD into `data/SARD/` (an API key from your Roboflow account is needed; Kaggle also works):

The API key lives in a `.env` file in the repo root, which is never committed. Create it once on your machine:

```bash
# sar-edge/.env   (one line, no quotes or spaces)
ROBOFLOW_API_KEY=your_key_here
```

Then make sure Git ignores it, and commit an example file without the real key so teammates know what to create:

```bash
echo ".env" >> .gitignore
echo "ROBOFLOW_API_KEY=" > .env.example
git add .gitignore .env.example
```

The download script reads the key from `.env` with python-dotenv (`pip install python-dotenv`):

```python
# sar-edge/scripts/download_sard.py   run from sar-edge/: python scripts/download_sard.py
import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()   # reads sar-edge/.env
key = os.getenv("ROBOFLOW_API_KEY")
if not key:
    raise SystemExit("ROBOFLOW_API_KEY missing: create sar-edge/.env (see .env.example)")

rf = Roboflow(api_key=key)
ds = (rf.workspace("datasets-pdabr").project("sard-8xjhy")
        .version(16).download("yolov8", location="data/SARD"))
print("downloaded to", ds.location)
```

Run `git status` before your first commit: `.env` must not appear in the list. We use version 16 (see the table below) and record it in the manifest. Afterwards `data/SARD/` should contain `train/`, `valid/` and `test/`, each with `images/` and `labels/`.

The Roboflow project has no version 1; its versions differ in preprocessing ([version list](https://universe.roboflow.com/datasets-pdabr/sard-8xjhy/dataset/9), checked 7 October 2026):

| Version | Name | Images | Preprocessing |
| --- | --- | --- | --- |
| 17 | TILES\_3x3 | 5,755 | 3x3 tiles, stretched to 640x640 |
| 16 | ORIGINAL | 1,980 | None |
| 15 | Sard\_640x640 | 3,367 | Stretched to 640x640 |
| 12 | sard\_tiles\_grayscale | 8,347 | Tiles, grayscale, stretched to 640x640 |
| 10 | sard\_original | 1,981 | None |
| 9 | 3-d\_tiles | 9,268 | 3x3 tiles, 640x640, augmented (flip, rotation, blur, noise) |

For the smoke test we use version 16: untouched full-size frames are closest to what a UAV camera delivers, and our own letterbox in `common.py` handles resizing. The stretched versions distort the aspect ratio, and the augmented version mixes synthetic copies into the images.

The class in this dataset is called `human` (index 0), not `person`.

Open question for Phase 2: train on full frames (v16) or on 3x3 tiles (v17)? Tiles make small people larger, which helps accuracy, but the UAV then runs 9 inferences per frame, which multiplies latency and energy. This trade-off fits our research question and should be decided as a group.

Step B, pick the smoke and calibration images with a fixed seed, so everyone gets the same files:

```python
# sar-edge/scripts/make_image_sets.py   run from sar-edge/: python scripts/make_image_sets.py
import random, shutil, glob, os

random.seed(42)
val = sorted(glob.glob("data/SARD/valid/images/*.jpg"))
train = sorted(glob.glob("data/SARD/train/images/*.jpg"))
print(len(val), "val images,", len(train), "train images")   # both must be > 0
os.makedirs("data/smoke", exist_ok=True)
os.makedirs("data/calib", exist_ok=True)
for f in random.sample(val, 20):
    shutil.copy(f, "data/smoke/")
for f in random.sample(train, 100):
    shutil.copy(f, "data/calib/")
```

Step C, pick `crowd.jpg` automatically: the validation image with the most labelled people.

```python
# sar-edge/scripts/pick_crowd.py   run from sar-edge/: python scripts/pick_crowd.py
# Copies the validation image with the most people to data/smoke/crowd.jpg.
import glob, os, shutil

def count_people(label_file):
    # one non-empty line per labelled person in YOLO label files
    with open(label_file) as f:
        return sum(1 for line in f if line.strip())

labels = glob.glob("data/SARD/valid/labels/*.txt")
best = max(labels, key=count_people)
name = os.path.splitext(os.path.basename(best))[0]
img = glob.glob("data/SARD/valid/images/" + name + ".*")[0]   # .jpg or .png
shutil.copy(img, "data/smoke/crowd.jpg")
print("crowd.jpg =", os.path.basename(img), "with", count_people(best), "people")
```

Record the printed file name and person count. Afterwards `ls data/smoke | wc -l` should print 21.

Step D, add one everyday photo as a pipeline check. The COCO-pretrained models find almost nobody in SARD frames (tested 7 October: YOLO26n gave 0 people on crowd.jpg, max person score 0.08, the same as Ultralytics' own predict). A smoke test where every output is empty proves little, so we add Ultralytics' bus.jpg, where the COCO model finds 4 people:

```bash
python -c "from ultralytics.utils import ASSETS; import shutil; shutil.copy(ASSETS / 'bus.jpg', 'data/smoke/bus.jpg')"
```

`data/smoke/` then holds 22 images. After fine-tuning on SARD, crowd.jpg takes over as the main check.

If it prints 0 images, check the folder names and the file extension (`.jpg` or `.png`) inside `data/SARD/` and adjust the two glob lines.

Add `data/SARD/` to `.gitignore`. The full dataset is too large for GitHub and every teammate can download it again; only `data/smoke/` and `data/calib/` are committed.

- Smoke images come from the validation split and calibration images from the training split, so they never overlap.
- Copy one validation image with four or more people to `data/smoke/crowd.jpg` by hand. It catches models that only find the most obvious person.
- Commit the file lists (`ls data/smoke > smoke_list.txt`) so the sets are documented.

### 1.3 File naming and the files to produce

Every file is named `{family}_{prune}_{precision}_{imgsz}.{ext}`, so the file name alone says which manifest row it belongs to.

| Field | Values |
| --- | --- |
| family | `yolo26n` (or the fallback, e.g. `yolo11n`), `ssdlite`, `fomo-a035`, `fomo-a010` |
| prune | `p00`, `p30`, `p50` (FOMO is always `p00`) |
| precision | `fp32`, `fp16`, `int8` |
| imgsz | `640`, `320` |
| ext | `.onnx` (laptop), `.engine` (built on the Jetson), `.eim` (Edge Impulse) |

The laptop produces 18 files. The Jetson builds its 12 `.engine` files itself from the FP32 ONNX files.

| Made on | Files | Count |
| --- | --- | --- |
| Laptop | `yolo26n_{p00,p30,p50}_fp32_640.onnx` and `..._int8_640.onnx` | 6 |
| Laptop | `ssdlite_{p00,p30,p50}_fp32_320.onnx` and `..._int8_320.onnx` | 6 |
| Edge Impulse | `fomo-{a035,a010}_p00_{fp32,int8}_320.eim` for the Pi, plus one Jetson build per alpha | 6 |
| Jetson | `{yolo26n,ssdlite}_{p00,p30,p50}_{fp32,fp16}_*.engine` | 12 |

The FP32 ONNX files are used twice: directly on the Pi, and as the source for both Jetson engines.

### 1.4 Shared preprocessing and postprocessing (common.py)

The most common reason INT8 or board results look wrong is that preprocessing differs between calibration, reference and board. So all of it lives in one file, `common.py`, which the laptop, the Pi and the Jetson import unchanged.

Rules for this file:

- It must run on Python 3.6, because that is the Jetson's system Python on JetPack 4.6.1 and the TensorRT 8.2 Python bindings only exist for it. The Ultralytics Docker image has a newer Python, but our TensorRT scripts run outside Docker, so no walrus operator, no `X | Y` type hints, no match statements.
- It only depends on `numpy` and `cv2`, which exist on all three machines.
- Thresholds are set once here: score 0.25, NMS IoU 0.45, and recorded in the manifest.
- Every function returns person boxes as `[x1, y1, x2, y2, score]` in original image pixels.

```python
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
CONF = 0.25          # minimum score for a detection to count as a person
IOU = 0.45           # boxes overlapping more than this are merged by NMS
PERSON_YOLO = 0      # person index in Ultralytics COCO (and human in our SARD model)
PERSON_TV = 1        # person index in torchvision COCO (index 0 is background)


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
    r = min(size / h, size / w)                      # scale so the longer side fits
    nh, nw = int(round(h * r)), int(round(w * r))    # new height and width
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2   # centre the image
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh))
    return out, r, (left, top)


def prep_yolo(img_bgr, size=640):
    """Input for YOLO: letterboxed, RGB, values 0 to 1, shape (1, 3, size, size)."""
    lb, r, pad = letterbox(img_bgr, size)
    x = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)[None], r, pad         # HWC -> CHW, add batch axis


def prep_ssd(img_bgr, size=320):
    """Input for SSDLite: stretched to size x size, RGB, values -1 to 1.

    torchvision's SSDLite was trained on stretched (not letterboxed) images
    normalised with mean 0.5 and std 0.5. Returns the scale per axis so boxes
    can be mapped back to the original image.
    """
    h, w = img_bgr.shape[:2]
    x = cv2.cvtColor(cv2.resize(img_bgr, (size, size)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5   # torchvision SSDLite normalisation
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
    order = scores.argsort()[::-1]          # highest score first
    keep = []
    while order.size:
        i = order[0]                        # best remaining box
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
    b[:, [0, 2]] -= pad[0]   # remove left padding from x1, x2
    b[:, [1, 3]] -= pad[1]   # remove top padding from y1, y2
    return b / r             # undo the scaling


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
    p = out[0].T                              # -> (8400, 4 + classes)
    s = p[:, 4 + PERSON_YOLO]                 # person score of each candidate
    p, s = p[s >= CONF], s[s >= CONF]         # drop weak candidates first (faster NMS)
    # convert centre/size to corner coordinates x1, y1, x2, y2
    b = np.stack([p[:, 0] - p[:, 2] / 2, p[:, 1] - p[:, 3] / 2,
                  p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2], 1)
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
    clip = np.log(1000.0 / 16)                  # torchvision's limit, stops exp() exploding
    # shift and scale each anchor by the predicted offsets
    cx = d[:, 0] / 10.0 * aw + ax               # box coder weights (10, 10, 5, 5)
    cy = d[:, 1] / 10.0 * ah + ay
    w = np.exp(np.minimum(d[:, 2] / 5.0, clip)) * aw
    h = np.exp(np.minimum(d[:, 3] / 5.0, clip)) * ah
    boxes = np.clip(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), 0, 320)
    # softmax over classes, then take the person probability
    e = np.exp(logits[0] - logits[0].max(1, keepdims=True))   # subtract max for stability
    s = (e / e.sum(1, keepdims=True))[:, PERSON_TV]
    boxes, s = boxes[s >= CONF], s[s >= CONF]
    k = nms(boxes, s)
    sx, sy = scale
    return np.hstack([boxes[k] * [sx, sy, sx, sy], s[k, None]])   # back to original pixels
```

After fine-tuning on SARD (one class, human), the human index is 0 for YOLO and 1 for SSDLite, because torchvision detectors keep index 0 for background (so SSDLite is trained with num\_classes=2 and cls\_logits becomes \[1, N, 2\]). The YOLO raw output becomes `(1, 5, 8400)`. Only the constants at the top change.

The exact YOLO26 output shape should be checked after export (1.5). `post_yolo_e2e` assumes the end-to-end `(1, 300, 6)` layout; if the export gives `(1, 84, 8400)`, use `post_yolo_raw` instead.

### 1.5 YOLO export to ONNX

YOLO26 (released January 2026) is the newest Ultralytics model available for download. YOLO27 was announced at YOLO Vision 2026 but is not released yet; we stay on YOLO26 for the whole project so results stay comparable ([Ultralytics models](https://docs.ultralytics.com/models/), [YOLO27 page](https://docs.ultralytics.com/models/yolo27)).

Export with a fixed input shape, batch size 1 and opset 12. TensorRT 8.2 on the Nano handles static shapes and older opsets best.

```python
# sar-edge/scripts/export_yolo.py   run from sar-edge/: python scripts/export_yolo.py
# Exports COCO-pretrained YOLO26n to models/yolo26n_p00_fp32_640.onnx
from ultralytics import YOLO
import shutil

m = YOLO("yolo26n.pt")   # COCO-pretrained, downloads automatically
path = m.export(format="onnx", imgsz=640, opset=12, simplify=True,
                dynamic=False, batch=1, half=False)
shutil.move(path, "models/yolo26n_p00_fp32_640.onnx")
```

Check the file before moving on:

```python
# sar-edge/scripts/check_onnx.py   run from sar-edge/: python scripts/check_onnx.py
# Checks an exported ONNX file: valid graph, input/output shapes, one inference.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so `import common` works
import onnx, onnxruntime as ort, cv2
from common import prep_yolo

onnx.checker.check_model("models/yolo26n_p00_fp32_640.onnx")
s = ort.InferenceSession("models/yolo26n_p00_fp32_640.onnx", providers=["CPUExecutionProvider"])
print([(i.name, i.shape) for i in s.get_inputs()])
print([(o.name, o.shape) for o in s.get_outputs()])
img = cv2.imread("data/smoke/crowd.jpg")
assert img is not None, "data/smoke/crowd.jpg missing: run scripts/pick_crowd.py first"
x, r, pad = prep_yolo(img)
print(s.run(None, {s.get_inputs()[0].name: x})[0].shape)
```

- Input should be `images` with shape `[1, 3, 640, 640]`.
- Write the output shape into the manifest notes, and choose `post_yolo_e2e` or `post_yolo_raw` in `common.py` accordingly.
- Open the file in [Netron](https://netron.app) once and note any unusual ops (for example `TopK`, `GatherElements`, `Mod`). Those are the usual suspects if TensorRT 8.2 refuses the model.
- If opset 12 export fails, try 13, then 11.
- Fallback model: run the same export for `yolo11n.pt` now as well, so a replacement is ready if YOLO26 fails on the Jetson. Name it `yolo11n_p00_fp32_640.onnx`.

Result on 7 October: our YOLO26n export has output `output0` with shape `[1, 84, 8400]`, the raw head (4 box values plus 80 COCO class scores for 8,400 candidates), not the end-to-end `[1, 300, 6]` layout. So all YOLO scripts use `post_yolo_raw`, which applies our own NMS. This is also the safer layout for TensorRT 8.2 on the Nano. Record the output shape and the Ultralytics version in the manifest.

Pipeline check against Ultralytics' own prediction:

```python
# sar-edge/scripts/debug_yolo.py   run from sar-edge/: python scripts/debug_yolo.py
# Compares our ONNX pipeline with Ultralytics' own prediction on two images.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import cv2, onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import ASSETS
from common import prep_yolo, post_yolo_raw, PERSON_YOLO

s = ort.InferenceSession("models/yolo26n_p00_fp32_640.onnx", providers=["CPUExecutionProvider"])
ref = YOLO("yolo26n.pt")

for path in [str(ASSETS / "bus.jpg"), "data/smoke/crowd.jpg"]:
    img = cv2.imread(path)
    x, r, pad = prep_yolo(img)
    out = s.run(None, {"images": x})[0]
    ours = post_yolo_raw(out, r, pad)
    theirs = ref.predict(path, conf=0.25, classes=[0], verbose=False)[0].boxes
    print(os.path.basename(path), img.shape[:2])
    print("  max person score in ONNX output:", round(float(out[0, 4 + PERSON_YOLO].max()), 3))
    print("  our pipeline:", len(ours), "people | Ultralytics:", len(theirs), "people")
```

Result on 7 October: bus.jpg 4 people in both (max score 0.855); crowd.jpg 0 in both (max score 0.082). Our YOLO pipeline matches Ultralytics.

### 1.6 SSDLite export without postprocessing

The full torchvision model contains its own resizing, normalisation, anchor decoding and NMS, with dynamic shapes. TensorRT 8.2 is likely to reject those parts. So we export only the network (backbone and head), save the anchors to a file, and do the rest with `post_ssd` in `common.py`.

```python
# sar-edge/scripts/export_ssdlite.py   run from sar-edge/: python scripts/export_ssdlite.py
# Exports SSDLite backbone + head (no postprocessing) and saves the anchors.
import torch, numpy as np
from torchvision.models.detection import (ssdlite320_mobilenet_v3_large,
                                          SSDLite320_MobileNet_V3_Large_Weights)
from torchvision.models.detection.image_list import ImageList

full = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1).eval()

class SSDRaw(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head
    def forward(self, x):
        feats = list(self.backbone(x).values())
        out = self.head(feats)
        return out["bbox_regression"], out["cls_logits"]

net = SSDRaw(full).eval()
x = torch.zeros(1, 3, 320, 320)

# anchors depend only on input size and feature map sizes, so save them once
with torch.no_grad():
    feats = list(full.backbone(x).values())
    anchors = full.anchor_generator(ImageList(x, [(320, 320)]), feats)[0]
np.save("models/ssdlite_anchors_320.npy", anchors.numpy())

torch.onnx.export(net, x, "models/ssdlite_p00_fp32_320.onnx", opset_version=12,
                  input_names=["images"], output_names=["bbox_regression", "cls_logits"],
                  do_constant_folding=True,
                  dynamo=False)   # classic exporter: supports opset 12 for TensorRT 8.2
```

Then simplify and check it the same way as YOLO:

```bash
python -m onnxsim models/ssdlite_p00_fp32_320.onnx models/ssdlite_p00_fp32_320.onnx
```

Check that the decoding is right before exporting anything else: run the full torchvision model and the ONNX file plus `post_ssd` on `crowd.jpg`. The person boxes should match closely. If they do not, the problem is in the decoding or normalisation, not the export.

- Expected outputs: `bbox_regression` `[1, N, 4]` and `cls_logits` `[1, N, 91]`, where N is around 3,000 anchors at 320x320.
- `ssdlite_anchors_320.npy` goes to both boards together with the ONNX files.
- Pruning does not change the anchors, because it removes channels, not feature map positions. The same anchor file serves p00, p30 and p50.

Pipeline check against torchvision's complete model:

```python
# sar-edge/scripts/debug_ssdlite.py   run from sar-edge/: python scripts/debug_ssdlite.py
# Compares our ONNX + post_ssd pipeline with torchvision's full SSDLite model.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import cv2, numpy as np, onnxruntime as ort, torch
from torchvision.models.detection import (ssdlite320_mobilenet_v3_large,
                                          SSDLite320_MobileNet_V3_Large_Weights)
from ultralytics.utils import ASSETS
from common import prep_ssd, post_ssd, CONF, PERSON_TV

anchors = np.load("models/ssdlite_anchors_320.npy")
s = ort.InferenceSession("models/ssdlite_p00_fp32_320.onnx", providers=["CPUExecutionProvider"])
print([(o.name, o.shape) for o in s.get_outputs()], "anchors:", anchors.shape)
full = ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1).eval()

for path in [str(ASSETS / "bus.jpg"), "data/smoke/crowd.jpg"]:
    img = cv2.imread(path)
    # our pipeline: ONNX network + our own decoding and NMS
    x, scale = prep_ssd(img)
    reg, logits = s.run(None, {"images": x})
    ours = post_ssd(reg, logits, anchors, scale)
    # torchvision reference: full model with its own resize, normalisation and NMS
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255
    with torch.no_grad():
        o = full([t])[0]
    keep = (o["labels"] == PERSON_TV) & (o["scores"] >= CONF)
    print(os.path.basename(path), "| ours:", len(ours), "people | torchvision:", int(keep.sum()), "people")
    if len(ours):
        print("  our top box:", ours[ours[:, 4].argmax()].round(1))
    if keep.any():
        i = o["scores"][keep].argmax()
        print("  tv top box: ", o["boxes"][keep][i].numpy().round(1), round(float(o["scores"][keep][i]), 3))
```

Result on 7 October: outputs `[1, 3234, 4]` and `[1, 3234, 91]` with 3,234 anchors. bus.jpg 4 people in both, top box within 0.3 px of torchvision (score 0.989); crowd.jpg 0 in both. After onnxsim the model is 13.2 MiB with 583.2M MACs, and contains only ops TensorRT 8.2 supports (Conv, Clip, HardSigmoid, GlobalAveragePool, Reshape, Transpose, Concat, Add, Mul, Relu).

### 1.7 Structured pruning with Torch-Pruning

The smoke test only checks that pruned models export, build and run on both boards. Accuracy after pruning is not checked yet, because pruned models need fine-tuning to recover, which happens in Phase 2.

What "30% pruned" means must be the same for both families. We use Torch-Pruning's channel pruning ratio: `pruning_ratio=0.3` removes 30% of channels in each prunable layer. Parameters and MACs drop by more than that (roughly 50% for ratio 0.3, 75% for 0.5), so we always record the measured parameter count and MACs as well.

SSDLite (uses `net` from 1.6, rebuilt at the top of the script):

```python
# sar-edge/scripts/prune_ssdlite.py   run from sar-edge/: python scripts/prune_ssdlite.py
# Rebuilds the same network as export_ssdlite.py, then makes the p30 and p50 versions.
import copy, torch, torch_pruning as tp
from torchvision.models.detection import (ssdlite320_mobilenet_v3_large,
                                          SSDLite320_MobileNet_V3_Large_Weights)

class SSDRaw(torch.nn.Module):   # same wrapper as in export_ssdlite.py
    def __init__(self, m):
        super().__init__()
        self.backbone, self.head = m.backbone, m.head
    def forward(self, x):
        out = self.head(list(self.backbone(x).values()))
        return out["bbox_regression"], out["cls_logits"]

net = SSDRaw(ssdlite320_mobilenet_v3_large(
    weights=SSDLite320_MobileNet_V3_Large_Weights.COCO_V1).eval()).eval()


example = torch.randn(1, 3, 320, 320)
for ratio, tag in [(0.3, "p30"), (0.5, "p50")]:
    m = copy.deepcopy(net)
    base_macs, base_params = tp.utils.count_ops_and_params(m, example)
    pruner = tp.pruner.MetaPruner(
        m, example,
        importance=tp.importance.MagnitudeImportance(p=2),
        pruning_ratio=ratio,
        ignored_layers=[m.head],   # keep output channels = anchors x classes
        round_to=None)             # same meaning of the ratio for every layer size
    pruner.step()
    macs, params = tp.utils.count_ops_and_params(m, example)
    print(tag, params / base_params, macs / base_macs)
    torch.onnx.export(m.eval(), example, "models/ssdlite_%s_fp32_320.onnx" % tag,
                      opset_version=12, input_names=["images"],
                      output_names=["bbox_regression", "cls_logits"],
                      dynamo=False)   # classic exporter, opset 12
```

YOLO:

```python
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
        c = old.c                                    # channels in each half
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
        for k in ("f", "i", "type", "np"):            # Ultralytics routing info
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
        log("%s: rewrote %d C2f/C3k2 blocks, max output difference %.2e" % (tag, n, (before - after).abs().max()))

        # 2. prune
        for p in model.parameters():
            p.requires_grad_(True)
        example = torch.randn(1, 3, 640, 640)
        base_macs, base_params = tp.utils.count_ops_and_params(model, example)

        ignored = [model.model[-1]]                  # detection head
        ignored += [m for m in model.modules() if type(m).__name__ in ("C2PSA", "PSABlock", "Attention")]
        log("%s: ignoring head + %d attention modules" % (tag, len(ignored) - 1))

        pruner = tp.pruner.MetaPruner(
            model, example,
            importance=tp.importance.MagnitudeImportance(p=2),
            pruning_ratio=ratio,
            ignored_layers=ignored,
            round_to=None)
        log("%s: pruning" % tag)
        pruner.step()
        macs, params = tp.utils.count_ops_and_params(model, example)
        log("%s: params %.3f, MACs %.3f of baseline" % (tag, params / base_params, macs / base_macs))

        # 3. export
        log("%s: exporting ONNX" % tag)
        y.model = model
        path = y.export(format="onnx", imgsz=640, opset=12, simplify=True, dynamic=False, batch=1)
        shutil.move(path, "models/yolo26n_%s_fp32_640.onnx" % tag)
        log("%s: done" % tag)
```

Results on 8 October (Torch-Pruning 1.6.1, Ultralytics 8.4.174, PyTorch 2.14.1), measured at 640x640, as a share of the unpruned model:

| Model | Pruning ratio | Params kept | MACs kept | MACs | Params | ONNX size |
| --- | --- | --- | --- | --- | --- | --- |
| yolo26n\_p30 (640) | 0.3 | 67.0% | 64.1% | about 1.65G (3.3 GFLOPs) | 1,560,869 | 6.2 MB |
| yolo26n\_p50 (640) | 0.5 | 51.1% | 47.9% | about 1.2G (2.4 GFLOPs) | 1,153,152 | 4.7 MB |
| ssdlite\_p00 (320) | 0 | 100% | 100% | 583.2M |  | 13.2 MiB |
| ssdlite\_p30 (320) | 0.3 | 75.1% | 69.7% | 405.4M |  | 9.9 MiB |
| ssdlite\_p50 (320) | 0.5 | 62.0% | 54.8% | 319.5M |  | 8.2 MiB |

The detection head and the 5 attention modules stay unpruned, so the real reduction is smaller than the ratio suggests. This is why the report quotes measured parameters and MACs next to the ratio. SSDLite MACs come from onnxsim's summary; empty cells are still to be recorded. Measured on 8 October with round\_to=None and graph tracing at 640x640; YOLO GFLOPs from Ultralytics' export summary. With round\_to=8 (first run, 7 October) YOLO p30 had kept only 57% of MACs, which shows how channel rounding inflates the real pruning on a nano model.

Pruning check (8 October): without fine-tuning, pruned YOLO26n loses all detections, even at ratio 0.05 (top person score 0.855 to 0.000 on bus.jpg, with round\_to=8 and with round\_to=None). We verified this is not a bug: zeroing the same 247 channels in the unpruned model gives exactly the same output as the pruned model (difference 0.00e+00), so Torch-Pruning cuts and rewires the right channels. The ONNX files also match the pruned PyTorch model (difference 2.4e-04). One-shot pruning simply wrecks this narrow nano model until it is retrained; SSDLite, a larger and more redundant network, keeps some detections.

Consequences:

- The pruned YOLO files are valid for the smoke test. Build, latency, model size and power do not depend on how accurate the weights are.
- Fine-tuning after pruning is required, not optional. Order for Phase 2: fine-tune the full model on SARD, prune that fine-tuned model, then fine-tune the pruned model again to recover (train, prune, retrain). Plan the retraining step early: Ultralytics' trainer rebuilds the model from its config, which would undo the pruning, so the pruned model has to be passed to training explicitly (Torch-Pruning's YOLOv8 example shows one way).
- Report point: pruned accuracy before and after fine-tuning, for both families.

Verification script, kept for the report:

```python
# sar-edge/scripts/verify_pruning_yolo.py   run from sar-edge/: python scripts/verify_pruning_yolo.py [ratio]
# Checks Torch-Pruning's wiring: physically pruning channels must give the same output
# as keeping them but forcing them to zero. Same output = wiring correct.
import os, sys, copy
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(here))      # repo root, for common.py
sys.path.insert(0, here)                       # scripts/, for prune_yolo.py
import cv2, numpy as np, torch, torch.nn as nn, torch_pruning as tp
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2f
from ultralytics.utils import ASSETS
from prune_yolo import SplitC2f, first_tensor
from common import prep_yolo, PERSON_YOLO

RATIO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
img = cv2.imread(str(ASSETS / "bus.jpg"))
x, r, pad = prep_yolo(img)
xt = torch.from_numpy(x)


def run(model):
    model.eval()
    with torch.no_grad():
        return first_tensor(model(xt)).numpy()


model = YOLO("yolo26n.pt").model.float().eval()
for i, m in enumerate(model.model):
    if isinstance(m, C2f):
        model.model[i] = SplitC2f(m)
masked = copy.deepcopy(model)                  # same model; channels will be zeroed, not removed
name_of = {m: n for n, m in model.named_modules()}
masked_mods = dict(masked.named_modules())

for p in model.parameters():
    p.requires_grad_(True)
ignored = [model.model[-1]]
ignored += [m for m in model.modules() if type(m).__name__ in ("C2PSA", "PSABlock", "Attention")]
pruner = tp.pruner.MetaPruner(model, torch.randn(1, 3, 640, 640),
                              importance=tp.importance.MagnitudeImportance(p=2),
                              pruning_ratio=RATIO, ignored_layers=ignored, round_to=None)

n_zeroed = 0
for group in pruner.step(interactive=True):
    # before removing, zero the same channels in the copy (BN weight and bias = 0 -> output 0)
    for dep, idxs in group:
        mod = dep.target.module
        if isinstance(mod, nn.BatchNorm2d):
            bn = masked_mods[name_of[mod]]
            ids = [getattr(i, "idx", i) for i in idxs]
            with torch.no_grad():
                bn.weight[ids] = 0
                bn.bias[ids] = 0
            n_zeroed += len(ids)
    group.prune()

out_p, out_m = run(model), run(masked)
print("ratio %.2f, %d channels zeroed in the copy" % (RATIO, n_zeroed))
print("pruned model : top person score %.3f" % out_p[0, 4 + PERSON_YOLO].max())
print("masked model : top person score %.3f" % out_m[0, 4 + PERSON_YOLO].max())
print("max output difference pruned vs masked: %.2e" % np.abs(out_p - out_m).max())
```

- Torch-Pruning's API has changed between versions (older releases used `ch_sparsity` instead of `pruning_ratio`). Pin the version in `requirements.txt` and record it.
- YOLO pruning is the less certain part. If `MetaPruner` fails on the split/chunk operations inside the C3k2 blocks, follow the workaround in Torch-Pruning's YOLOv8 example (replacing the block with a version without split), or prune `yolo11n` instead.
- Run each pruned ONNX file through the same checks as 1.5 (checker, shapes, one inference).
- Record parameter count, MACs and file size for every prune level in the manifest. These are numbers we report anyway.

Open question: do we keep per-layer ratio 0.3 and 0.5, or tune global pruning so the parameter reduction is exactly 30% and 50%? Decide before Phase 2 and state it in the report.

### 1.8 INT8 static quantization for the Pi

Static quantization runs the 100 calibration images through the FP32 model, records the value range of every tensor, and stores weights and activations as 8-bit integers. The calibration images must go through exactly the same preprocessing as at inference, which is why the reader below calls `common.py`.

```python
# sar-edge/scripts/quantize_int8.py   run from sar-edge/: python scripts/quantize_int8.py
# Makes the INT8 ONNX files for the Pi from the FP32 files, calibrated on data/calib/.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so `import common` works
import glob, re, cv2, onnx
from onnx import version_converter
from onnxruntime.quantization import (CalibrationDataReader, quantize_static,
                                      QuantFormat, QuantType, CalibrationMethod)
from onnxruntime.quantization.shape_inference import quant_pre_process
from common import prep_yolo, prep_ssd


class SarReader(CalibrationDataReader):
    """Feeds the calibration images, preprocessed exactly as at inference."""
    def __init__(self, prep, input_name="images"):
        self.files = iter(sorted(glob.glob("data/calib/*.jpg")))
        self.prep, self.name = prep, input_name
    def get_next(self):
        f = next(self.files, None)
        if f is None:
            return None
        return {self.name: self.prep(cv2.imread(f))[0]}


def yolo_head_nodes(path):
    """Non-conv nodes of the YOLO detection head: box decoding, sigmoid, final concat.

    They put pixel coordinates (0-640) and scores (0-1) into one tensor. With one
    INT8 scale for both, every score rounds to 0, so these nodes stay FP32.
    """
    g = onnx.load(path).graph
    idx = max(int(m.group(1)) for n in g.node
              for m in [re.match(r"/model\.(\d+)/", n.name)] if m)
    head = "/model.%d/" % idx                   # the last block is the detection head
    return [n.name for n in g.node if n.name.startswith(head) and n.op_type != "Conv"]


def to_int8(src, dst, prep, keep_head_fp32=False):
    # Per-channel INT8 needs opset 13 (DequantizeLinear only has 'axis' from opset 13).
    # Our FP32 files are opset 12 for the Jetson's TensorRT 8.2, but INT8 only runs
    # on the Pi with ONNX Runtime, so we upgrade a copy to opset 13 here.
    up = src.replace(".onnx", "_op13.onnx")
    onnx.save(version_converter.convert_version(onnx.load(src), 13), up)
    tmp = src.replace(".onnx", "_prep.onnx")
    quant_pre_process(up, tmp)                  # shape inference + graph cleanup
    os.remove(up)
    exclude = yolo_head_nodes(tmp) if keep_head_fp32 else []
    try:
        quantize_static(tmp, dst,
                        calibration_data_reader=SarReader(prep),   # new reader per model
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True,
                        calibrate_method=CalibrationMethod.MinMax,
                        nodes_to_exclude=exclude)
    finally:
        os.remove(tmp)                          # the _prep file is only an intermediate
    print("made", dst, "(%d head nodes kept FP32)" % len(exclude))


for tag in ["p00", "p30", "p50"]:
    to_int8("models/yolo26n_%s_fp32_640.onnx" % tag, "models/yolo26n_%s_int8_640.onnx" % tag,
            prep_yolo, keep_head_fp32=True)
    to_int8("models/ssdlite_%s_fp32_320.onnx" % tag, "models/ssdlite_%s_int8_320.onnx" % tag, prep_ssd)
```

First run on 7 October, with the whole YOLO model in INT8: yolo26n\_p00 found 0 people on bus.jpg instead of 4, although the overall output cosine was 0.998. The YOLO output tensor mixes box coordinates (0 to 640) with scores (0 to 1); one INT8 scale of about 2.5 per step rounds every score to 0. Keeping the head's non-conv nodes in FP32 (`keep_head_fp32=True`) is the fix. Verified on 8 October: with 55 head nodes kept in FP32, yolo26n\_p00 INT8 finds the 4 people again.

INT8 vs FP32 on the laptop, 8 October (COCO weights, 22 smoke images):

| Model | bus.jpg people FP32 to INT8 | Cosine boxes | Cosine scores |
| --- | --- | --- | --- |
| yolo26n\_p00 | 4 to 4 | 0.9980 | 0.9175 |
| yolo26n\_p30 | 0 to 0 | 0.9999 | 0.6992 |
| yolo26n\_p50 | 0 to 0 | 1.0000 | 0.8910 |
| ssdlite\_p00 | 4 to 4 | 0.9865 | 0.9603 |
| ssdlite\_p30 | 6 to 5 | 0.9949 | 0.9982 |
| ssdlite\_p50 | 2 to 2 | 0.9950 | 0.9982 |

The score cosine covers all 80 classes at all 8,400 positions, almost all of them close to zero, where INT8 rounding noise dominates. That makes it pessimistic for YOLO, and meaningless for the unfine-tuned pruned YOLO models whose scores are all low. The decisive check for the smoke test is that the unpruned models find the same people. INT8 accuracy is measured properly with mAP after fine-tuning, and calibration is redone then with the fine-tuned weights.

- Delete the `_prep.onnx` files afterwards; they are intermediate.
- The reader is an iterator that gets used up, so create a new one for every model (the function above does).
- Preview INT8 on the laptop before it goes to the Pi: run the INT8 file with ONNX Runtime on `crowd.jpg` and compare with FP32. If it already fails here, fix it here.
- If INT8 accuracy collapses: first try `CalibrationMethod.Percentile`, then exclude the last detection-head nodes from quantization with `nodes_to_exclude=[...]` (node names from Netron). Record whatever we changed.
- Record the ONNX Runtime version used for quantization. The Pi should run the same or a newer version.

#### Checking INT8 against FP32 on the laptop

Before the INT8 files go to the Pi, compare each one with its FP32 version on `data/smoke/`: people found on bus.jpg, and how similar the raw outputs are.

```python
# sar-edge/scripts/compare_int8.py   run from sar-edge/: python scripts/compare_int8.py
# Compares each INT8 model with its FP32 version on data/smoke/: people found on
# bus.jpg, and how similar the raw outputs are (cosine, 1.0 = identical).
# Boxes and scores are compared separately: box values are large (pixels), so a
# combined cosine can look perfect while every score is broken.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
import glob
import cv2, numpy as np, onnxruntime as ort
import common as C

anchors = np.load("models/ssdlite_anchors_320.npy")
images = sorted(glob.glob("data/smoke/*.jpg"))


def outputs(path):
    """Return {image: (person boxes, raw box values, raw score values)} for one model."""
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    res = {}
    for f in images:
        img = cv2.imread(f)
        key = os.path.basename(f)
        if "ssdlite" in path:
            x, scale = C.prep_ssd(img)
            reg, logits = s.run(None, {"images": x})
            res[key] = (C.post_ssd(reg, logits, anchors, scale), reg.ravel(), logits.ravel())
        else:
            x, r, pad = C.prep_yolo(img)
            out = s.run(None, {s.get_inputs()[0].name: x})[0]
            res[key] = (C.post_yolo_raw(out, r, pad), out[0, :4].ravel(), out[0, 4:].ravel())
    return res


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


for q in sorted(glob.glob("models/*_int8_*.onnx")):
    f = q.replace("_int8_", "_fp32_")
    a, b = outputs(f), outputs(q)
    cos_box = np.mean([cosine(a[k][1], b[k][1]) for k in a])
    cos_score = np.mean([cosine(a[k][2], b[k][2]) for k in a])
    print("%-22s bus.jpg: fp32 %d -> int8 %d people | cosine boxes %.4f, scores %.4f"
          % (os.path.basename(q)[:-5], len(a["bus.jpg"][0]), len(b["bus.jpg"][0]), cos_box, cos_score))
```

How to read the result:

| Cosine (boxes and scores, scores first) | Meaning | Action |
| --- | --- | --- |
| 0.99 or higher | INT8 is a faithful copy of FP32 | Good, send to the Pi |
| 0.95 to 0.99 | Some precision loss | Usable, note it in the manifest |
| Well below 0.95, or p00 finds far fewer than 4 people on bus.jpg | Quantization damaged the model | Try `CalibrationMethod.Percentile`, then exclude the last head layers with `nodes_to_exclude` |

The p00 models matter most here. For p30 and p50 only the cosine is meaningful, since their box counts are unreliable before fine-tuning.

### 1.9 Reference detections on the laptop

References are what each model outputs on the laptop with ONNX Runtime FP32. Step 4 compares every board run against them. We save two things per model: the final person boxes (JSON) and the raw output tensors (NPY).

The raw tensors matter for the pruned models. Without fine-tuning they may detect nobody, and "zero boxes on the laptop, zero on the board" proves nothing. Comparing raw outputs still shows whether the board computes the same thing.

```python
# sar-edge/scripts/make_references.py   run from sar-edge/: python scripts/make_references.py
# Runs every FP32 model on data/smoke/ and saves, per model:
#   references/reference_<model>.json  person boxes [x1, y1, x2, y2, score] per image
#   references/raw_<model>.npz         raw output tensors per image (for pruned models)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so `import common` works
import glob, json
import cv2, numpy as np, onnxruntime as ort
import common as C

os.makedirs("references", exist_ok=True)
anchors = np.load("models/ssdlite_anchors_320.npy")
images = sorted(glob.glob("data/smoke/*.jpg"))
assert images, "no images in data/smoke/: run make_image_sets.py and pick_crowd.py first"


def run(model_path):
    s = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    name = os.path.basename(model_path).replace(".onnx", "")
    boxes, raw = {}, {}
    for f in images:
        img = cv2.imread(f)
        key = os.path.basename(f)
        if name.startswith("ssdlite"):
            x, scale = C.prep_ssd(img)
            reg, logits = s.run(None, {"images": x})
            det = C.post_ssd(reg, logits, anchors, scale)
            raw[key + "_reg"], raw[key + "_cls"] = reg, logits
        else:
            x, r, pad = C.prep_yolo(img)
            out = s.run(None, {s.get_inputs()[0].name: x})[0]
            det = C.post_yolo_raw(out, r, pad)   # our export is [1, 84, 8400], see 1.5
            raw[key] = out
        boxes[key] = det.round(2).tolist()

    with open("references/reference_%s.json" % name, "w") as fh:
        json.dump(boxes, fh, indent=1)
    np.savez_compressed("references/raw_%s.npz" % name, **raw)

    # one-line summary: bus.jpg proves the pipeline for unpruned COCO models
    total = sum(len(v) for v in boxes.values())
    print("%-28s bus.jpg: %s people | all %d images: %d boxes"
          % (name, len(boxes.get("bus.jpg", [])), len(images), total))


for m in sorted(glob.glob("models/*_fp32_*.onnx")):
    if m.endswith("_prep.onnx"):   # skip leftovers from quantize_int8.py
        continue
    run(m)
```

Expected with COCO weights: `yolo26n_p00` and `ssdlite_p00` find 4 people on bus.jpg. The pruned models may find fewer or none, since they are not fine-tuned yet; for them the raw NPZ files are the reference.

- Look at the boxes for `crowd.jpg` drawn on the image (`cv2.rectangle`) for the unpruned models. If the reference itself is wrong, every later comparison is meaningless.
- References are made from FP32 only. INT8 and FP16 board runs are compared against the FP32 reference of the same prune level.
- Commit `references/` to the repo. The JSON files are small; the NPZ files may be a few MB each.

### 1.10 FOMO builds in Edge Impulse

FOMO has no COCO-pretrained person model we can export, so for the smoke test we train a quick one in Edge Impulse Studio on 300 SARD images. The goal is a working `.eim` file for each board, not accuracy. Edge Impulse puts preprocessing inside the `.eim`, so `common.py` is not used for FOMO. Button names in Studio may differ slightly from those below.

#### Step A: make the upload set (laptop)

Edge Impulse reads YOLO TXT labels directly, as long as each uploaded folder has `images/`, `labels/` and a class list (`data.yaml` or `classes.txt`) ([supported formats](https://docs.edgeimpulse.com/tools/specifications/data-annotation/object-detection)).

```python
# sar-edge/scripts/make_fomo_upload.py   run from sar-edge/: python scripts/make_fomo_upload.py
# Copies 300 SARD training images and 50 validation images, with their YOLO labels,
# into data/fomo_upload/ for uploading to Edge Impulse.
import glob, os, random, shutil

random.seed(42)

def copy_set(images, out):
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs(os.path.join(out, "labels"), exist_ok=True)
    for img in images:
        name = os.path.splitext(os.path.basename(img))[0]
        label = os.path.join(os.path.dirname(os.path.dirname(img)), "labels", name + ".txt")
        shutil.copy(img, os.path.join(out, "images"))
        if os.path.exists(label):
            shutil.copy(label, os.path.join(out, "labels"))

train = sorted(glob.glob("data/SARD/train/images/*.jpg"))
valid = sorted(glob.glob("data/SARD/valid/images/*.jpg"))
copy_set(random.sample(train, 300), "data/fomo_upload/train")
copy_set(random.sample(valid, 50), "data/fomo_upload/test")

# Edge Impulse maps class numbers to names from data.yaml or classes.txt,
# which must be inside each folder we upload
for part in ("train", "test"):
    shutil.copy("data/SARD/data.yaml", "data/fomo_upload/%s/data.yaml" % part)
    with open("data/fomo_upload/%s/classes.txt" % part, "w") as fh:
        fh.write("human\n")
print("done: 300 train, 50 test images in data/fomo_upload/")
```

Each folder then holds `images/` (jpg), `labels/` (txt), `data.yaml` and `classes.txt`. Add `data/fomo_upload/` to `.gitignore`; it can be recreated any time.

#### Step B: create the project

1. Log in at studio.edgeimpulse.com (the free plan is enough) and create a project, for example `sar-edge-fomo`.
2. Dashboard, Project info: set **Labeling method** to **Bounding boxes (object detection)**. Otherwise the boxes are ignored on upload.
3. Dashboard, Project info: set **Target device** to Raspberry Pi 5. This only affects Studio's latency estimates.
4. Dashboard, Collaborators: add the other group members.

#### Step C: upload the data

1. Data acquisition, Add data, Upload data. Upload mode **Select a folder**, choose `data/fomo_upload/train`, category **Training**, label **Leave data unlabeled** (the boxes come from the label files). Upload; if asked for the annotation format, choose **YOLO TXT**.
2. Repeat with `data/fomo_upload/test` and category **Test**.
3. Open three or four samples and check that the people have boxes labelled `human`.

#### Step D: design the impulse

1. Impulse design, Create impulse. Image data: 320 x 320, resize mode **Fit longest axis** (pads instead of stretching, like our YOLO letterbox).
2. Add processing block **Image** and learning block **Object Detection (Images)**. Save Impulse.
3. Image: colour depth **RGB**, save parameters, then **Generate features**.

#### Step E: train both alphas

1. Object detection: Choose a different model, **FOMO (Faster Objects, More Objects) MobileNetV2 0.35**. Training cycles 30, other settings default. Start training.
2. Write down the F1 score and the on-device estimates (latency, RAM, flash).
3. Dashboard, Versioning: **Store a new version** named `fomo-a035`.
4. Back in Object detection: choose **FOMO MobileNetV2 0.1**, train with the same settings, note F1 and estimates, and store version `fomo-a010`.

#### Step F: quick test

1. Model testing, **Classify all**: note the test result.
2. Live classification, upload `data/smoke/crowd.jpg`: note how many people FOMO marks (FOMO marks object centres, not boxes). This count is the FOMO reference for the board check in Step 4.
3. For the other alpha, restore its version from Versioning and repeat.

#### Step G: download the builds

For each version (restore it first from Versioning):

1. Deployment, search **Linux (AARCH64)**. Build once as **Quantized (int8)** and once as **Unoptimized (float32)**; each downloads an `.eim` file.
2. Search **NVIDIA Jetson** and build the option for **JetPack 4.6** (our Nano runs 4.6.1). Write down which precision options that page offers.
3. Rename and put in `models/`:

| Build | File name |
| --- | --- |
| alpha 0.35, Linux AARCH64, int8 | `fomo-a035_p00_int8_320.eim` |
| alpha 0.35, Linux AARCH64, float32 | `fomo-a035_p00_fp32_320.eim` |
| alpha 0.35, Jetson | `fomo-a035_p00_jetson_320.eim` |
| alpha 0.10, the same three | `fomo-a010_p00_..._320.eim` |

The `.eim` files are Linux programs and do not run on a Mac; the first real test is on the boards (Step 2.5 on the Pi, Step 3.7 on the Jetson).

#### Step H: record for the manifest

For each alpha: input 320 x 320, Fit longest axis, RGB, 30 training cycles, 300 training images, F1 score, Studio's latency estimate, and the crowd.jpg count.

- FOMO outputs centroids, not boxes, so it gets precision, recall and F1 instead of mAP, as in the proposal.
- The free plan limits training time per job; 300 images at 320 x 320 fits. Full training on SARD comes in Phase 2.

### 1.11 Sharing files, hashes and the Step 1 checklist

Everything goes into the group's GitHub repo, so both boards get files with one `git pull` and we always know which version was tested.

- The ONNX files are small (roughly 5 to 15 MB each) and fit under GitHub's 100 MB file limit. If the repo gets heavy, use Git LFS for `models/`.
- Never commit `.engine` files. They only work on the Jetson that built them, so we rebuild them there and record their hashes.
- Pin package versions with `pip freeze > requirements.txt` on the laptop.

Generate hashes after the last file is made. On a Mac, `sha256sum` does not exist; use `shasum`:

```bash
# Mac
shasum -a 256 models/* > models/hashes.txt
# Linux, Pi, Jetson, Colab
sha256sum models/* > models/hashes.txt
```

On each board, after `git pull`, run `sha256sum -c models/hashes.txt`. Every line should say OK, which proves the board runs exactly the files we exported.

Step 1 is done when:

- [ ] `data/smoke/` (20 images + `crowd.jpg`) and `data/calib/` (100 images) are committed
- [ ] `common.py` runs on Python 3.6 (test with `python3.6` or on the Jetson)
- [ ] 6 YOLO ONNX files (p00, p30, p50 in FP32 and INT8) pass the 1.5 checks
- [ ] Fallback `yolo11n_p00_fp32_640.onnx` exported
- [ ] 6 SSDLite ONNX files plus `ssdlite_anchors_320.npy` pass the checks, and decoding matches torchvision
- [ ] Parameter count, MACs and file size recorded for each prune level
- [ ] INT8 previewed on the laptop for p00 of both families
- [ ] Reference JSON and NPZ files exist for all 6 FP32 models and look right on `crowd.jpg`
- [ ] FOMO a035 and a010 `.eim` files for the Pi (int8, float32) and the Jetson, with crowd.jpg counts noted
- [ ] `requirements.txt` and `models/hashes.txt` committed

## Where each file goes

Every machine keeps the same repo, `sar-edge`, in the same place, so paths in scripts and commands work everywhere without editing. On both boards it lives at `~/sar-edge` (`/home/pi/sar-edge` on the Pi, `/home/nano/sar-edge` on the Jetson). The Pi's Python venv `~/sar` is separate and only holds packages.

| File or folder | Laptop | Raspberry Pi | Jetson Nano |
| --- | --- | --- | --- |
| `common.py`, `scripts/` | Write and commit here | `~/sar-edge/` | `~/sar-edge/` |
| `data/smoke/` (20 images + crowd.jpg) | Yes | Yes | Yes |
| `data/calib/` (100 images) | Yes, used for INT8 | Not needed | Not needed |
| `models/*_fp32_*.onnx` | Made here | Run directly | Source for the engines |
| `models/*_int8_*.onnx` | Made here | Run directly | Not needed |
| `models/ssdlite_anchors_320.npy` | Made here | Yes | Yes |
| `models/fomo-*_{fp32,int8}_320.eim` (Linux AARCH64 builds) | Downloaded here | Yes | Not needed |
| `models/fomo-*_jetson.eim` (Jetson GPU builds) | Downloaded here | Not needed | Yes |
| `models/hashes.txt` | Made here | Check with `sha256sum -c` | Check with `sha256sum -c` |
| `engines/*.engine` | Not used | Not used | Built here by trtexec, never committed |
| `references/` | Made here | Optional | Optional |
| `results/pi/`, `results/jetson/` | Collected here for analysis | Pi writes `results/pi/` | Jetson writes `results/jetson/` |

Getting files onto a board, either way works:

1. Git (preferred): `git clone <repo url> ~/sar-edge` once, then `cd ~/sar-edge && git pull` before every session. Then run `sha256sum -c models/hashes.txt`.
2. Cyberduck or scp: copy into the matching subfolder, for example the laptop's `models/` into `~/sar-edge/models/` on the board: `scp -r models/ pi@pi.local:~/sar-edge/`.

Getting results back: each board only writes into its own `results/<board>/` folder, so pushing from both boards never conflicts. Commit and push from the board, or copy `results/<board>/` back to the laptop with Cyberduck.

Result file names follow the model name plus the power mode, for example `results/jetson/yolo26n_p00_fp16_640_5W_trtexec.txt`, `..._tegrastats.log` and `..._detections.json`.

## Step 2: Raspberry Pi 5

1. Health check. Update the OS, confirm a 64-bit OS, and confirm no power or heat throttling:

```bash
sudo apt update && sudo apt full-upgrade -y && sudo reboot
uname -m                 # must print aarch64, otherwise re-flash 64-bit
vcgencmd get_throttled   # must print throttled=0x0
```

`0x50000` or `0x50005` means undervoltage (use the official 27W supply). `0x80000` or `0x80008` means it hit the temperature limit (check the active cooler).

2. Python environment:

```bash
python3 -m venv ~/sar && source ~/sar/bin/activate
pip install onnxruntime opencv-python-headless numpy
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

Done on our Pi: Python 3.13.5 with onnxruntime 1.30.0 in `~/sar`. Always activate the venv first (`source ~/sar/bin/activate`), because Debian 13 blocks system-wide `pip install`.

3. Run each ONNX file (YOLO and SSDLite, FP32 and INT8, unpruned and 50% pruned) with the shared timing script: batch size 1, 10 warm-up runs, then 100 timed runs on the test images. Log mean and p95 latency and save detections to JSON.
4. Measure power during the timed loop, in a second terminal:

```bash
watch -n 0.5 vcgencmd pmic_read_adc
```

Sum voltage times current across the rails for a board estimate. Record idle power first so we can report the difference. Cross-check with a USB-C inline meter if we have one.

5. FOMO:

```bash
curl -sL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt install -y nodejs
sudo npm install edge-impulse-linux -g --unsafe-perm
chmod +x fomo_a035_int8.eim
edge-impulse-linux-runner --model-file fomo_a035_int8.eim
```

The runner prints timing per frame. The `edge_impulse_linux` Python SDK lets us use the same timing script as the other models.

6. After every run, check `vcgencmd measure_temp` and `vcgencmd get_throttled`. If throttled is not `0x0`, redo that run.

## Step 3: Jetson Nano

Test the YOLO TensorRT build first. It is the riskiest item in the whole plan.

1. Record the exact software version and install jtop:

```bash
cat /etc/nv_tegra_release      # expect R32.7.x
sudo apt install python3-pip && sudo pip3 install jetson-stats && sudo reboot
jetson_release
```

2. Free memory for engine builds: add 4GB swap and boot without the desktop.

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo systemctl set-default multi-user.target && sudo reboot
```

Only switch off the desktop once `ssh nano@nano-desktop.local` works from the Mac, otherwise the screen shows only a text login. Turn the desktop back on with `sudo systemctl set-default graphical.target`.

3. Fix the power mode so results are repeatable, and record it:

```bash
sudo nvpmodel -m 0     # 0 = 10W MAXN, 1 = 5W
sudo nvpmodel -q
sudo jetson_clocks
```

4. Build and benchmark each ONNX file with trtexec. Always build engines on the Nano itself, since engines are tied to the device and TensorRT version.

```bash
/usr/src/tensorrt/bin/trtexec --onnx=models/yolo26n_p00_fp32_640.onnx \
    --saveEngine=engines/yolo26n_p00_fp16_640.engine --fp16 --workspace=1024
/usr/src/tensorrt/bin/trtexec --onnx=models/yolo26n_p00_fp32_640.onnx \
    --saveEngine=engines/yolo26n_p00_fp32_640.engine --workspace=1024
```

Record the mean GPU compute time and throughput from the summary. If a build runs out of memory, lower `--workspace` to 512.

5. For accuracy checks with Python, use the Ultralytics JetPack 4 Docker image, since Ultralytics no longer supports Python 3.6:

```bash
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4
sudo docker run -it --ipc=host --runtime=nvidia -v ~/sar-edge:/sar-edge \
    ultralytics/ultralytics:latest-jetson-jetpack4
```

For SSDLite, run the engine with the TensorRT Python bindings that ship with JetPack and apply our own decoding and NMS.

6. Log power and temperature during each timed run:

```bash
sudo tegrastats --interval 500 --logfile results/jetson/yolo26n_p00_fp16_640_5W_tegrastats.log
```

Use POM\_5V\_IN (total board input, current and average in mW). POM\_5V\_GPU and POM\_5V\_CPU give the split. Record idle power too.

7. FOMO: install the Edge Impulse tooling for Jetson and run the GPU build with `edge-impulse-linux-runner --model-file <file>.eim`. Record which precision the build actually uses.

## Step 4: Sanity check the outputs

A model can run fast and still output nothing useful, especially after INT8, so every file is compared against its laptop reference on the 20 images.

- Person detections per image are within about 1 of the reference count.
- Matched boxes have IoU above 0.8 with the reference. FP16 should be nearly identical; INT8 can be a little looser.
- The crowded test image still shows several people.

If INT8 loses most detections, check the calibration first (preprocessing mismatch or too few calibration images) before blaming the board.

With COCO-pretrained weights, bus.jpg is the image that proves the pipeline: every unpruned YOLO and SSDLite file must find the same 4 people there on every board. On the SARD images, compare the raw output tensors (from `references/raw_*.npz`) instead of box counts, since the box counts are mostly zero before fine-tuning.

## Step 5: Record everything in the manifest

Each board gets one environment snapshot, and each model file on each board gets one manifest row.

Environment snapshot, run once per board and commit to GitHub:

```bash
{ date; uname -a; head -2 /etc/os-release; python3 --version
  cat /etc/nv_tegra_release 2>/dev/null; nvpmodel -q 2>/dev/null
  pip list 2>/dev/null | grep -iE "onnx|torch|ultralytics|tensorrt|edge"
} > env_$(hostname).txt
sha256sum models/* > hashes_$(hostname).txt
```

Manifest columns (one row per model file per board, in a shared sheet):

| Column | Example |
| --- | --- |
| Board and power mode | Jetson Nano 4GB, 10W MAXN |
| OS, JetPack, runtime version | JetPack 4.6.4, TensorRT 8.2.1 |
| Model file and sha256 | yolo26n\_p00\_fp16\_640.engine, 3f9a... |
| Family, prune %, alpha | YOLO26n, 0% |
| Precision and input size | FP16, 640x640 |
| Export command and opset | opset 12, trtexec --fp16 |
| Status | pass, pass with fallback, fail |
| Error or note | YOLO26 failed at opset 12, used YOLO11n |
| Mean and p95 latency (ms), FPS |  |
| Idle and load power (W) |  |
| Max temperature, throttled? |  |
| Sanity check passed? | yes |

When all 32 rows pass, the model list is locked for Phase 3.

## Order of work and fallbacks

Do the riskiest checks first so there is time to switch models before Phase 3.

- [ ] Connect to both boards and run the health checks (Pi Step 2.1, Jetson Step 3.1 to 3.3)
- [ ] YOLO26n unpruned ONNX to TensorRT FP16 on the Jetson
- [ ] SSDLite unpruned ONNX to TensorRT FP16 on the Jetson
- [ ] 50% pruned YOLO and SSDLite build on the Jetson
- [ ] INT8 YOLO and SSDLite run and pass the sanity check on the Pi
- [ ] FOMO runs on both boards
- [ ] Remaining rows, then manifest complete

| Problem | Fallback |
| --- | --- |
| YOLO26 fails to build in TensorRT 8.2.1 | Retry with opset 11, then switch to YOLO11n or YOLOv8n and log the error |
| SSDLite export fails | Export without postprocessing, do decoding and NMS in Python |
| TensorRT build runs out of memory | Check swap is on, desktop off, lower `--workspace` to 512 |
| INT8 drops most detections on the Pi | Fix calibration preprocessing, use more calibration images |
| Pi throttles (get\_throttled not 0x0) | Official 27W supply, active cooler, redo the run |
| Jetson brownout or reboot under load | Barrel-jack 5V 4A supply with J48 jumper, not micro-USB |
| Jetson has no internet (`eth0` unavailable in `nmcli device status`) | Plug Ethernet into a router, or share the Mac's Wi-Fi over Ethernet (Internet Sharing), or Android USB tethering |
| `pip install` refused on the Pi (externally managed environment) | Activate the venv `~/sar` first |
| A package has no Python 3.13 build on the Pi | Create the venv with Python 3.11, e.g. `uv venv --python 3.11 ~/sar` |
| balenaEtcher error `requestMetadata is not a function` | Flash the unzipped `.img` with Raspberry Pi Imager, Use custom |
| Cyberduck says Connection refused | Set protocol to SFTP, port 22 |
| Edge Impulse Jetson build has no FP16 option | Record the precision it uses and note it in the report |
| SSH name does not resolve | Use `.local` on Mac, or the IP from `hostname -I` or the router client list |
