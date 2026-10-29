# Low-power person detection for UAV search and rescue

Course project (Tiny ML, group 29, Stockholm University, DSV) comparing three lightweight person-detection models on two edge devices, for a hypothetical solar-powered search-and-rescue UAV.

- **Models:** YOLO26n, SSDLite-MobileNetV3 and FOMO (Edge Impulse)
- **Compression:** structured pruning (p00, p30, p50) for YOLO26n and SSDLite, width multiplier (α 0.35 and 0.1) for FOMO, and quantization (INT8 on the Raspberry Pi 5, FP16 on the Jetson Nano)
- **Devices:** Raspberry Pi 5 and NVIDIA Jetson Nano 4GB (5 W mode)
- **Dataset:** SARD v16 from Roboflow (full 1920 × 1080 aerial frames, one class: `human`)
- **Simulation:** a Webots search flight that streams camera frames to the Jetson Nano in real time

Team: Bhanuka Chandiru Uyanage, Yueying Zhou, Yoke Cheng Lee.

## Main results

Best configuration of each model on each device (test set: 197 SARD images, 655 people).

| Model | Device | Configuration | Time/frame (ms) | FPS | Energy/frame (mJ) | Accuracy (test) |
|---|---|---|---|---|---|---|
| YOLO26n | Raspberry Pi 5 | p00 INT8 | 65.4 | 15.3 | 398 | mAP50 0.912, recall 0.859 |
| YOLO26n | Jetson Nano | p00 FP16 | 62.7 | 15.8 | 233 | mAP50 0.917, recall 0.863 |
| SSDLite | Raspberry Pi 5 | p50 FP32 | 13.9 | 71.7 | 90 | mAP50 0.527, recall 0.499 |
| SSDLite | Jetson Nano | p50 FP16 | 18.1 | 53.6 | 65 | mAP50 0.527, recall 0.499 |
| FOMO | Raspberry Pi 5 | α 0.35 INT8 | 9.9 | 101 | 34 | F1 0.84 |
| FOMO | Jetson Nano | α 0.35 FP32 | 50.9 | 20 | 135 | F1 0.83 |

FOMO's F1 comes from Edge Impulse Studio and is not directly comparable with mAP50. YOLO26n runs at 640 × 640, SSDLite and FOMO at 320 × 320.

Recommendation: YOLO26n p00 FP16 on the Jetson Nano when accuracy matters most; FOMO α 0.35 INT8 on the Raspberry Pi 5 for low-power scanning. In the Webots simulation, YOLO26n p00 FP16 on the Jetson found all 14 people, covered 94% of in-view person-frames, and ran the full loop at about 97 ms per frame (about 10 FPS).

## Repository layout

```
sar-edge-TinyML/
├── common.py              shared pre- and post-processing (Python 3.6 compatible, used on every machine)
├── data/
│   ├── sard.yaml          Ultralytics dataset config for SARD v16
│   ├── SARD/              the dataset (only test/ is in Git; download the rest)
│   └── smoke/             22 fixed benchmark images (20 SARD images, crowd.jpg, bus.jpg)
├── models/                ONNX models, SSDLite anchors, FOMO .eim builds, trained .pt models
├── references/            FP32 reference outputs used to check deployed models
├── results/               benchmark results, logs and figures per device
├── scripts/               training, export, benchmarking and evaluation scripts
├── sim/                   Webots world and controller
└── engines/               TensorRT engines (built on the Jetson, not in Git)
```

## Model files

File names follow `<model>_<prune>_<format>_<input size>`, for example `yolo26n-sard_p30_int8_640.onnx`.

- `yolo26n-sard_*` and `ssdlite-sard_*`: models trained on SARD (Phase 2, the reported results)
- `yolo26n_*` and `ssdlite_*` (without `-sard`): COCO-pretrained smoke-test models
- `fomo-a035-sard_*`, `fomo-a010-sard_*`: FOMO builds from Edge Impulse (`int8`, `fp32` for the Pi, `jetson` for the Jetson)
- `ssdlite-sard_anchors_320.npy`: SSDLite anchors sized for SARD's small people (the smoke-test models use `ssdlite_anchors_320.npy`)

## Software versions

| Machine | Versions |
|---|---|
| Laptop (training, export) | Python 3.13, PyTorch 2.14.1, Ultralytics 8.4.174 (pinned), Torch-Pruning 1.6.1, ONNX Runtime 1.30.0 |
| Raspberry Pi 5 | Debian 13, Python 3.13.5, ONNX Runtime 1.30.0, venv `~/sar` |
| Jetson Nano 4GB | JetPack 4.6.1, Python 3.6, CUDA 10.2, TensorRT 8.2.1, pycuda; FOMO in a Python 3.8 venv `~/sar38` |

On both boards, run `git config core.fileMode false` so file permissions do not block `git pull`.

## Setup

### Dataset

Create a `.env` file in the repository root with your Roboflow key (`ROBOFLOW_API_KEY=...`), then:

```bash
python scripts/download_sard.py        # SARD v16, full frames, YOLO format
python scripts/make_image_sets.py      # data/smoke (benchmark images) and data/calib (INT8 calibration)
python scripts/pick_crowd.py           # adds data/smoke/crowd.jpg
```

Splits: 1,387 train, 396 validation, 197 test images (4,598, 1,277 and 655 labelled people).

## Training and export (laptop)

### YOLO26n

```bash
python scripts/train_yolo.py 100                     # p00: fine-tune COCO weights on SARD
python scripts/prune_yolo_sard.py                    # p30 and p50 (Torch-Pruning, SplitC2f rewrite)
python scripts/train_pruned_yolo.py p30 50           # retrain the pruned models
python scripts/train_pruned_yolo.py p50 50
python scripts/verify_yolo_sard.py                   # parameters, test mAP, example pictures
python scripts/export_yolo_sard.py                   # ONNX, 640 x 640, opset 12
python scripts/quantize_int8.py                      # INT8 for the Pi (detection head kept in FP32)
python scripts/eval_yolo_onnx.py                     # test mAP of the deployed ONNX files
```

The C3k2 blocks are rewritten as `SplitC2f` (in `scripts/yolo_prune_utils.py`) so Torch-Pruning can follow their channels. The retraining script overrides the Ultralytics trainer's `get_model()` so the pruned network is not rebuilt at full size.

### SSDLite-MobileNetV3

```bash
python scripts/train_ssdlite.py 100                  # p00, with SARD-sized anchors
python scripts/prune_ssdlite_sard.py                 # p30 and p50
python scripts/train_pruned_ssdlite.py p30 50
python scripts/train_pruned_ssdlite.py p50 50
python scripts/export_ssdlite_sard.py                # ONNX and the SARD anchors
python scripts/quantize_int8.py                      # INT8 for the Pi
python scripts/eval_ssdlite_onnx.py                  # test mAP of the deployed ONNX files
```

SSDLite's default COCO anchors (smallest about 64 px) are far larger than SARD's people at 320 × 320 (median about 6 × 15 px), so `train_ssdlite.py` replaces them with anchors from 3% to 30% of the image. mAP is computed with Ultralytics' matching rule and AP functions, so it is comparable with YOLO26n.

### FOMO

FOMO was trained in Edge Impulse Studio (MobileNetV2 α 0.35 and α 0.1, 320 × 320 RGB, fit longest axis, 33 training cycles, learning rate 0.001). `scripts/make_fomo_upload_full.py` prepares the upload set. Deploy as **Linux (AARCH64)** (int8 and float32) for the Pi and **NVIDIA Jetson** for the Jetson, save the files as `models/fomo-<alpha>-sard_p00_<format>_320.eim`, and make them executable (`chmod +x`).

### References for the device checks

```bash
python scripts/make_references.py                    # FP32 reference outputs in references/
```

Commit `models/` and `references/`, then `git pull` on both boards.

## Benchmarking

### Raspberry Pi 5

```bash
source ~/sar/bin/activate
vcgencmd get_throttled                               # should be 0x0
python scripts/bench_onnx.py "models/yolo26n-sard_*.onnx"
python scripts/bench_onnx.py "models/ssdlite-sard_*.onnx"
python scripts/bench_eim.py "models/fomo-*-sard_p00_*_320.eim"
```

Each model runs for at least 20 s and 100 inferences after 10 warm-up runs, cycling through the 22 images in `data/smoke/`. Power is read from the PMIC (`vcgencmd pmic_read_adc`) every 0.5 s.

### Jetson Nano

```bash
sudo nvpmodel -m 1 && sudo jetson_clocks             # 5 W mode, fixed clocks (repeat after every reboot)
nohup bash scripts/jetson_build_all.sh "models/yolo26n-sard_*_fp32_*.onnx" > build.log 2>&1 &
python3 scripts/jetson_check.py                      # compare engine outputs with the references
sudo -v
bash scripts/jetson_bench_all.sh "engines/yolo26n-sard_*.engine"
source ~/sar38/bin/activate
python scripts/jetson_bench_eim.py "models/fomo-*-sard_p00_jetson_320.eim"
```

`jetson_build_all.sh` builds an FP32 and an FP16 engine from each ONNX file. `jetson_bench_all.sh` runs trtexec for 20 s per engine (random input) while tegrastats logs power, and `jetson_summary.py` collects the results in `results/jetson/summary.json`.

### FOMO test-set check on the devices

```bash
python scripts/eval_fomo_device.py                   # on each board: detections on the 197 test images
python scripts/score_centres.py 24 results/pi/fomo_test/*.json   # on the laptop: per-person scoring
```

Energy per frame is calculated as load power multiplied by time per frame (idle power not subtracted).

## Simulation (Webots)

A Mavic 2 Pro flies a lawnmower search pattern at 25 m over 14 pedestrians, starting and landing at a fixed home point, and sends every frame to the Jetson Nano for detection.

1. On the Jetson (system `python3`, with TensorRT and pycuda):
   ```bash
   sudo jetson_clocks
   python3 scripts/jetson_detect_server.py           # loads engines/yolo26n-sard_p00_fp16_640.engine, port 5000
   ```
2. On the Mac, open `sim/worlds/sar_search.wbt` in Webots (Python command set to the project's venv), check the `HOST` address in `sim/controllers/sar_patrol/sar_patrol.py`, then press Reset and Play.

The controller shows the annotated camera view live, and writes `results/sim/flight.mp4`, `frames.csv` and `summary.json`. `scripts/stream_test_client.py` streams SARD test images to the same server, to measure the end-to-end loop with real frames.

Notes: on recent macOS, Terminal, VS Code and Webots each need Local Network permission (System Settings, Privacy & Security, Local Network) to reach the Jetson. Webots R2025a does not start on macOS 27; use a newer nightly build.

## Known limitations

- FOMO is evaluated with Edge Impulse's cell-based F1, not mAP, and was benchmarked in float32 only on the Jetson.
- Jetson FP16 accuracy is taken as equal to FP32, based on matching outputs on two images.
- Jetson power for YOLO26n and SSDLite is averaged over the whole trtexec run, including engine loading, so their energy per frame is likely somewhat underestimated.
- Power comes from the boards' own sensors, not an external power meter.
- The Jetson was tested in 5 W mode only.

## References

- SARD: S. Sambolek and M. Ivasic-Kos, "Automatic Person Detection in Search and Rescue Operations Using Deep CNN Detectors", IEEE Access, 2021. Dataset on [Roboflow](https://universe.roboflow.com/datasets-pdabr/sard-8xjhy).
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26)
- A. Howard et al., "Searching for MobileNetV3", 2019, [arXiv:1905.02244](https://arxiv.org/abs/1905.02244)
- W. Liu et al., "SSD: Single Shot MultiBox Detector", ECCV 2016, [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)
- [Edge Impulse FOMO documentation](https://docs.edgeimpulse.com/studio/projects/learning-blocks/blocks/object-detection/fomo)
- G. Fang et al., "DepGraph: Towards Any Structural Pruning", CVPR 2023, [Torch-Pruning](https://github.com/VainF/Torch-Pruning)
- [Webots](https://cyberbotics.com/)
