# sar-edge/scripts/bench_onnx.py   run on the Pi from sar-edge/: python scripts/bench_onnx.py [pattern]
# Benchmarks ONNX models with ONNX Runtime on the CPU and checks them against references/.
# Per model: inference latency (mean, p95), FPS, end-to-end time per frame,
# average power during the timed loop, max temperature, and a sanity check.
import os, sys, glob, json, time, re, threading, subprocess

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root
import itertools
import cv2, numpy as np, onnxruntime as ort
import common as C

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "models/*.onnx"
WARMUP, RUNS = 10, 100
OUT = "results/pi"
os.makedirs(OUT, exist_ok=True)
anchors = np.load("models/ssdlite_anchors_320.npy")
images = sorted(glob.glob("data/smoke/*.jpg"))


def pmic_power():
    """Board power estimate in watts: sum of volts x amps over the PMIC rails."""
    out = subprocess.run(
        ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True
    ).stdout
    cur, volt = {}, {}
    for name, kind, val in re.findall(r"(\S+)_([AV]) \w+\(\d+\)=([\d.]+)", out):
        (cur if kind == "A" else volt)[name] = float(val)
    return sum(cur[n] * volt[n] for n in cur if n in volt)


def temp_c():
    out = subprocess.run(
        ["vcgencmd", "measure_temp"], capture_output=True, text=True
    ).stdout
    return float(re.findall(r"[\d.]+", out)[0])


class Sampler(threading.Thread):
    """Reads power and temperature every 0.5 s in the background."""

    def __init__(self):
        super().__init__(daemon=True)
        self.power, self.temps, self.stop = [], [], False

    def run(self):
        while not self.stop:
            self.power.append(pmic_power())
            self.temps.append(temp_c())
            time.sleep(0.5)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bench(path):
    name = os.path.basename(path)[:-5]
    is_ssd = name.startswith("ssdlite")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4  # the Pi 5 has 4 cores
    s = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0].name

    # preprocess every image once, so the timed loop measures inference only
    prepped = []
    for f in images:
        img = cv2.imread(f)
        prepped.append(
            (os.path.basename(f), img)
            + ((C.prep_ssd(img),) if is_ssd else (C.prep_yolo(img),))
        )

    for i in range(WARMUP):
        s.run(None, {inp: prepped[i % len(prepped)][2][0]})

    idle = np.mean([pmic_power() for _ in range(4)])
    sampler = Sampler()
    sampler.start()
    times = []
    t_start = time.perf_counter()
    # at least RUNS runs AND at least 20 s, so power gets about 40 samples
    for i in itertools.count():
        if i >= RUNS and time.perf_counter() - t_start >= 20:
            break
        x = prepped[i % len(prepped)][2][0]
        t = time.perf_counter()
        s.run(None, {inp: x})
        times.append((time.perf_counter() - t) * 1000)
    sampler.stop = True
    sampler.join()

    # one pass with pre- and postprocessing: end-to-end time and detections
    ref_name = name.replace("_int8_", "_fp32_")
    raw_ref = np.load("references/raw_%s.npz" % ref_name)
    dets, e2e, cos = {}, [], []
    for key, img, _ in prepped:
        t = time.perf_counter()
        if is_ssd:
            x, scale = C.prep_ssd(img)
            reg, logits = s.run(None, {inp: x})
            det = C.post_ssd(reg, logits, anchors, scale)
            cos.append(cosine(logits.ravel(), raw_ref[key + "_cls"].ravel()))
        else:
            x, r, pad = C.prep_yolo(img)
            out = s.run(None, {inp: x})[0]
            det = C.post_yolo_raw(out, r, pad)
            cos.append(cosine(out[0, 4:].ravel(), raw_ref[key][0, 4:].ravel()))
        e2e.append((time.perf_counter() - t) * 1000)
        dets[key] = det.round(2).tolist()

    with open("references/reference_%s.json" % ref_name) as fh:
        ref_bus = len(json.load(fh).get("bus.jpg", []))
    res = {
        "model": name,
        "runs": len(times),
        "threads": 4,
        "latency_mean_ms": round(float(np.mean(times)), 2),
        "latency_p95_ms": round(float(np.percentile(times, 95)), 2),
        "fps": round(1000.0 / float(np.mean(times)), 1),
        "end_to_end_mean_ms": round(float(np.mean(e2e)), 2),
        "power_idle_w": round(float(idle), 2),
        "power_load_w": (
            round(float(np.mean(sampler.power)), 2) if sampler.power else None
        ),
        "temp_max_c": max(sampler.temps) if sampler.temps else None,
        "bus_people": len(dets.get("bus.jpg", [])),
        "bus_people_reference": ref_bus,
        "score_cosine_vs_fp32_reference": round(float(np.mean(cos)), 4),
        "onnxruntime": ort.__version__,
    }
    with open(os.path.join(OUT, name + "_bench.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(OUT, name + "_detections.json"), "w") as fh:
        json.dump(dets, fh, indent=1)
    print(
        "%-24s %7.1f ms  p95 %7.1f  %5.1f FPS | %5.2f W (idle %4.2f) | %4.1f C | bus %d/%d | cos %.4f"
        % (
            name,
            res["latency_mean_ms"],
            res["latency_p95_ms"],
            res["fps"],
            res["power_load_w"] or 0,
            res["power_idle_w"],
            res["temp_max_c"] or 0,
            res["bus_people"],
            ref_bus,
            res["score_cosine_vs_fp32_reference"],
        ),
        flush=True,
    )


for path in sorted(glob.glob(PATTERN)):
    bench(path)
    time.sleep(5)  # let the board cool down between models
print(
    "throttled:",
    subprocess.run(
        ["vcgencmd", "get_throttled"], capture_output=True, text=True
    ).stdout.strip(),
)
