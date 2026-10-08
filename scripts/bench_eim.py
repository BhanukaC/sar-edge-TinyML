# sar-edge/scripts/bench_eim.py   run on the Pi from sar-edge/: python scripts/bench_eim.py [pattern]
# Benchmarks Edge Impulse FOMO models (.eim) the same way bench_onnx.py does ONNX:
# latency, FPS, power, temperature, and people counted on bus.jpg and crowd.jpg.
import os, sys, glob, json, time, re, threading, subprocess
import itertools
import cv2, numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "models/fomo-*_p00_*_320.eim"
SKIP = "_jetson_"  # Jetson builds need CUDA and hang on the Pi
WARMUP, RUNS = 10, 100
OUT = "results/pi"
os.makedirs(OUT, exist_ok=True)
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


def features_for(runner, img_rgb):
    # apply the same resize mode as set in Studio (Fit longest axis); older SDKs lack this
    try:
        return runner.get_features_from_image_auto_studio_settings(img_rgb)[0]
    except AttributeError:
        return runner.get_features_from_image(img_rgb)[0]


def bench(path):
    name = os.path.basename(path)[:-4]
    runner = ImageImpulseRunner(os.path.abspath(path))
    try:
        runner.init()
        feats = {}
        for f in images:
            img = cv2.cvtColor(
                cv2.imread(f), cv2.COLOR_BGR2RGB
            )  # Edge Impulse expects RGB
            feats[os.path.basename(f)] = features_for(runner, img)
        keys = sorted(feats)

        for i in range(WARMUP):
            runner.classify(feats[keys[i % len(keys)]])

        idle = np.mean([pmic_power() for _ in range(4)])
        sampler = Sampler()
        sampler.start()
        times, ei_times = [], []
        t_start = time.perf_counter()
        # at least RUNS runs AND at least 20 s, so power gets about 40 samples
        for i in itertools.count():
            if i >= RUNS and time.perf_counter() - t_start >= 20:
                   break
            t = time.perf_counter()
            res = runner.classify(feats[keys[i % len(keys)]])
            times.append((time.perf_counter() - t) * 1000)
            ei_times.append(
                res["timing"]["classification"]
            )  # model time as reported by the .eim
        sampler.stop = True
        sampler.join()

        counts = {
            k: len(runner.classify(feats[k])["result"].get("bounding_boxes", []))
            for k in ("bus.jpg", "crowd.jpg")
        }
    finally:
        runner.stop()

    res = {
        "model": name,
        "runs": RUNS,
        "latency_mean_ms": round(
            float(np.mean(times)), 2
        ),  # includes talking to the .eim process
        "latency_p95_ms": round(float(np.percentile(times, 95)), 2),
        "model_time_mean_ms": round(
            float(np.mean(ei_times)), 2
        ),  # the .eim's own measurement
        "fps": round(1000.0 / float(np.mean(times)), 1),
        "power_idle_w": round(float(idle), 2),
        "power_load_w": (
            round(float(np.mean(sampler.power)), 2) if sampler.power else None
        ),
        "temp_max_c": max(sampler.temps) if sampler.temps else None,
        "bus_people": counts["bus.jpg"],
        "crowd_people": counts["crowd.jpg"],
    }
    with open(os.path.join(OUT, name + "_bench.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(
        "%-26s %6.1f ms (model %5.1f) p95 %6.1f  %5.1f FPS | %5.2f W (idle %4.2f) | %4.1f C | bus %d crowd %d"
        % (
            name,
            res["latency_mean_ms"],
            res["model_time_mean_ms"],
            res["latency_p95_ms"],
            res["fps"],
            res["power_load_w"] or 0,
            res["power_idle_w"],
            res["temp_max_c"] or 0,
            counts["bus.jpg"],
            counts["crowd.jpg"],
        ),
        flush=True,
    )


for path in sorted(glob.glob(PATTERN)):
    if SKIP in path:
        continue
    bench(path)
    time.sleep(5)  # let the board cool down between models
print(
    "throttled:",
    subprocess.run(
        ["vcgencmd", "get_throttled"], capture_output=True, text=True
    ).stdout.strip(),
)
