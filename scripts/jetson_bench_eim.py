# sar-edge/scripts/jetson_bench_eim.py   run on the Jetson in the Python 3.8 venv (~/sar38),
# from the repo folder: python scripts/jetson_bench_eim.py [pattern]
# Benchmarks the Jetson FOMO builds (.eim, GPU) like bench_eim.py does on the Pi:
# latency, FPS, power and GPU temperature from tegrastats, people on bus.jpg and crowd.jpg.
import os, sys, glob, json, time, re, threading, subprocess, itertools
import cv2, numpy as np
from edge_impulse_linux.image import ImageImpulseRunner

PATTERN = sys.argv[1] if len(sys.argv) > 1 else "models/fomo-*_p00_jetson_320.eim"
WARMUP, RUNS = 10, 100
OUT = "results/jetson"
os.makedirs(OUT, exist_ok=True)
images = sorted(glob.glob("data/smoke/*.jpg"))


class Tegra(threading.Thread):
    """Runs tegrastats and keeps board power (POM_5V_IN, W) and GPU temperature readings."""
    def __init__(self):
        super().__init__(daemon=True)
        self.power, self.temps = [], []
        # sudo: without it tegrastats leaves out the power sensor (POM_5V_IN)
        self.proc = subprocess.Popen(["sudo", "-n", "tegrastats", "--interval", "500"],
                                     stdout=subprocess.PIPE, text=True)
    def run(self):
        for line in self.proc.stdout:
            p = re.search(r"POM_5V_IN (\d+)/", line)
            t = re.search(r"GPU@([\d.]+)C", line)
            if p:
                self.power.append(int(p.group(1)) / 1000.0)
            if t:
                self.temps.append(float(t.group(1)))
    def stop(self):
        subprocess.run(["sudo", "-n", "pkill", "tegrastats"])   # sudo'd process needs sudo to stop
        self.proc.wait(timeout=5)
        self.join(timeout=2)


def power_mode():
    out = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout
    m = re.search(r"MAXN|5W", out)
    return m.group(0) if m else "unknown"


def features_for(runner, img_rgb):
    # apply the same resize mode as set in Studio (Fit longest axis); older SDKs lack this
    try:
        return runner.get_features_from_image_auto_studio_settings(img_rgb)[0]
    except AttributeError:
        return runner.get_features_from_image(img_rgb)[0]


def bench(path, mode):
    name = os.path.basename(path)[:-4]
    runner = ImageImpulseRunner(os.path.abspath(path))
    try:
        runner.init()
        feats = {}
        for f in images:
            img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)   # Edge Impulse expects RGB
            feats[os.path.basename(f)] = features_for(runner, img)
        keys = sorted(feats)
        for i in range(WARMUP):
            runner.classify(feats[keys[i % len(keys)]])

        idle = Tegra()
        idle.start()
        time.sleep(5)
        idle.stop()

        load = Tegra()
        load.start()
        times, ei_times = [], []
        t_start = time.perf_counter()
        # at least RUNS runs AND at least 20 s, so power gets about 40 samples
        for i in itertools.count():
            if i >= RUNS and time.perf_counter() - t_start >= 20:
                break
            t = time.perf_counter()
            out = runner.classify(feats[keys[i % len(keys)]])
            times.append((time.perf_counter() - t) * 1000)
            ei_times.append(out["timing"]["classification"])   # model time as reported by the .eim
        load.stop()

        counts = {k: len(runner.classify(feats[k])["result"].get("bounding_boxes", []))
                  for k in ("bus.jpg", "crowd.jpg")}
    finally:
        runner.stop()

    res = {
        "model": name, "power_mode": mode, "runs": len(times),
        "latency_mean_ms": round(float(np.mean(times)), 2),          # includes talking to the .eim process
        "latency_p95_ms": round(float(np.percentile(times, 95)), 2),
        "model_time_mean_ms": round(float(np.mean(ei_times)), 2),     # the .eim's own measurement
        "fps": round(1000.0 / float(np.mean(times)), 1),
        "power_idle_w": round(float(np.mean(idle.power)), 2) if idle.power else None,
        "power_load_w": round(float(np.mean(load.power)), 2) if load.power else None,
        "gpu_temp_max_c": max(load.temps) if load.temps else None,
        "bus_people": counts["bus.jpg"], "crowd_people": counts["crowd.jpg"],
    }
    with open(os.path.join(OUT, "%s_%s_bench.json" % (name, mode)), "w") as fh:
        json.dump(res, fh, indent=1)
    print("%-26s %s %6.1f ms (model %5.1f) p95 %6.1f  %5.1f FPS | %s W (idle %s) | %s C | bus %d crowd %d"
          % (name, mode, res["latency_mean_ms"], res["model_time_mean_ms"], res["latency_p95_ms"], res["fps"],
             res["power_load_w"], res["power_idle_w"], res["gpu_temp_max_c"],
             counts["bus.jpg"], counts["crowd.jpg"]), flush=True)


mode = power_mode()
for path in sorted(glob.glob(PATTERN)):
    if "_jetson_" not in path:                      # only the GPU builds belong on the Jetson
        continue
    bench(path, mode)
    time.sleep(5)                                    # let the board cool down between models
