# sar-edge/scripts/jetson_summary.py   run on the Jetson from ~/sar-edge: python3 scripts/jetson_summary.py
# Python 3.6 compatible. Reads trtexec and tegrastats logs from results/jetson/ and prints
# one line per engine and power mode; also saves results/jetson/summary.json.
import glob, json, os, re


def avg_power_w(path):
    """Average total board power (POM_5V_IN, current value in mW) from a tegrastats log."""
    if not os.path.exists(path):
        return None
    vals = [int(v) for v in re.findall(r"POM_5V_IN (\d+)/\d+", open(path).read())]
    return round(sum(vals) / len(vals) / 1000.0, 2) if vals else None


def max_temp(path):
    if not os.path.exists(path):
        return None
    vals = [float(v) for v in re.findall(r"GPU@([\d.]+)C", open(path).read())]
    return max(vals) if vals else None


rows = []
for f in sorted(glob.glob("results/jetson/*_trtexec.txt")):
    tag = os.path.basename(f)[: -len("_trtexec.txt")]  # <model>_<mode>
    text = open(f).read()
    # TensorRT 8.2 writes "GPU Compute: min: .., mean: .. ms, .. percentile: .. ms at 99%";
    # newer versions write "GPU Compute Time: .. mean = .. ms, .. percentile(99%) = .. ms"
    gpu = [l for l in text.splitlines() if "GPU Compute" in l and "Total" not in l]
    mean = re.search(r"mean\s*[:=]\s*([\d.]+) ms", gpu[-1]) if gpu else None
    p99 = (
        re.search(
            r"percentile\(99%\) = ([\d.]+) ms|percentile: ([\d.]+) ms at 99%", gpu[-1]
        )
        if gpu
        else None
    )
    thr = re.search(r"Throughput: ([\d.]+) qps", text)
    row = {
        "run": tag,
        "gpu_mean_ms": float(mean.group(1)) if mean else None,
        "gpu_p99_ms": float(p99.group(1) or p99.group(2)) if p99 else None,
        "fps": float(thr.group(1)) if thr else None,
        "power_idle_w": avg_power_w("results/jetson/%s_idle.log" % tag),
        "power_load_w": avg_power_w("results/jetson/%s_tegrastats.log" % tag),
        "gpu_temp_max_c": max_temp("results/jetson/%s_tegrastats.log" % tag),
    }
    rows.append(row)
    print(
        "%-30s %7s ms  p99 %7s  %6s FPS | %5s W (idle %5s) | %s C"
        % (
            tag,
            row["gpu_mean_ms"],
            row["gpu_p99_ms"],
            row["fps"],
            row["power_load_w"],
            row["power_idle_w"],
            row["gpu_temp_max_c"],
        )
    )

with open("results/jetson/summary.json", "w") as fh:
    json.dump(rows, fh, indent=1)
