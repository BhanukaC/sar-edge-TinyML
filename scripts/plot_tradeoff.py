# sar-edge/scripts/plot_tradeoff.py   run from sar-edge/: python scripts/plot_tradeoff.py
# Accuracy versus energy per frame for every deployed model, from the final results.
# Left: YOLO26n and SSDLite (same mAP50 metric). Right: FOMO (Edge Impulse F1, a different
# metric, so it gets its own panel). Energy = total board power x time per frame.
import matplotlib.pyplot as plt

# (family, board/format, [(label, energy mJ, mAP50), ...])
MAP_RESULTS = [
    (
        "YOLO26n",
        "Jetson FP16",
        [("p00", 233, 0.917), ("p30", 224, 0.885), ("p50", 162, 0.790)],
    ),
    (
        "YOLO26n",
        "Pi INT8",
        [("p00", 398, 0.912), ("p30", 356, 0.880), ("p50", 271, 0.743)],
    ),
    (
        "YOLO26n",
        "Pi FP32",
        [("p00", 816, 0.917), ("p30", 568, 0.885), ("p50", 413, 0.790)],
    ),
    (
        "SSDLite",
        "Jetson FP16",
        [("p00", 107, 0.540), ("p30", 84, 0.515), ("p50", 65, 0.527)],
    ),
    (
        "SSDLite",
        "Pi FP32",
        [("p00", 185, 0.540), ("p30", 127, 0.515), ("p50", 90, 0.527)],
    ),
    (
        "SSDLite",
        "Pi INT8",
        [("p00", 109, 0.470), ("p30", 91, 0.433), ("p50", 67, 0.419)],
    ),
]
FOMO_RESULTS = [
    ("Pi INT8", [("a035", 34, 0.84), ("a010", 37, 0.79)]),
    ("Pi float32", [("a035", 49, 0.83), ("a010", 47, 0.80)]),
    ("Jetson float32", [("a035", 135, 0.83), ("a010", 127, 0.80)]),
]
MARKERS = {"YOLO26n": "o", "SSDLite": "s"}
COLOURS = {
    "Jetson FP16": "tab:green",
    "Pi INT8": "tab:blue",
    "Pi FP32": "tab:orange",
    "Pi float32": "tab:orange",
    "Jetson float32": "tab:green",
}

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2, 1]}
)

for family, setup, points in MAP_RESULTS:
    xs = [e for _, e, _ in points]
    ys = [m for _, _, m in points]
    ax1.plot(
        xs,
        ys,
        marker=MARKERS[family],
        color=COLOURS[setup],
        linestyle="-",
        alpha=0.8,
        label="%s, %s" % (family, setup),
    )
    for label, e, m in points:
        ax1.annotate(
            label, (e, m), textcoords="offset points", xytext=(5, 4), fontsize=8
        )
ax1.set_xscale("log")
ax1.set_xlabel("Energy per frame (mJ, log scale)")
ax1.set_ylabel("Test mAP50 (197 SARD images)")
ax1.set_title("YOLO26n (640) and SSDLite (320)")
ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=8, loc="center right")

for setup, points in FOMO_RESULTS:
    xs = [e for _, e, _ in points]
    ys = [f for _, _, f in points]
    ax2.plot(
        xs,
        ys,
        marker="^",
        color=COLOURS[setup],
        linestyle="none",
        label="FOMO, " + setup,
    )
    for label, e, f in points:
        ax2.annotate(
            label, (e, f), textcoords="offset points", xytext=(5, 4), fontsize=8
        )
ax2.set_xscale("log")
ax2.set_xlabel("Energy per frame (mJ, log scale)")
ax2.set_ylabel("Test F1 (Edge Impulse, cell-based)")
ax2.set_title("FOMO (320), different metric")
ax2.grid(alpha=0.3, which="both")
ax2.legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig("results/tradeoff_accuracy_energy.png", dpi=200)
print("saved results/tradeoff_accuracy_energy.png")
