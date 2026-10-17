#!/bin/bash
# sar-edge/scripts/jetson_bench_all.sh   run on the Jetson from the repo folder: bash scripts/jetson_bench_all.sh
# Benchmarks every engine for 20 s with trtexec while tegrastats logs power and temperature.
TRT=/usr/src/tensorrt/bin/trtexec
mkdir -p results/jetson
if ! ls engines/*.engine > /dev/null 2>&1; then
  echo "No engines in engines/: run scripts/jetson_build_all.sh first"
  exit 1
fi
MODE=$(sudo nvpmodel -q | grep -o "MAXN\|5W" | head -1)        # current power mode
for eng in ${1:-engines/*.engine}; do
  name=$(basename "$eng" .engine)
  echo "=== $name ($MODE)  $(date +%H:%M)"
  sleep 10                                                     # let the board settle
  # idle power for 5 s (output redirected: --logfile needs --start on JetPack 4.6)
  sudo tegrastats --interval 500 > "results/jetson/${name}_${MODE}_idle.log" &
  sleep 5; sudo pkill tegrastats; sleep 1
  # power and temperature during the 20 s benchmark
  sudo tegrastats --interval 500 > "results/jetson/${name}_${MODE}_tegrastats.log" &
  $TRT --loadEngine="$eng" --warmUp=2000 --duration=20 > "results/jetson/${name}_${MODE}_trtexec.txt" 2>&1
  sudo pkill tegrastats; sleep 1
done
python3 scripts/jetson_summary.py