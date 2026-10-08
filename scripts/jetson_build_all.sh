#!/bin/bash
# sar-edge/scripts/jetson_build_all.sh   run on the Jetson from the repo folder: bash scripts/jetson_build_all.sh
# Builds FP32 and FP16 TensorRT engines from every FP32 ONNX file. Engines only work on
# the device and TensorRT version that built them, so they are never committed.
TRT=/usr/src/tensorrt/bin/trtexec
mkdir -p engines results/jetson
for onnx in models/*_fp32_*.onnx; do
  base=$(basename "$onnx" .onnx)                    # e.g. yolo26n_p00_fp32_640
  for prec in fp32 fp16; do
    name=${base/_fp32_/_${prec}_}                   # e.g. yolo26n_p00_fp16_640
    if [ -f "engines/$name.engine" ]; then echo "skip $name (already built)"; continue; fi
    flag=""; [ "$prec" = "fp16" ] && flag="--fp16"
    echo "=== building $name  $(date +%H:%M)"
    $TRT --onnx="$onnx" --saveEngine="engines/$name.engine" $flag --workspace=1024 \
         > "results/jetson/${name}_build.txt" 2>&1
    if grep -q "PASSED" "results/jetson/${name}_build.txt"; then
      echo "    ok"
    else
      echo "    FAILED: see results/jetson/${name}_build.txt"
    fi
  done
done
