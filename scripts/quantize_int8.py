# sar-edge/scripts/quantize_int8.py   run from sar-edge/: python scripts/quantize_int8.py
# Makes the INT8 ONNX files for the Pi from the FP32 files, calibrated on data/calib/.
import os, sys

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root, so `import common` works
import glob, re, cv2, onnx
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)
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
    idx = max(
        int(m.group(1))
        for n in g.node
        for m in [re.match(r"/model\.(\d+)/", n.name)]
        if m
    )
    head = "/model.%d/" % idx  # the last block is the detection head
    return [n.name for n in g.node if n.name.startswith(head) and n.op_type != "Conv"]


def to_int8(src, dst, prep, keep_head_fp32=False):
    # Per-channel INT8 needs opset 13 (DequantizeLinear only has 'axis' from opset 13).
    # Our FP32 files are opset 12 for the Jetson's TensorRT 8.2, but INT8 only runs
    # on the Pi with ONNX Runtime, so we upgrade a copy to opset 13 here.
    up = src.replace(".onnx", "_op13.onnx")
    onnx.save(version_converter.convert_version(onnx.load(src), 13), up)
    tmp = src.replace(".onnx", "_prep.onnx")
    quant_pre_process(up, tmp)  # shape inference + graph cleanup
    os.remove(up)
    exclude = yolo_head_nodes(tmp) if keep_head_fp32 else []
    try:
        quantize_static(
            tmp,
            dst,
            calibration_data_reader=SarReader(prep),  # new reader per model
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
        )
    finally:
        os.remove(tmp)  # the _prep file is only an intermediate
    print("made", dst, "(%d head nodes kept FP32)" % len(exclude))


for tag in ["p00", "p30", "p50"]:
    to_int8(
        "models/yolo26n-sard_%s_fp32_640.onnx" % tag,
        "models/yolo26n-sard_%s_int8_640.onnx" % tag,
        prep_yolo,
        keep_head_fp32=True,
    )
    # to_int8(
    #     "models/yolo26n_%s_fp32_640.onnx" % tag,
    #     "models/yolo26n_%s_int8_640.onnx" % tag,
    #     prep_yolo,
    #     keep_head_fp32=True,
    # )

    # to_int8(
    #     "models/ssdlite_%s_fp32_320.onnx" % tag,
    #     "models/ssdlite_%s_int8_320.onnx" % tag,
    #     prep_ssd,
    # )
    to_int8(
        "models/ssdlite-sard_%s_fp32_320.onnx" % tag,
        "models/ssdlite-sard_%s_int8_320.onnx" % tag,
        prep_ssd,
    )
