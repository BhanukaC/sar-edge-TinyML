# sar-edge/scripts/jetson_detect_server.py   run on the Jetson from the repo folder:
#   python3 scripts/jetson_detect_server.py [engine] [port]
# Python 3.6 compatible (system python3, which has TensorRT 8.2 and pycuda).
#
# Loads a YOLO TensorRT engine once (default: the best model, YOLO26n p00 FP16, SARD) and
# serves detections over TCP. Protocol, per frame:
#   client -> server: 4-byte big-endian length, then a JPEG image
#   server -> client: 4-byte big-endian length, then JSON:
#     {"boxes": [[x1, y1, x2, y2, score], ...],   (pixels of the original frame)
#      "ms": {"decode": .., "prep": .., "infer": .., "post": .., "total": ..}}
# Pre- and post-processing are the same functions as in all benchmarks (common.py).
import os, sys, json, time, socket, struct

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # repo root
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # creates the CUDA context
import common as C

ENGINE = (
    sys.argv[1] if len(sys.argv) > 1 else "engines/yolo26n-sard_p00_fp16_640.engine"
)
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 5000


class TrtModel:
    """Loads a TensorRT engine once and runs it frame after frame (no trtexec per image)."""

    def __init__(self, path):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as fh:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(fh.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs, self.outputs, self.bindings = [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(
                self.engine.get_binding_dtype(i)
            )  # float32: trtexec keeps I/O in FP32
            host = cuda.pagelocked_empty(
                trt.volume(shape), dtype
            )  # page-locked: faster copies
            dev = cuda.mem_alloc(host.nbytes)
            self.bindings.append(int(dev))
            slot = (host, dev, shape)
            (self.inputs if self.engine.binding_is_input(i) else self.outputs).append(
                slot
            )

    def __call__(self, x):
        host, dev, _ = self.inputs[0]
        np.copyto(host, x.ravel())
        cuda.memcpy_htod_async(dev, host, self.stream)  # image to the GPU
        self.context.execute_async_v2(self.bindings, self.stream.handle)
        for h, d, _ in self.outputs:
            cuda.memcpy_dtoh_async(h, d, self.stream)  # results back
        self.stream.synchronize()
        return [h.reshape(s) for h, _, s in self.outputs]


def recv_exact(conn, n):
    """Read exactly n bytes (a TCP read can return less than asked for)."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None  # client disconnected
        buf += chunk
    return buf


model = TrtModel(ENGINE)
print("loaded %s" % ENGINE, flush=True)

# warm-up, so the first real frame is not slower than the rest
dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
for _ in range(10):
    model(dummy)

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", PORT))
srv.listen(1)
print("listening on port %d (Ctrl+C to stop)" % PORT, flush=True)

while True:
    conn, addr = srv.accept()
    conn.setsockopt(
        socket.IPPROTO_TCP, socket.TCP_NODELAY, 1
    )  # send replies immediately
    print("client connected from %s" % addr[0], flush=True)
    frames = 0
    while True:
        head = recv_exact(conn, 4)
        if head is None:
            break
        data = recv_exact(conn, struct.unpack(">I", head)[0])
        if data is None:
            break
        t0 = time.time()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        t1 = time.time()
        x, r, pad = C.prep_yolo(img)  # letterbox to 640x640
        t2 = time.time()
        out = model(x)[0]
        t3 = time.time()
        det = C.post_yolo_raw(out, r, pad)  # boxes in original pixels
        t4 = time.time()
        reply = json.dumps(
            {
                "boxes": [
                    [round(float(v), 1) for v in b[:4]] + [round(float(b[4]), 3)]
                    for b in det
                ],
                "ms": {
                    "decode": 1000 * (t1 - t0),
                    "prep": 1000 * (t2 - t1),
                    "infer": 1000 * (t3 - t2),
                    "post": 1000 * (t4 - t3),
                    "total": 1000 * (t4 - t0),
                },
            }
        ).encode()
        conn.sendall(struct.pack(">I", len(reply)) + reply)
        frames += 1
    conn.close()
    print("client disconnected after %d frames" % frames, flush=True)
