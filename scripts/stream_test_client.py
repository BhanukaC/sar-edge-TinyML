# sar-edge/scripts/stream_test_client.py   run on the Mac from sar-edge/:
#   python scripts/stream_test_client.py [host] [number of images]
# Streams SARD test images to jetson_detect_server.py, as a drone camera would, and measures
# the end-to-end loop: JPEG encoding, network, Jetson processing, reply. Saves a few
# annotated frames to results/stream_test/ and prints a timing summary.
import os, sys, glob, json, time, socket, struct
import numpy as np, cv2

HOST = sys.argv[1] if len(sys.argv) > 1 else "nano-desktop.local"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 50
PORT = 5000
OUT = "results/stream_test"
os.makedirs(OUT, exist_ok=True)


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("server closed the connection")
        buf += chunk
    return buf


images = sorted(glob.glob("data/SARD/test/images/*.jpg"))[:N]
sock = socket.create_connection((HOST, PORT))
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print("connected to %s:%d, sending %d frames" % (HOST, PORT, len(images)))

loop_ms, jetson = [], {"decode": [], "prep": [], "infer": [], "post": [], "total": []}
for k, f in enumerate(images):
    frame = cv2.imread(f)  # 1920x1080, like the drone camera
    t0 = time.time()
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = jpg.tobytes()
    sock.sendall(struct.pack(">I", len(data)) + data)
    reply = json.loads(recv_exact(sock, struct.unpack(">I", recv_exact(sock, 4))[0]))
    loop_ms.append(1000 * (time.time() - t0))
    for key in jetson:
        jetson[key].append(reply["ms"][key])
    if k < 5:  # save a few for a visual check
        for x1, y1, x2, y2, s in reply["boxes"]:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(OUT, os.path.basename(f)), frame)
sock.close()

skip = 2  # first frames include connection set-up
print("frames: %d (first %d skipped in the averages)" % (len(loop_ms), skip))
print(
    "end-to-end per frame: mean %.1f ms, p95 %.1f ms, %.1f FPS"
    % (
        np.mean(loop_ms[skip:]),
        np.percentile(loop_ms[skip:], 95),
        1000 / np.mean(loop_ms[skip:]),
    )
)
print(
    "on the Jetson (mean): "
    + ", ".join("%s %.1f ms" % (k, np.mean(v[skip:])) for k, v in jetson.items())
)
print(
    "network + JPEG encoding (mean): %.1f ms"
    % (np.mean(loop_ms[skip:]) - np.mean(jetson["total"][skip:]))
)
print("annotated frames in", OUT)
