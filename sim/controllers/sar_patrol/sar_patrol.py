# sar-edge/sim/controllers/sar_patrol/sar_patrol.py   (Webots controller for the Mavic 2 Pro)
#
# Simulated search-and-rescue flight, the same every run:
#   1. start on the ground at HOME, 2. climb vertically to ALTITUDE,
#   3. fly a lawnmower pattern over all Pedestrian nodes, 4. return to HOME and land.
# The drone is moved directly by the supervisor (no flight control needed). While at search
# altitude, every FRAME_DT seconds of simulated time the camera frame goes to
# jetson_detect_server.py on the Jetson Nano, and the detections come back. Because the
# supervisor knows every person's true position, it checks per frame which people are in
# view and whether a detection covers them, and reports recall over the flight.
#
# Setup: Mavic2Pro node -> controller "sar_patrol", supervisor TRUE. Press Reset, then Play.
# Output: results/sim/ (annotated video, per-frame log, summary, latest frame); live view on
# a Display device that the controller adds to the drone on its first run.
import os, json, math, time, socket, struct, csv
import numpy as np, cv2
from controller import Supervisor, Display

HOST = os.environ.get("JETSON_HOST", "192.168.2.3")   # the Nano, via the Ethernet cable
PORT = 5000
HOME = (-10.0, -40.0)  # launch and landing point (x, y) in metres; pick an open spot in the scene
ALTITUDE = 25.0        # search height above the ground, metres
CAM_PITCH = 1.2        # camera tilt below the horizon in radians (pi/2 = straight down)
SPEED = 4.0            # horizontal speed, m/s
CLIMB = 2.0            # vertical speed for takeoff and landing, m/s
FRAME_DT = 0.2         # simulated seconds between frames (a 5 FPS camera)
OVERLAP = 0.7          # distance between search lines, as a share of the camera's footprint
MARGIN = 10.0          # metres added around the people's area
TOL_PX = 10            # a detection counts if its box, widened by this many pixels, covers the person

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(REPO, "results", "sim")
os.makedirs(OUT, exist_ok=True)

robot = Supervisor()
dt_ms = int(robot.getBasicTimeStep())
me = robot.getSelf()
trans_f, rot_f = me.getField("translation"), me.getField("rotation")

# --- camera and gimbal ---
camera = robot.getDevice("camera")
camera.enable(dt_ms)
W, H, FOV = camera.getWidth(), camera.getHeight(), camera.getFov()
F_PX = (W / 2.0) / math.tan(FOV / 2.0)                  # focal length in pixels
robot.getDevice("camera pitch").setPosition(CAM_PITCH)   # tilt the camera down
for name in ("camera roll", "camera yaw"):
    try:
        robot.getDevice(name).setPosition(0.0)
    except Exception:
        pass

# --- all people in the world (their true positions) ---
root = robot.getRoot().getField("children")
P = np.array([root.getMFNode(i).getPosition() for i in range(root.getCount())
              if root.getMFNode(i).getTypeName() == "Pedestrian"])
assert len(P), "no Pedestrian nodes found in the world"
# ground height from the people (a standing Pedestrian's origin is about 1.27 m above its feet),
# never from the drone's saved position
ground_z = float(np.median(P[:, 2])) - 1.27

# --- put the drone on the ground at HOME before anything else ---
trans_f.setSFVec3f([HOME[0], HOME[1], ground_z + 0.1])
rot_f.setSFRotation([0.0, 0.0, 1.0, 0.0])
me.resetPhysics()

# --- live view: a Display device on the drone, added automatically on the first run ---
display = robot.getDevice("display")
if display is None:
    slot = me.getField("bodySlot") or me.getField("extensionSlot")
    already = slot is not None and any(
        slot.getMFNode(i).getTypeName() == "Display" for i in range(slot.getCount()))
    if slot is not None and not already:
        # devices only become visible to a controller when the world is loaded, so add the
        # Display, save the world (drone on the ground at HOME) and reload it
        slot.importMFNodeFromString(-1, 'Display { name "display" width %d height %d }' % (W, H))
        print("added a Display to the drone; saving and reloading the world")
        robot.step(dt_ms)
        robot.worldSave()
        robot.worldReload()
        raise SystemExit                               # the controller restarts after the reload
    print("no Display available; live view disabled")

print("found %d people, ground height %.2f m, camera %dx%d, field of view %.1f deg"
      % (len(P), ground_z, W, H, math.degrees(FOV)))

# --- flight plan: takeoff at HOME, lawnmower over the people, back to HOME, landing ---
x0, x1 = P[:, 0].min() - MARGIN, P[:, 0].max() + MARGIN
y0, y1 = P[:, 1].min() - MARGIN, P[:, 1].max() + MARGIN
slant = ALTITUDE / math.sin(CAM_PITCH)                   # distance to the ground along the view
footprint = 2 * slant * math.tan(FOV / 2.0)             # rough ground width seen by the camera
spacing = max(OVERLAP * footprint, 1.0)
rows = np.arange(y0, y1 + spacing, spacing)
z_air, z_ground = ground_z + ALTITUDE, ground_z + 0.1
plan = [np.array([HOME[0], HOME[1], z_ground]), np.array([HOME[0], HOME[1], z_air])]
for k, y in enumerate(rows):
    xs = (x0, x1) if k % 2 == 0 else (x1, x0)           # alternate direction each line
    plan += [np.array([xs[0], y, z_air]), np.array([xs[1], y, z_air])]
plan += [np.array([HOME[0], HOME[1], z_air]), np.array([HOME[0], HOME[1], z_ground])]
print("home (%.0f, %.0f), search area %.0f x %.0f m, %d lines %.1f m apart, altitude %.0f m"
      % (HOME[0], HOME[1], x1 - x0, y1 - y0, len(rows), spacing, ALTITUDE))
print("a 0.5 m wide person is about %.0f px wide in the frame" % (0.5 * F_PX / slant))


def project(p, cam_pos, yaw):
    """Pixel position of a world point in the camera image, or None if out of view.
    The camera heads along the drone's yaw, tilted CAM_PITCH down (Webots: +x forward, +y left, +z up)."""
    cp, sp = math.cos(CAM_PITCH), math.sin(CAM_PITCH)
    fwd = np.array([math.cos(yaw) * cp, math.sin(yaw) * cp, -sp])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.cross(fwd, left)
    q = np.array([fwd, left, up]).dot(p - cam_pos)
    if q[0] <= 0.1:
        return None
    u = W / 2.0 - F_PX * q[1] / q[0]
    v = H / 2.0 - F_PX * q[2] / q[0]
    return (u, v) if 0 <= u < W and 0 <= v < H else None


def recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Jetson server closed the connection")
        buf += chunk
    return buf


sock = socket.create_connection((HOST, PORT), timeout=10)
sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
print("connected to the Jetson at %s:%d" % (HOST, PORT))

video = cv2.VideoWriter(os.path.join(OUT, "flight.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                        1.0 / FRAME_DT, (W, H))
log = csv.writer(open(os.path.join(OUT, "frames.csv"), "w", newline=""))
log.writerow(["sim_time", "x", "y", "detections", "people_in_view", "people_found",
              "loop_ms", "jetson_infer_ms", "jetson_total_ms"])

pos, wp, yaw, next_frame = plan[0].copy(), 1, 0.0, 0.0
seen, found, loop_ms, infer_ms = set(), set(), [], []

while robot.step(dt_ms) != -1 and wp < len(plan):
    # move towards the next point: horizontally at SPEED, vertically at CLIMB
    d = plan[wp] - pos
    vertical = abs(d[0]) < 1e-6 and abs(d[1]) < 1e-6
    stepd = (CLIMB if vertical else SPEED) * dt_ms / 1000.0
    dist = np.linalg.norm(d)
    if dist <= stepd:
        pos, wp = plan[wp].copy(), wp + 1
    else:
        pos = pos + d / dist * stepd
    if not vertical:
        yaw = math.atan2(d[1], d[0])                    # face the direction of travel
    trans_f.setSFVec3f([float(pos[0]), float(pos[1]), float(pos[2])])
    rot_f.setSFRotation([0.0, 0.0, 1.0, yaw])
    me.resetPhysics()                                    # no falling or drifting between steps

    # frames only at search altitude, and only during the search itself (not the way home)
    searching = abs(pos[2] - z_air) < 0.05 and 2 <= wp <= len(plan) - 3
    if not searching or robot.getTime() < next_frame:
        continue
    next_frame = robot.getTime() + FRAME_DT

    # camera frame (BGRA) -> JPEG -> Jetson -> detections
    img = np.frombuffer(camera.getImage(), np.uint8).reshape(H, W, 4)[:, :, :3].copy()
    t0 = time.time()
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = jpg.tobytes()
    sock.sendall(struct.pack(">I", len(data)) + data)
    reply = json.loads(recv_exact(sock, struct.unpack(">I", recv_exact(sock, 4))[0]))
    loop_ms.append(1000 * (time.time() - t0))
    infer_ms.append(reply["ms"]["infer"])
    boxes = reply["boxes"]

    # ground truth: which people are in view, and which are covered by a detection
    in_view, hit = [], []
    for i, p in enumerate(P):
        uv = project(p, pos, yaw)
        if uv is None:
            continue
        in_view.append(i)
        seen.add(i)
        u, v = uv
        if any(b[0] - TOL_PX <= u <= b[2] + TOL_PX and b[1] - TOL_PX <= v <= b[3] + TOL_PX for b in boxes):
            hit.append(i)
            found.add(i)
        cv2.circle(img, (int(u), int(v)), 6, (0, 255, 0) if i in hit else (0, 255, 255), 2)
    for x1b, y1b, x2b, y2b, s in boxes:
        cv2.rectangle(img, (int(x1b), int(y1b)), (int(x2b), int(y2b)), (0, 0, 255), 2)
    video.write(img)

    if display is not None:                              # live view in Webots
        live = img.copy()
        cv2.putText(live, "det %d  found %d/%d  %.0f ms" % (len(boxes), len(hit), len(in_view), loop_ms[-1]),
                    (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        ir = display.imageNew(cv2.cvtColor(live, cv2.COLOR_BGR2BGRA).tobytes(), Display.BGRA, W, H)
        display.imagePaste(ir, 0, 0, False)
        display.imageDelete(ir)

    if len(loop_ms) % 10 == 0:                          # progress every 10 frames
        print("t=%5.1fs  pos (%5.1f, %5.1f)  detections %d  in view %d, found %d  |  loop %.0f ms  |  "
              "so far %d of %d seen people found"
              % (robot.getTime(), pos[0], pos[1], len(boxes), len(in_view), len(hit),
                 loop_ms[-1], len(found), len(seen)), flush=True)
        cv2.imwrite(os.path.join(OUT, "latest.jpg"), img)

    log.writerow(["%.2f" % robot.getTime(), "%.1f" % pos[0], "%.1f" % pos[1], len(boxes),
                  len(in_view), len(hit), "%.1f" % loop_ms[-1], "%.1f" % infer_ms[-1],
                  "%.1f" % reply["ms"]["total"]])

sock.close()
video.release()
summary = {
    "people_total": len(P), "people_seen": len(seen), "people_found": len(found),
    "recall_over_flight": len(found) / float(len(seen)) if seen else 0.0,
    "frames": len(loop_ms),
    "end_to_end_ms_mean": float(np.mean(loop_ms)), "end_to_end_ms_p95": float(np.percentile(loop_ms, 95)),
    "jetson_infer_ms_mean": float(np.mean(infer_ms)),
    "home": list(HOME), "altitude_m": ALTITUDE, "camera": [W, H], "camera_pitch_rad": CAM_PITCH,
}
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print(json.dumps(summary, indent=2))
robot.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)   # drone is back on the ground at HOME