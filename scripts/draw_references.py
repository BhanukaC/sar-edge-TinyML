# sar-edge/scripts/draw_references.py   run from sar-edge/: python scripts/draw_references.py
# Draws the boxes from references/reference_<model>.json onto the smoke images,
# saving one folder of pictures per model in results/refs/<model>/.
import glob, json, os
import cv2

for ref in sorted(glob.glob("references/reference_*.json")):
    model = os.path.basename(ref)[len("reference_") : -len(".json")]
    out_dir = os.path.join("results", "refs", model)
    os.makedirs(out_dir, exist_ok=True)
    with open(ref) as fh:
        boxes = json.load(fh)
    for name, dets in boxes.items():
        img = cv2.imread(os.path.join("data", "smoke", name))
        for x1, y1, x2, y2, score in dets:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(
                img,
                "%.2f" % score,
                (int(x1), max(int(y1) - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        cv2.imwrite(os.path.join(out_dir, name), img)
    print(model, "->", out_dir)
