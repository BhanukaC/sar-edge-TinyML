import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()  # reads sar-edge/.env
key = os.getenv("ROBOFLOW_API_KEY")
if not key:
    raise SystemExit(
        "ROBOFLOW_API_KEY missing: create sar-edge/.env (see .env.example)"
    )

rf = Roboflow(api_key=key)
ds = (
    rf.workspace("datasets-pdabr")
    .project("sard-8xjhy")
    .version(16)
    .download("yolov8", location="data/SARD")
)
print("downloaded to", ds.location)
