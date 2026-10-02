"""Settings from the environment (.env is read if present). Everything has a working default."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
except OSError:
    pass

RAW = Path(os.getenv("HARVEST_RAW", ROOT / "data" / "raw"))
OUT = Path(os.getenv("HARVEST_OUT", ROOT / "out"))

# search: "vast" (the event's semantic search), "clip" (local CLIP), or "motion" (always works)
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "motion")
VAST_SEARCH_URL = os.getenv("VAST_SEARCH_URL", "")      # POST {"query","k"} -> [{"video","start","end","score"}]
VAST_API_KEY = os.getenv("VAST_API_KEY", "")

# detection (YOLO, cheap filter in front of Cosmos)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "")              # "" = auto, "cpu", "0"
YOLO_FPS = float(os.getenv("YOLO_FPS", "5"))

# Cosmos (any OpenAI-compatible endpoint: NVIDIA NIM / build.nvidia.com / CoreWeave / W&B)
COSMOS_BASE_URL = os.getenv("COSMOS_BASE_URL", "https://integrate.api.nvidia.com/v1")
COSMOS_API_KEY = os.getenv("COSMOS_API_KEY", "")
COSMOS_MODEL = os.getenv("COSMOS_MODEL", "nvidia/cosmos-reason1-7b")
COSMOS_INPUT = os.getenv("COSMOS_INPUT", "frames")      # "video" (data URL) or "frames" (N jpgs)
COSMOS_FRAMES = int(os.getenv("COSMOS_FRAMES", "8"))
MOCK = os.getenv("HARVEST_MOCK", "0") == "1"            # no API calls: deterministic fake steps

# cost ($ per GPU-hour) for the cost table
GPU_DOLLARS_PER_HOUR = float(os.getenv("GPU_DOLLARS_PER_HOUR", "2.5"))

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "harvest")

STEP_NAMES = ["reach", "grasp", "lift_or_pull", "move", "place", "conceal", "release", "idle"]
LABELS = ["take_item", "return_item", "conceal_item", "open_drawer", "pour", "other"]
