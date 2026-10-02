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
# vss = the event's VAST search (auto-picked on the event VM, where INGRESS_URL is set)
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "vss" if os.getenv("INGRESS_URL") or Path("/config").is_dir() else "motion")
VAST_SEARCH_URL = os.getenv("VAST_SEARCH_URL", "")      # POST {"query","k"} -> [{"video","start","end","score"}]
VAST_API_KEY = os.getenv("VAST_API_KEY", "")

# detection (YOLO, cheap filter in front of Cosmos)
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
YOLO_DEVICE = os.getenv("YOLO_DEVICE", "")              # "" = auto, "cpu", "0"
YOLO_FPS = float(os.getenv("YOLO_FPS", "5"))

# Cosmos (any OpenAI-compatible endpoint: NVIDIA NIM / build.nvidia.com / CoreWeave / W&B)
# On the event VM these come from COSMOS3_REASON_URL / GPU_BEARER_TOKEN (see harvest/vss.py)
COSMOS_BASE_URL = os.getenv("COSMOS_BASE_URL", "")
COSMOS_API_KEY = os.getenv("COSMOS_API_KEY", "")
COSMOS_MODEL = os.getenv("COSMOS_MODEL", "")
COSMOS_INPUT = os.getenv("COSMOS_INPUT", "auto")        # "video", "frames", or "auto" (video, else frames)
COSMOS_FRAMES = int(os.getenv("COSMOS_FRAMES", "5"))      # the event's Cosmos takes at most 5 images
MOCK = os.getenv("HARVEST_MOCK", "0") == "1"            # no API calls: deterministic fake steps

# cost ($ per GPU-hour) for the cost table
GPU_DOLLARS_PER_HOUR = float(os.getenv("GPU_DOLLARS_PER_HOUR", "2.5"))

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "harvest")

# Training data for any vision system: each domain has its own action labels and step vocabulary
DOMAINS = {
    "warehouse": {"desc": "warehouse robots and operations (workers, forklifts, pallets, carts)",
                  "labels": ["pick_up_object", "carry_object", "place_object", "push_or_pull_cart",
                             "operate_forklift", "walk_through", "other"],
                  "steps": ["approach", "reach", "grasp", "lift", "carry", "place", "push_or_pull", "walk", "idle"]},
    "traffic": {"desc": "autonomous driving and traffic systems (vehicles, intersections, highways)",
                "labels": ["vehicle_turn", "lane_change", "vehicle_stop", "vehicle_pass", "pedestrian_crossing",
                           "near_miss", "other"],
                "steps": ["approach", "slow_down", "stop", "accelerate", "turn", "lane_change", "cross", "pass", "idle"]},
    "people": {"desc": "safety and facility systems (people indoors, near vehicles or equipment)",
               "labels": ["person_walks_through", "person_picks_up", "person_hands_over", "person_near_vehicle",
                          "group_gathers", "fall_or_slip", "other"],
               "steps": ["enter", "walk", "stop", "reach", "pick_up", "hand_over", "carry", "exit", "idle"]},
}
DOMAIN = os.getenv("HARVEST_DOMAIN", "warehouse")
STEP_NAMES = sorted({s for d in DOMAINS.values() for s in d["steps"]})
LABELS = sorted({l for d in DOMAINS.values() for l in d["labels"]})
