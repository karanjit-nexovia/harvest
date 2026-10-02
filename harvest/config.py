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

OUT = Path(os.getenv("HARVEST_OUT", ROOT / "out"))

# NVIDIA Cosmos3-Reason (any OpenAI-compatible endpoint). On the event VM these are found automatically
# from COSMOS3_REASON_URL / GPU_BEARER_TOKEN (see harvest/vss.py); set them only to point elsewhere.
COSMOS_BASE_URL = os.getenv("COSMOS_BASE_URL", "")
COSMOS_API_KEY = os.getenv("COSMOS_API_KEY", "")
COSMOS_MODEL = os.getenv("COSMOS_MODEL", "")
COSMOS_FRAMES = int(os.getenv("COSMOS_FRAMES", "5"))      # the event's Cosmos takes at most 5 images
MOCK = os.getenv("HARVEST_MOCK", "0") == "1"            # no Cosmos calls: a fixed sample answer

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "harvest")
