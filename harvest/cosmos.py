"""NVIDIA Cosmos3-Reason client: any OpenAI-compatible endpoint (on the event VM: the CoreWeave GPU host).

A clip is sent as a base64 video, or, if the endpoint refuses video, as up to 5 timestamped frames."""
import base64
import json
import re

import cv2

from . import config

_CLIENT = {}
_MODE = {"input": None}   # what this endpoint accepted: "video" or "frames"


def client():
    if "c" not in _CLIENT:
        from openai import OpenAI
        from . import vss
        base = config.COSMOS_BASE_URL or (vss.cosmos_url().rstrip("/") + "/v1")
        key = config.COSMOS_API_KEY or vss.gpu_token() or "none"
        _CLIENT["c"] = OpenAI(base_url=base, api_key=key)
        _CLIENT["model"] = config.COSMOS_MODEL or _CLIENT["c"].models.list().data[0].id
    return _CLIENT["c"]


def model_id():
    client()
    return _CLIENT["model"]


def _frames_content(path, n):
    cap = cv2.VideoCapture(str(path))
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 25
    parts = []
    for k in range(n):
        idx = int((k + 0.5) * total / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        f = cv2.resize(f, (int(f.shape[1] * 448 / f.shape[0]), 448))
        b64 = base64.b64encode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
        parts += [{"type": "text", "text": f"t={idx / fps:.1f}s"},
                  {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]
    cap.release()
    return parts


def _content(path, mode):
    if mode == "video":
        b64 = base64.b64encode(path.read_bytes()).decode()
        return [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]
    return _frames_content(path, min(5, config.COSMOS_FRAMES))


def _nearest(name, allowed, default):
    """Map Cosmos's wording onto a fixed vocabulary ('electric scooter' -> 'scooter')."""
    n = re.sub(r"[^a-z_]", "", str(name).lower().replace(" ", "_"))
    if n in allowed:
        return n
    for a in allowed:
        if a in n or n in a:
            return a
    for a in allowed:
        if a.split("_")[0] in n:
            return a
    return default


def _loads(raw):
    """json.loads that forgives the usual model slips (trailing or missing commas)."""
    try:
        return json.loads(raw)
    except ValueError:
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)
        fixed = re.sub(r"}\s*{", "},{", fixed)
        fixed = re.sub(r"\"\s*\n\s*\"", '","', fixed)
        return json.loads(fixed)
