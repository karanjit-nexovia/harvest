"""Step 3: Cosmos segments each kept clip into action steps, as JSON.

Calls any OpenAI-compatible chat endpoint (NVIDIA NIM / build.nvidia.com / CoreWeave / W&B
inference). COSMOS_INPUT=video sends the clip as a base64 data URL; =frames (default, works with
any vision model) sends COSMOS_FRAMES evenly spaced JPEGs with their timestamps. Replies are checked
against the schema; one retry on bad JSON. HARVEST_MOCK=1 returns plausible fake steps (no API)."""
import base64
import json
import re
import time

import cv2
import jsonschema

from . import config

PROMPT = f"""You are labeling training data for warehouse robots. Watch the clip and segment the main
person's (or forklift's) physical action into steps. Use only these step names: {", ".join(config.STEP_NAMES)}.
Return ONLY JSON, no prose:
{{"label": one of {config.LABELS},
 "steps": [{{"name": ..., "start_s": ..., "end_s": ..., "conf": 0-1}}],
 "objects": ["..."], "notes": "<one sentence>"}}
Times are seconds from the start of the clip. If there is no clear hand-object action, use
label "other" and steps []."""

SCHEMA = {
    "type": "object", "required": ["label", "steps"],
    "properties": {
        "label": {"enum": config.LABELS},
        "steps": {"type": "array", "items": {
            "type": "object", "required": ["name", "start_s", "end_s"],
            "properties": {"name": {"enum": config.STEP_NAMES}, "start_s": {"type": "number"},
                           "end_s": {"type": "number"}, "conf": {"type": "number"}}}},
        "objects": {"type": "array"}, "notes": {"type": "string"}}}

_CLIENT = {}


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


_MODE = {"input": None}      # what this endpoint accepted last ("video" / "frames")


def _content(path, mode):
    if mode == "video":
        b64 = base64.b64encode(path.read_bytes()).decode()
        return [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]
    return _frames_content(path, min(5, config.COSMOS_FRAMES))


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    data = json.loads(m.group(0))
    jsonschema.validate(data, SCHEMA)
    return data


def _mock(path, duration):
    d = max(duration, 1.0)
    cuts = [0, 0.2, 0.4, 0.7, 1.0]
    names = ["approach", "reach", "grasp", "lift"]
    return {"label": "pick_up_object", "objects": ["item"], "notes": "mock",
            "steps": [{"name": n, "start_s": round(a * d, 1), "end_s": round(b * d, 1), "conf": 0.5}
                      for n, a, b in zip(names, cuts, cuts[1:])]}


def segment(path, duration):
    """-> (result dict, seconds spent, error or None)"""
    t0 = time.time()
    if config.MOCK:
        return _mock(path, duration), 0.0, None
    mode = config.COSMOS_INPUT if config.COSMOS_INPUT != "auto" else (_MODE["input"] or "video")
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}] + _content(path, mode)}]
    err = None
    if config.COSMOS_INPUT == "auto" and mode == "video" and _MODE["input"] is None:
        try:   # does this endpoint take video? (once)
            r = client().chat.completions.create(model=model_id(), messages=messages, temperature=0.2, max_tokens=800)
            _MODE["input"] = "video"
            try:
                return _parse(r.choices[0].message.content or ""), time.time() - t0, None
            except Exception as e:  # noqa: BLE001 -- it answered; fix the JSON below
                text = r.choices[0].message.content or ""
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "That was not valid JSON for the schema. Return ONLY the JSON."}]
        except Exception as e:  # noqa: BLE001 -- no video input here: frames from now on
            print("cosmos: video input refused, using frames --", f"{type(e).__name__}: {e}"[:200])
            _MODE["input"] = "frames"
            messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}] + _content(path, "frames")}]
    for attempt in range(2):
        try:
            r = client().chat.completions.create(model=model_id(), messages=messages,
                                                 temperature=0.2, max_tokens=800)
            text = r.choices[0].message.content or ""
            return _parse(text), time.time() - t0, None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
            if attempt == 0 and "text" in locals():
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "That was not valid JSON for the schema. Return ONLY the JSON."}]
    return {"label": "other", "steps": [], "objects": [], "notes": "failed"}, time.time() - t0, err
