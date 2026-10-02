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

def prompt(domain, request=""):
    d = config.DOMAINS.get(domain, config.DOMAINS["warehouse"])
    ask = (f'First, VERIFY: does this clip actually show "{request}"? Search engines return clips that '
           f'only look similar; be strict. ') if request else ""
    return f"""You are building verified training data for {d['desc']}. Watch the clip.
{ask}Then segment the main actor's (person's or vehicle's) action into steps, using only these step
names: {", ".join(d['steps'])}.
Return ONLY JSON, no prose:
{{"matches_request": true or false, "match_conf": 0-1, "evidence": "<what you saw that decides it>",
 "label": one of {d['labels']},
 "steps": [{{"name": ..., "start_s": ..., "end_s": ..., "conf": 0-1}}],
 "objects": ["..."], "notes": "<one sentence>"}}
Times are seconds from the start of the clip. If there is no clear action, use label "other"."""


PROMPT = prompt(config.DOMAIN)

SCHEMA = {
    "type": "object", "required": ["label", "steps"],
    "properties": {
        "label": {"enum": config.LABELS},
        "steps": {"type": "array", "items": {
            "type": "object", "required": ["name", "start_s", "end_s"],
            "properties": {"name": {"enum": config.STEP_NAMES}, "start_s": {"type": "number"},
                           "end_s": {"type": "number"}, "conf": {"type": "number"}}}},
        "objects": {"type": "array"}, "notes": {"type": "string"},
        "matches_request": {"type": "boolean"}, "match_conf": {"type": "number"}, "evidence": {"type": "string"}}}

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


_MODE = {"input": None, "guided": True}   # what this endpoint accepted ("video"/"frames"; guided JSON)


def _create(messages):
    kw = dict(model=model_id(), messages=messages, temperature=0.2, max_tokens=800)
    if _MODE["guided"]:
        try:
            return client().chat.completions.create(**kw, extra_body={"guided_json": SCHEMA})
        except Exception as e:  # noqa: BLE001
            if "guided" not in str(e).lower() and "422" not in str(e) and "400" not in str(e):
                raise
            _MODE["guided"] = False
            print("cosmos: guided JSON not supported, parsing leniently")
    return client().chat.completions.create(**kw)


def _content(path, mode):
    if mode == "video":
        b64 = base64.b64encode(path.read_bytes()).decode()
        return [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}]
    return _frames_content(path, min(5, config.COSMOS_FRAMES))


def _nearest(name, allowed, default):
    n = re.sub(r"[^a-z_]", "", str(name).lower().replace(" ", "_"))
    if n in allowed:
        return n
    for a in allowed:                      # push_or_pull_cart -> push_or_pull, picking_up -> ...
        if a in n or n in a:
            return a
    for a in allowed:
        if a.split("_")[0] in n:
            return a
    return default


def _loads(raw):
    try:
        return json.loads(raw)
    except ValueError:
        fixed = re.sub(r",\s*([}\]])", r"\1", raw)                     # trailing commas
        fixed = re.sub(r"}\s*{", "},{", fixed)                          # missing comma between objects
        fixed = re.sub(r"\"\s*\n\s*\"", '","', fixed)                   # missing comma between strings
        return json.loads(fixed)


def _parse(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    data = _loads(m.group(0))
    # forgiving: map near-miss names onto the vocabulary instead of failing the clip
    data["label"] = _nearest(data.get("label", "other"), config.LABELS, "other")
    steps = []
    for st in data.get("steps") or []:
        try:
            a, b = float(st.get("start_s", 0)), float(st.get("end_s", 0))
        except (TypeError, ValueError):
            continue
        steps.append({"name": _nearest(st.get("name", "idle"), config.STEP_NAMES, "idle"),
                      "start_s": min(a, b), "end_s": max(a, b), "conf": float(st.get("conf", 0.5) or 0.5)})
    data["steps"] = steps
    data.setdefault("objects", [])
    data["objects"] = [str(o) for o in data["objects"]] if isinstance(data["objects"], list) else []
    data["notes"] = str(data.get("notes", ""))
    m = data.get("matches_request", True)
    data["matches_request"] = m if isinstance(m, bool) else str(m).strip().lower() in ("true", "yes", "1")
    try:
        data["match_conf"] = float(data.get("match_conf", 0.5))
    except (TypeError, ValueError):
        data["match_conf"] = 0.5
    data["evidence"] = str(data.get("evidence", ""))
    jsonschema.validate(data, SCHEMA)
    return data


def _mock(path, duration):
    d = max(duration, 1.0)
    cuts = [0, 0.2, 0.4, 0.7, 1.0]
    names = ["approach", "reach", "grasp", "lift"]
    return {"label": "pick_up_object", "objects": ["item"], "notes": "mock", "matches_request": True,
            "match_conf": 0.5, "evidence": "mock",
            "steps": [{"name": n, "start_s": round(a * d, 1), "end_s": round(b * d, 1), "conf": 0.5}
                      for n, a, b in zip(names, cuts, cuts[1:])]}


def segment(path, duration, domain=None, request=""):
    """-> (result dict, seconds spent, error or None)"""
    PROMPT = prompt(domain or config.DOMAIN, request)
    t0 = time.time()
    if config.MOCK:
        return _mock(path, duration), 0.0, None
    mode = config.COSMOS_INPUT if config.COSMOS_INPUT != "auto" else (_MODE["input"] or "video")
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}] + _content(path, mode)}]
    err = None
    if config.COSMOS_INPUT == "auto" and mode == "video" and _MODE["input"] is None:
        try:   # does this endpoint take video? (once)
            r = _create(messages)
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
            r = _create(messages)
            text = r.choices[0].message.content or ""
            return _parse(text), time.time() - t0, None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
            if attempt == 0 and "text" in locals():
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": "That was not valid JSON for the schema. Return ONLY the JSON."}]
    return {"label": "other", "steps": [], "objects": [], "notes": "failed"}, time.time() - t0, err
