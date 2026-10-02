"""Step 1: natural-language query -> candidate time ranges in the raw videos.

Backends (config.SEARCH_BACKEND):
  vast   -- the event's semantic search over footage (POST VAST_SEARCH_URL)
  clip   -- local CLIP: a frame every 2 s embedded, matched to the query text
  motion -- no model: stretches where the picture changes (people moving). Query-agnostic, but
            never blocks the demo.
Each returns [{"video", "start", "end", "score"}], padded +-2 s and merged where they overlap."""
import json
import urllib.request

import cv2
import numpy as np

from . import config


def videos():
    return sorted(p for p in config.RAW.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"))


def duration(path):
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return n / fps


def _merge(ranges, pad=2.0):
    out = []
    for r in sorted(ranges, key=lambda r: (r["video"], r["start"])):
        r = dict(r, start=max(0.0, r["start"] - pad), end=r["end"] + pad)
        if out and out[-1]["video"] == r["video"] and r["start"] <= out[-1]["end"]:
            out[-1]["end"] = max(out[-1]["end"], r["end"])
            out[-1]["score"] = max(out[-1]["score"], r["score"])
        else:
            out.append(r)
    for r in out:
        r["end"] = min(r["end"], duration(config.RAW / r["video"]))
    return out


def _vast(query, k):
    req = urllib.request.Request(config.VAST_SEARCH_URL, data=json.dumps({"query": query, "k": k}).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {config.VAST_API_KEY}"} if config.VAST_API_KEY else {})})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _frames(path, every=2.0, size=224):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step, i, out = max(1, int(fps * every)), 0, []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                out.append((i / fps, cv2.resize(f, (size, size))))
        i += 1
    cap.release()
    return out


def _clip(query, k):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    with torch.no_grad():
        t = model.get_text_features(**proc(text=[query], return_tensors="pt").to(dev))
        t = t / t.norm(dim=-1, keepdim=True)
        hits = []
        for v in videos():
            fr = _frames(v)
            for j in range(0, len(fr), 32):
                batch = fr[j:j + 32]
                im = model.get_image_features(**proc(images=[f[:, :, ::-1] for _, f in batch], return_tensors="pt").to(dev))
                im = im / im.norm(dim=-1, keepdim=True)
                for (ts, _), s in zip(batch, (im @ t.T).squeeze(1).tolist()):
                    hits.append({"video": v.name, "start": ts, "end": ts + 2.0, "score": float(s)})
    hits.sort(key=lambda h: -h["score"])
    return hits[:k]


def _motion(query, k, every=0.5, thresh=2.5, min_len=1.5):
    hits = []
    for v in videos():
        fr = _frames(v, every=every, size=160)
        prev, run = None, None
        for ts, f in fr:
            g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
            act = float(np.abs(g - prev).mean()) if prev is not None else 0.0
            prev = g
            if act > thresh:
                run = run or {"video": v.name, "start": ts, "end": ts, "score": 0.0}
                run["end"], run["score"] = ts, max(run["score"], act / 50.0)
            elif run:
                if run["end"] - run["start"] >= min_len:
                    hits.append(run)
                run = None
        if run and run["end"] - run["start"] >= min_len:
            hits.append(run)
    hits.sort(key=lambda h: -h["score"])
    return hits[:k]


def _windows(ranges, max_len=10.0):
    """Long stretches split into <= max_len windows (Cosmos reasons best over short clips)."""
    out = []
    for r in ranges:
        t = r["start"]
        while t < r["end"] - 0.5:
            out.append(dict(r, start=t, end=min(r["end"], t + max_len)))
            t += max_len
    return out


def search(query, k=50, backend=None):
    backend = backend or config.SEARCH_BACKEND
    fn = {"vast": _vast, "clip": _clip, "motion": _motion}[backend]
    return _windows(_merge(fn(query, k)))[:k]
