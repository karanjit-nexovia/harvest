"""The event's stack (VAST Builders Challenge): the VSS backend at INGRESS_URL (login -> JWT,
POST /api/v1/search, /videos/stream, /videos/detections) and the GPU endpoints (Cosmos3-Reason at
COSMOS3_REASON_URL, Bearer GPU_BEARER_TOKEN). All values come from the VM's environment, or from the
single /config/<team>.config if the environment lacks them. Nothing is printed or written to disk."""
import glob
import json
import os
import urllib.error
import urllib.parse
import urllib.request

_STATE = {}


def env(name, default=""):
    if name not in os.environ:
        files = sorted(glob.glob("/config/*.config"))
        if len(files) == 1 and "loaded" not in _STATE:
            _STATE["loaded"] = True
            for line in open(files[0]):
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.environ.get(name, default)


def available():
    return bool(env("INGRESS_URL") and env("USERNAME") and env("PASSWORD"))


def _req(method, path, body=None, auth=True, raw=False, timeout=120):
    url = env("INGRESS_URL").rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {token()}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read()[:600].decode("utf-8", "replace")
        raise RuntimeError(f"{method} {path.split('?')[0]} -> HTTP {e.code}: {detail}") from None
    return payload if raw else json.loads(payload)


def token(refresh=False):
    if refresh or "token" not in _STATE:
        r = _req("POST", "/api/v1/auth/login", {"username": env("USERNAME"), "password": env("PASSWORD")}, auth=False)
        _STATE["token"] = r["access_token"]
    return _STATE["token"]


def search(query, top_k=50, min_similarity=0.3, metadata_filters=None):
    """-> list of hits: {source, original_video, start, end, score, reasoning, camera_id, location}"""
    body = {"query": query, "top_k": min(100, top_k), "llm_top_n": 1, "min_similarity": min_similarity,
            "include_public": True}
    if metadata_filters:
        body["metadata_filters"] = metadata_filters
    try:
        r = _req("POST", "/api/v1/search", body)
    except RuntimeError as e:
        if "HTTP 422" not in str(e):
            raise
        print("search: full request rejected, retrying the minimal one --", str(e)[:300])
        r = _req("POST", "/api/v1/search", {"query": query, "top_k": min(100, top_k)})
    hits = []
    for x in r.get("results", []):
        def g(*keys, default=None):
            for k in keys:
                if x.get(k) is not None:
                    return x[k]
            return default
        start = float(g("start_sec", "segment_start_sec", "start_time", "start", default=0) or 0)
        end = float(g("end_sec", "segment_end_sec", "end_time", "end", default=start + 5) or start + 5)
        hits.append({"source": g("source", "segment_source", "s3_uri"), "original_video": g("original_video", default=""),
                     "start": start, "end": end, "score": float(g("similarity_score", "score", default=0) or 0),
                     "reasoning": g("reasoning_content", "reasoning", "description", default=""),
                     "camera_id": g("camera_id", default=""), "location": g("location", default="")})
    return [h for h in hits if h["source"]]


def download(source, out_path):
    q = urllib.parse.urlencode({"source": source, "token": token()})
    data = _req("GET", f"/api/v1/videos/stream?{q}", auth=False, raw=True, timeout=300)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return out_path


def detections(source):
    """The YOLO11 sidecar the pipeline already wrote for this segment (None if there is none)."""
    try:
        return _req("GET", "/api/v1/videos/detections?" + urllib.parse.urlencode({"source": source}))
    except Exception:  # noqa: BLE001 -- 404: no sidecar
        return None


def class_counts(det):
    """{class name: max count in any frame} from a detections sidecar, whatever its exact shape."""
    counts = {}

    def walk(o):
        if isinstance(o, dict):
            name = o.get("class_name") or o.get("label") or o.get("class") or o.get("name")
            if isinstance(name, str) and any(k in o for k in ("bbox", "box", "xyxy", "confidence", "score", "conf")):
                counts.setdefault("_frame", {})
                counts[name] = counts.get(name, 0) + 1
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(det or {})
    counts.pop("_frame", None)
    if isinstance(det, dict) and isinstance(det.get("object_classes"), (list, dict)):
        for c in det["object_classes"]:
            counts.setdefault(c if isinstance(c, str) else str(c), 1)
    return counts


def cosmos_url():
    return env("COSMOS3_REASON_URL")


def gpu_token():
    return env("GPU_BEARER_TOKEN")
