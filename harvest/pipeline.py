"""query -> search -> YOLO filter -> cut clips -> Cosmos steps -> out/<run>/clips.jsonl + stats.json

    python -m harvest.pipeline "person takes an item from a shelf" [--k 50] [--backend motion]
"""
import argparse
import json
import re
import time

from . import config, detect, search, segment


def slug(q):
    return re.sub(r"[^a-z0-9]+", "_", q.lower()).strip("_")[:40]


def clip_seconds(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 25
    cap.release()
    return n / fps if n else 0.0


def run_vss(query, k, progress):
    """The event stack: VAST search -> segments, YOLO detections already computed at ingest (free
    filter), download the segment, Cosmos3-Reason segments the steps."""
    from . import vss
    run_dir = config.OUT / slug(query)
    (run_dir / "clips").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    hits = vss.search(query, top_k=k)
    progress(f"VAST search: {len(hits)} segments")
    need = [c.strip() for c in __import__("os").getenv("HARVEST_NEED", "person").split(",") if c.strip()]
    stats = {"query": query, "backend": "vss", "ranges": len(hits), "kept": 0, "labelled": 0, "yolo_s": 0.0,
             "cosmos_s": 0.0, "failed": 0, "searched_video_s": 0.0,
             "candidate_s": round(sum(h["end"] - h["start"] for h in hits), 1)}
    records = []
    for n, h in enumerate(hits):
        counts = vss.class_counts(vss.detections(h["source"]))
        keep = (not counts) or any(counts.get(c, 0) for c in need)   # no sidecar: let Cosmos judge
        progress(f"[{n + 1}/{len(hits)}] {h['camera_id'] or h['source'][-40:]} score {h['score']:.2f} "
                 f"{'KEEP' if keep else 'drop'} {dict(list(counts.items())[:6])}")
        if not keep:
            continue
        stats["kept"] += 1
        clip_id = f"seg{n:03d}"
        path = vss.download(h["source"], run_dir / "clips" / f"{clip_id}.mp4")
        dur = clip_seconds(path) or (h["end"] - h["start"])
        result, secs, err = segment.segment(path, dur)
        stats["cosmos_s"] += secs
        stats["failed"] += err is not None
        stats["labelled"] += err is None and result["label"] != "other"
        records.append({"clip_id": clip_id, "video": h["original_video"] or h["source"], "source": h["source"],
                        "start": 0.0, "end": round(dur, 2), "query": query, "search_score": round(h["score"], 3),
                        "camera_id": h["camera_id"], "location": h["location"], "index_caption": h["reasoning"][:400],
                        "tracks": counts, "label": result["label"], "steps": result["steps"],
                        "objects": result.get("objects", []), "notes": result.get("notes", ""), "error": err,
                        "gpu_s": {"yolo": 0.0, "cosmos": round(secs, 2)}, "file": f"clips/{clip_id}.mp4"})
        with open(run_dir / "clips.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)
    stats["searched_video_s"] = round(sum(r["end"] for r in records), 1)
    stats["wall_s"] = round(time.time() - t0, 1)
    stats["cosmos_s"] = round(stats["cosmos_s"], 1)
    json.dump(stats, open(run_dir / "stats.json", "w"), indent=1)
    with open(run_dir / "clips.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    progress(f"done: {stats}")
    return run_dir, records, stats


def run(query, k=50, backend=None, progress=print):
    if (backend or config.SEARCH_BACKEND) == "vss":
        return run_vss(query, k, progress)
    run_dir = config.OUT / slug(query)
    (run_dir / "clips").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ranges = search.search(query, k=k, backend=backend)
    progress(f"search: {len(ranges)} candidate ranges")
    stats = {"query": query, "backend": backend or config.SEARCH_BACKEND, "ranges": len(ranges), "kept": 0,
             "labelled": 0, "yolo_s": 0.0, "cosmos_s": 0.0, "failed": 0,
             "searched_video_s": round(sum(search.duration(v) for v in search.videos()), 1),
             "candidate_s": round(sum(r["end"] - r["start"] for r in ranges), 1)}
    records = []
    for n, rng in enumerate(ranges):
        keep, st = detect.screen(rng)
        stats["yolo_s"] += st["yolo_s"]
        progress(f"[{n + 1}/{len(ranges)}] {rng['video']} {rng['start']:.1f}-{rng['end']:.1f}s "
                 f"{'KEEP' if keep else 'drop'} {st}")
        if not keep:
            continue
        stats["kept"] += 1
        clip_id = f"{rng['video'].rsplit('.', 1)[0]}_{int(rng['start'] * 10):06d}"
        path = detect.cut(rng, run_dir / "clips" / f"{clip_id}.mp4")
        result, secs, err = segment.segment(path, rng["end"] - rng["start"])
        stats["cosmos_s"] += secs
        stats["failed"] += err is not None
        stats["labelled"] += err is None and result["label"] != "other"
        rec = {"clip_id": clip_id, "video": rng["video"], "start": round(rng["start"], 2),
               "end": round(rng["end"], 2), "query": query, "search_score": round(rng["score"], 3),
               "tracks": st, "label": result["label"], "steps": result["steps"],
               "objects": result.get("objects", []), "notes": result.get("notes", ""),
               "error": err, "gpu_s": {"yolo": st["yolo_s"], "cosmos": round(secs, 2)},
               "file": f"clips/{clip_id}.mp4"}
        records.append(rec)
        with open(run_dir / "clips.jsonl", "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in records)
    stats["wall_s"] = round(time.time() - t0, 1)
    stats["yolo_s"], stats["cosmos_s"] = round(stats["yolo_s"], 1), round(stats["cosmos_s"], 1)
    json.dump(stats, open(run_dir / "stats.json", "w"), indent=1)
    with open(run_dir / "clips.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in records)
    progress(f"done: {stats}")
    return run_dir, records, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--backend", default=None)
    a = ap.parse_args()
    run(a.query, a.k, a.backend)
