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


def run(query, k=50, backend=None, progress=print):
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
