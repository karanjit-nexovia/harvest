"""Use-case datasets: pick what you want to train on (close calls, unsafe acts...), Harvest finds the moments
Cosmos3-Reason described in the archive and exports them as a labelled dataset.

    python -m harvest.datasets "Close calls"            # build from every clip judged so far
    python -m harvest.datasets "Close calls" --more 6   # first search the archive for 6 more clips per camera
"""
import argparse
import json
import re
import shutil
import time
from pathlib import Path

from . import audit, config

USE_CASES = {
    "Close calls": {"types": ["close_call", "collision"], "query": "forklift close to a person",
                    "cameras": ["sdg_warehouse_cam-2", "smartspace_cam-1"],
                    "desc": "A person, forklift or vehicle comes dangerously close to another. Trains near-miss alerts."},
    "Unsafe behaviour": {"types": ["unsafe_act"], "query": "person walking in a vehicle lane",
                         "cameras": ["sdg_warehouse_cam-2", "smartspace_cam-1", "pie_cam-3"],
                         "desc": "People or drivers doing something unsafe. Trains safety-compliance models."},
    "Person–vehicle interaction": {"types": ["person_vehicle_interaction", "close_call"], "query": "pedestrian near a vehicle",
                                   "cameras": ["pie_cam-3", "smartspace_cam-1"],
                                   "desc": "People and vehicles sharing space. Trains robots and ADAS to yield."},
    "Loading & unloading": {"types": ["loading_unloading"], "query": "forklift lifting a pallet",
                            "cameras": ["sdg_warehouse_cam-2", "smartspace_cam-1"],
                            "desc": "Forklifts and workers moving goods. Trains warehouse-robot skills."},
    "Traffic congestion": {"types": ["congestion", "vehicle_interaction"], "query": "heavy traffic",
                           "cameras": ["i24_cam-1", "pie_cam-3"],
                           "desc": "Dense or merging traffic. Trains traffic analytics and driving models."},
}
SEV_RANK = {"low": 0, "medium": 1, "high": 2}


def pool():
    """Every clip Cosmos has described so far (full audits, live tests, dataset searches), newest first."""
    seen, recs = set(), []
    dirs = sorted([p for p in config.OUT.glob("*") if p.is_dir() and p.name.split("_")[0] in ("audit", "live", "ds")
                   and (p / "clips.jsonl").exists()], key=lambda p: -p.stat().st_mtime)
    for d in dirs:
        for l in open(d / "clips.jsonl"):
            r = json.loads(l)
            key = r.get("source") or r["clip_id"]
            if r.get("error") or key in seen or "events" not in r:
                continue
            seen.add(key)
            r["_dir"] = str(d)
            recs.append(r)
    return recs


def match(recs, types=(), text="", min_sev="low"):
    """[(record, [matching events])] for events of the given types and/or mentioning the text."""
    words = [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2]
    out = []
    for r in recs:
        hits = []
        for e in r.get("events", []):
            if SEV_RANK.get(e.get("severity"), 0) < SEV_RANK[min_sev]:
                continue
            blob = f"{e['type'].replace('_', ' ')} {e.get('description', '')} {r.get('summary', '')}".lower()
            if (types and e["type"] in types) or (words and all(w in blob for w in words)):
                hits.append(e)
        if hits:
            out.append((r, hits))
    return sorted(out, key=lambda m: -max(SEV_RANK.get(e["severity"], 0) for e in m[1]))


def collect(use_case, per_camera=4, cameras=None, on_clip=None, query=None):
    """Search the archive for more clips for this use case; Cosmos describes each one."""
    uc = USE_CASES.get(use_case, {})
    q = query or uc.get("query") or use_case
    return audit.run(cameras or uc.get("cameras") or ["sdg_warehouse_cam-2"], per_camera,
                     queries=[q] + audit.DEFAULT_QUERIES, progress=lambda m: None, on_clip=on_clip, prefix="ds")


def _frames(src, dst_dir, stem, times):
    try:
        import cv2
    except ImportError:
        return []
    cap, saved = cv2.VideoCapture(str(src)), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if ok:
            name = f"{stem}_{t:05.1f}s.jpg"
            cv2.imwrite(str(dst_dir / name), f)
            saved.append(f"frames/{name}")
    cap.release()
    return saved


def build(name, matches, wandb_log=False):
    """Write the dataset: clips/, frames/ (start, middle, end of each event), annotations.jsonl, README.md, .zip"""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    ds = config.OUT / "datasets" / f"{slug}_{time.strftime('%m%d_%H%M%S')}"
    (ds / "clips").mkdir(parents=True)
    (ds / "frames").mkdir()
    rows = []
    for r, events in matches:
        src = Path(r["_dir"]) / r["file"]
        if not src.exists():
            continue
        clip = f"clips/{r['clip_id']}.mp4"
        shutil.copy(src, ds / clip)
        for k, e in enumerate(events):
            mid = round((e["start_s"] + e["end_s"]) / 2, 1)
            frames = _frames(src, ds / "frames", f"{r['clip_id']}_e{k}", sorted({e["start_s"], mid, e["end_s"]}))
            rows.append({"clip": clip, "camera_id": r["camera_id"], "source": r.get("source"),
                         "event": e["type"], "severity": e["severity"], "start_s": e["start_s"], "end_s": e["end_s"],
                         "description": e.get("description", ""), "clip_summary": r.get("summary", ""),
                         "objects": r.get("inventory", []), "conditions": r.get("conditions", {}),
                         "frames": frames, "labeller": "NVIDIA Cosmos3-Reason"})
    with open(ds / "annotations.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rows)
    counts = {}
    for x in rows:
        counts[x["event"]] = counts.get(x["event"], 0) + 1
    (ds / "README.md").write_text(
        f"# {name}: Harvest dataset\n\n{USE_CASES.get(name, {}).get('desc', '')}\n\n"
        f"- {len({x['clip'] for x in rows})} clips, {len(rows)} labelled events: {json.dumps(counts)}\n"
        "- Source: VAST-indexed camera archive. Labels: NVIDIA Cosmos3-Reason (event type, severity, start/end "
        "seconds, description, objects, conditions).\n"
        "- `annotations.jsonl`: one row per event. `frames/`: the start, middle and end frame of each event.\n"
        "- Labels are machine-made: review a sample before training. Boxes are not included.\n")
    z = shutil.make_archive(str(ds), "zip", ds)
    if wandb_log:
        import os
        import wandb
        wb = wandb.init(project=config.WANDB_PROJECT, entity=os.getenv("WANDB_TEAM") or None,
                        name=f"dataset-{ds.name}", job_type="dataset")
        art = wandb.Artifact(slug, type="dataset", description=f"{name}: {len(rows)} events",
                             metadata={"events": counts, "clips": len({x['clip'] for x in rows})})
        art.add_dir(str(ds))
        wb.log_artifact(art)
        wb.finish()
    return ds, z, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("use_case", nargs="?", default="Close calls", help=f"one of {list(USE_CASES)} or free text")
    ap.add_argument("--more", type=int, default=0, help="search the archive for N more clips per camera first")
    ap.add_argument("--describe", help="add Cosmos summaries + events to an existing audit dir, then exit")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    if a.describe:
        audit.describe(a.describe)
        raise SystemExit
    if a.more:
        collect(a.use_case, a.more, on_clip=lambda r, o: print(" ", r["camera_id"], "|", r.get("summary", "")[:100]))
    uc = USE_CASES.get(a.use_case, {})
    m = match(pool(), uc.get("types", []), "" if uc else a.use_case)
    print(f"{len(m)} clips match '{a.use_case}'")
    if m:
        ds, z, rows = build(a.use_case, m, a.wandb)
        print(f"dataset: {z}  ({len(rows)} events)")
