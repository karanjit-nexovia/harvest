"""Harvest Blindspot audit: audit the perception model the stack already runs, against the video archive.

The VAST DataEngine ran YOLO11 (COCO, 80 classes) on every segment at ingest and stored the detections.
Nobody checks them. Harvest samples segments from each camera pack (VAST search), asks NVIDIA
Cosmos3-Reason for a strict inventory of what is really in each clip (objects + conditions), and
compares it with YOLO's stored detections:

  * detection rate per object  = clips where YOLO reported the object / clips where Cosmos saw it
  * the same, split by camera pack and by condition (lighting, crowding, occlusion, distance)
  * objects the deployed model has no class for at all (forklift, pallet, cart, box ...)
  * the failing clips, with Cosmos's evidence -> a retraining set

    python -m harvest.audit --cameras sdg_warehouse_cam-2,i24_cam-1,pie_cam-3,smartspace_cam-1 --per-camera 15
"""
import argparse
import collections
import json
import time
from pathlib import Path

from . import config, segment, vss

# The inventory vocabulary, and the COCO class that would cover it (None: no class in the model)
OBJECTS = {"person": "person", "car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
           "bicycle": "bicycle", "traffic_light": "traffic light", "forklift": None, "pallet": None,
           "box": None, "cart": None, "shelf_rack": None, "cone": None}
CONDITIONS = {"lighting": ["day", "night", "backlit", "dim"], "crowding": ["empty", "sparse", "moderate", "dense"],
              "occlusion": ["low", "medium", "high"], "distance": ["near", "mid", "far"]}
DEFAULT_QUERIES = ["people", "vehicles", "forklift", "workers in an aisle", "busy scene", "empty scene"]

PROMPT = f"""You are auditing an object detector. Watch the clip and list, strictly, what is visible.
Objects (use only these names): {", ".join(OBJECTS)}.
Return ONLY JSON:
{{"objects": [{{"name": ..., "count": <how many are typically visible at the same time>, "visibility": "clear|partial|tiny"}}],
 "conditions": {{"lighting": one of {CONDITIONS['lighting']}, "crowding": one of {CONDITIONS['crowding']},
                "occlusion": one of {CONDITIONS['occlusion']}, "distance": one of {CONDITIONS['distance']}}},
 "notes": "<one sentence on what is hard to see in this clip>"}}
Only list an object if you are sure it is visible."""

SCHEMA = {"type": "object", "required": ["objects", "conditions"],
          "properties": {"objects": {"type": "array", "items": {"type": "object", "required": ["name"],
                                     "properties": {"name": {"enum": list(OBJECTS)}, "count": {"type": "number"},
                                                    "visibility": {"type": "string"}}}},
                         "conditions": {"type": "object"}, "notes": {"type": "string"}}}


def inventory(path):
    """Cosmos3-Reason's inventory of one clip -> (dict, seconds, error)."""
    import re
    t0 = time.time()
    if config.MOCK:
        return {"objects": [{"name": "person", "count": 2, "visibility": "clear"},
                            {"name": "forklift", "count": 1, "visibility": "clear"}],
                "conditions": {"lighting": "dim", "crowding": "sparse", "occlusion": "low", "distance": "mid"},
                "notes": "mock"}, 0.0, None
    mode = segment._MODE["input"] or "video"
    msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}] + segment._content(path, mode)}]
    err = None
    for attempt in range(2):
        try:
            kw = dict(model=segment.model_id(), messages=msgs, temperature=0.1, max_tokens=600)
            try:
                r = segment.client().chat.completions.create(**kw, extra_body={"guided_json": SCHEMA})
            except Exception:  # noqa: BLE001 -- no guided decoding / no video: plain call on frames
                segment._MODE["input"] = "frames"
                msgs = [{"role": "user", "content": [{"type": "text", "text": PROMPT}] + segment._content(path, "frames")}]
                r = segment.client().chat.completions.create(**dict(kw, messages=msgs))
            text = r.choices[0].message.content or ""
            data = segment._loads(re.search(r"\{.*\}", text, re.S).group(0))
            objs = []
            for o in data.get("objects") or []:
                name = segment._nearest(o.get("name", ""), list(OBJECTS), None)
                if name:
                    try:
                        cnt = int(float(o.get("count", 1) or 1))
                    except (TypeError, ValueError):
                        cnt = 1
                    objs.append({"name": name, "count": max(1, cnt), "visibility": str(o.get("visibility", ""))})
            cond = data.get("conditions") or {}
            cond = {k: segment._nearest(cond.get(k, ""), v, v[0]) for k, v in CONDITIONS.items()}
            return {"objects": objs, "conditions": cond, "notes": str(data.get("notes", ""))}, time.time() - t0, None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"[:300]
    return {"objects": [], "conditions": {}, "notes": ""}, time.time() - t0, err


def yolo_classes(counts):
    return {c.lower() for c in counts}


def sample(camera, per_camera, queries, progress):
    seen, hits = set(), []
    for q in queries:
        try:
            for h in vss.search(q, top_k=per_camera, min_similarity=0.0, metadata_filters={"camera_id": camera}):
                if h["source"] not in seen and h.get("camera_id", camera) in (camera, ""):
                    seen.add(h["source"]); hits.append(h)
        except Exception as e:  # noqa: BLE001
            progress(f"  search '{q}' on {camera} failed: {str(e)[:120]}")
        if len(hits) >= per_camera:
            break
    return hits[:per_camera]


def run(cameras, per_camera=15, queries=None, progress=print):
    out = config.OUT / f"audit_{time.strftime('%m%d_%H%M')}"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    queries = queries or DEFAULT_QUERIES
    recs, t0, cosmos_s = [], time.time(), 0.0
    for cam in cameras:
        hits = sample(cam, per_camera, queries, progress)
        progress(f"{cam}: {len(hits)} segments")
        for n, h in enumerate(hits):
            det = vss.detections(h["source"])
            yolo = vss.class_counts(det)
            yolo_avg = vss.per_frame(det)
            cid = f"{cam}_{n:03d}"
            path = vss.download(h["source"], out / "clips" / f"{cid}.mp4")
            inv, secs, err = inventory(path)
            cosmos_s += secs
            have = yolo_classes(yolo)
            checks = []
            for o in inv["objects"]:
                coco = OBJECTS[o["name"]]
                avg = yolo_avg.get(coco, 0.0) if coco else 0.0
                checks.append({"object": o["name"], "cosmos_count": o["count"], "visibility": o["visibility"],
                               "coco_class": coco, "yolo_found": bool(coco and coco in have),
                               "yolo_avg_count": avg,
                               # share of the objects YOLO finds in a typical frame (capped at 1)
                               "count_recall": round(min(1.0, avg / o["count"]), 3) if coco and o["count"] else 0.0})
            # phantoms: classes YOLO reported that nothing Cosmos saw could explain (a forklift called "boat")
            explained = {OBJECTS[o["name"]] for o in inv["objects"] if OBJECTS[o["name"]]}
            phantoms = sorted(c for c in have if c not in explained) if not err and inv["objects"] else []
            recs.append({"clip_id": cid, "camera_id": cam, "source": h["source"], "file": f"clips/{cid}.mp4",
                         "yolo": yolo, "yolo_avg": yolo_avg, "inventory": inv["objects"], "conditions": inv["conditions"],
                         "notes": inv["notes"], "checks": checks, "phantoms": phantoms, "error": err,
                         "cosmos_s": round(secs, 2)})
            misses = [c["object"] for c in checks if not c["yolo_found"]]
            progress(f"  [{n + 1}/{len(hits)}] cosmos sees {[o['name'] for o in inv['objects']]} | "
                     f"yolo misses {misses} | yolo phantoms {phantoms}{' | ERR ' + err[:60] if err else ''}")
            with open(out / "clips.jsonl", "w") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in recs)
    rep = report(recs)
    rep["run"] = {"cameras": cameras, "clips": len(recs), "cosmos_s": round(cosmos_s, 1),
                  "wall_s": round(time.time() - t0, 1), "archive_segments": vss.archive_segments()}
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    progress(f"done: {out}  {rep['headline']}")
    return out, recs, rep


def _rate(found, total):
    return round(found / total, 3) if total else None


def report(recs):
    ok = [r for r in recs if not r.get("error")]
    cr_obj = collections.defaultdict(list)
    cr_cond = collections.defaultdict(list)
    by_obj = collections.defaultdict(lambda: [0, 0])
    by_obj_cam = collections.defaultdict(lambda: [0, 0])
    by_cond = collections.defaultdict(lambda: [0, 0])
    fails = collections.defaultdict(list)
    calls = collections.defaultdict(collections.Counter)
    for r in ok:
        for c in r["checks"]:
            if c["coco_class"] is None:
                key = (c["object"],)
            else:
                key = (c["object"],)
            by_obj[key[0]][1] += 1
            by_obj[key[0]][0] += c["yolo_found"]
            by_obj_cam[(c["object"], r["camera_id"])][1] += 1
            by_obj_cam[(c["object"], r["camera_id"])][0] += c["yolo_found"]
            if c["coco_class"]:
                cr_obj[c["object"]].append(c.get("count_recall", 0.0))
                for k, v in r["conditions"].items():
                    cr_cond[(c["object"], k, v)].append(c.get("count_recall", 0.0))
                    by_cond[(c["object"], k, v)][1] += 1
                    by_cond[(c["object"], k, v)][0] += c["yolo_found"]
            if not c["yolo_found"]:
                fails[c["object"]].append(r["clip_id"])
                calls[c["object"]].update(r.get("phantoms", []))
    def mean(xs):
        return round(sum(xs) / len(xs), 3) if xs else None
    objects = {o: {"detected": f, "seen": t, "rate": _rate(f, t), "has_class": OBJECTS[o] is not None,
                   "count_recall": mean(cr_obj.get(o, [])) if OBJECTS[o] else 0.0}
               for o, (f, t) in sorted(by_obj.items(), key=lambda kv: -kv[1][1])}
    phantom = collections.Counter(p for r in ok for p in r.get("phantoms", []))
    phantom_by_cam = collections.Counter((r["camera_id"], p) for r in ok for p in r.get("phantoms", []))
    worst = sorted(((o, v) for o, v in objects.items() if v["seen"] >= 2), key=lambda kv: (kv[1]["rate"] or 0))
    head = ", ".join(f"{o} {int(100 * (v['rate'] or 0))}% detected ({v['seen']} clips)" for o, v in worst[:3])
    # count_recall stays in the data but out of the headline: the sidecar's frame_count covers every
    # frame while YOLO ran on a sample, so objects-per-frame is not calibrated yet
    mislabels = {o: [{"yolo_label": p, "clips": n} for p, n in calls[o].most_common(4)] for o in fails if calls[o]}
    if worst and worst[0][0] in mislabels:
        o = worst[0][0]
        head += f"; YOLO calls the {o} " + ", ".join(f"'{m['yolo_label']}'" for m in mislabels[o][:3])
    clips_per_cam = collections.Counter(r["camera_id"] for r in ok)
    if phantom_by_cam:
        (cam, p), n = phantom_by_cam.most_common(1)[0]
        head += f"; '{p}' reported on {cam} in {n} of {clips_per_cam[cam]} clips"
    return {"headline": head, "objects": objects, "mislabels": mislabels,
            "phantoms": [{"yolo_label": p, "clips": n, "of": len(ok)} for p, n in phantom.most_common()],
            "phantoms_by_camera": [{"camera": c, "yolo_label": p, "clips": n} for (c, p), n in phantom_by_cam.most_common()],
            "by_camera": [{"object": o, "camera": c, "detected": f, "seen": t, "rate": _rate(f, t)}
                          for (o, c), (f, t) in sorted(by_obj_cam.items())],
            "by_condition": [{"object": o, "condition": k, "value": v, "detected": f, "seen": t, "rate": _rate(f, t),
                              "count_recall": mean(cr_cond.get((o, k, v), []))}
                             for (o, k, v), (f, t) in sorted(by_cond.items()) if t >= 2],
            "retrain": {o: ids for o, ids in fails.items()}, "errors": len(recs) - len(ok)}


def log_wandb(out_dir):
    import os
    import wandb
    out_dir = Path(out_dir)
    recs = [json.loads(l) for l in open(out_dir / "clips.jsonl")]
    rep = json.load(open(out_dir / "report.json"))
    wb = wandb.init(project=config.WANDB_PROJECT, entity=os.getenv("WANDB_TEAM") or None,
                    name=f"blindspot-{out_dir.name}", job_type="audit", config=rep.get("run", {}))
    wb.summary["headline"] = rep["headline"]
    wb.log({"mislabels": wandb.Table(columns=["missed_object", "yolo_said_instead", "clips"],
                                     data=[[o, m["yolo_label"], m["clips"]] for o, ms in rep.get("mislabels", {}).items() for m in ms]),
            "phantoms_by_camera": wandb.Table(dataframe=__import__("pandas").DataFrame(rep.get("phantoms_by_camera", []))
                                              if rep.get("phantoms_by_camera") else None)})
    wb.log({"objects": wandb.plot.bar(wandb.Table(
        data=[[o, v["rate"] or 0] for o, v in rep["objects"].items()], columns=["object", "yolo_detection_rate"]),
        "object", "yolo_detection_rate", title="YOLO11 detection rate vs Cosmos3-Reason inventory")})
    wb.log({"by_camera": wandb.Table(dataframe=__import__("pandas").DataFrame(rep["by_camera"])),
            "by_condition": wandb.Table(dataframe=__import__("pandas").DataFrame(rep["by_condition"]))})
    t = wandb.Table(columns=["clip", "camera", "cosmos_sees", "yolo_reported", "yolo_missed", "conditions", "notes", "video"])
    for r in recs[:60]:
        t.add_data(r["clip_id"], r["camera_id"], ", ".join(f"{o['name']}x{o['count']}" for o in r["inventory"]),
                   ", ".join(r["yolo"]), ", ".join(c["object"] for c in r["checks"] if not c["yolo_found"]),
                   json.dumps(r["conditions"]), r["notes"], wandb.Video(str(out_dir / r["file"]), format="mp4"))
    wb.log({"clips": t})
    wb.finish()


def export_retrain(out_dir):
    import shutil
    out_dir = Path(out_dir)
    recs = {r["clip_id"]: r for r in (json.loads(l) for l in open(out_dir / "clips.jsonl"))}
    rep = json.load(open(out_dir / "report.json"))
    ds = out_dir / "retrain"
    shutil.rmtree(ds, ignore_errors=True)
    (ds / "clips").mkdir(parents=True)
    with open(ds / "annotations.jsonl", "w") as fh:
        for obj, ids in rep["retrain"].items():
            for cid in ids:
                r = recs[cid]
                shutil.copy(out_dir / r["file"], ds / r["file"])
                fh.write(json.dumps({"clip_id": cid, "file": r["file"], "missed_object": obj,
                                     "cosmos_inventory": r["inventory"], "conditions": r["conditions"],
                                     "camera_id": r["camera_id"], "source": r["source"]}) + "\n")
    (ds / "README.md").write_text("# Harvest retraining set (Blindspot audit)\n\nClips where the deployed YOLO11 missed an object "
                                  "Cosmos3-Reason saw. Weak labels: Cosmos inventory + conditions.\n\n"
                                  f"{json.dumps(rep['objects'], indent=1)}\n")
    return shutil.make_archive(str(out_dir / "retrain"), "zip", ds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="sdg_warehouse_cam-2,i24_cam-1,pie_cam-3,smartspace_cam-1")
    ap.add_argument("--per-camera", type=int, default=15)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--report", help="rebuild report.json for an existing audit dir (no new Cosmos calls)")
    a = ap.parse_args()
    if a.report:
        out = Path(a.report)
        old = json.load(open(out / "report.json"))
        rep = report([json.loads(l) for l in open(out / "clips.jsonl")])
        rep["run"] = old.get("run", {})
        json.dump(rep, open(out / "report.json", "w"), indent=1)
    else:
        out, _r, rep = run([c.strip() for c in a.cameras.split(",") if c.strip()], a.per_camera)
    print(json.dumps({k: rep[k] for k in ("headline", "mislabels", "run")}, indent=1))
    if a.wandb:
        log_wandb(out)
    print("retraining set:", export_retrain(out))
