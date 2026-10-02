"""The Harvest flow: use case -> clips from VAST -> Cosmos checks each clip -> compare with YOLO11 ->
Cosmos suggests model fixes -> the fixed dataset, exported to Weights & Biases.

    python -m harvest.flow "Close call training" --clips 6 --wandb
"""
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

from . import audit, config, vss

PRESETS = {
    "Forklift training": {"queries": ["forklift driving in a warehouse aisle", "forklift lifting a pallet"],
                          "cameras": ["sdg_warehouse_cam-2", "smartspace_cam-1"]},
    "Close call training": {"queries": ["forklift passing close to a worker", "person walking near a moving vehicle"],
                            "cameras": ["sdg_warehouse_cam-2", "smartspace_cam-1", "pie_cam-3"]},
    "Break-in training": {"queries": ["person trying a car door", "person climbing a fence or gate",
                                      "person looking into a window at night"],
                          "cameras": ["neighborhood_cam-1", "sf_streets_cam-1", "smartspace_cam-1"]},
    "Traffic analysis": {"queries": ["heavy traffic on a highway", "cars changing lanes", "trucks on the road"],
                         "cameras": ["i24_cam-1", "pie_cam-3"]},
}
ALL_CAMERAS = ["sdg_warehouse_cam-2", "smartspace_cam-1", "i24_cam-1", "pie_cam-3", "neighborhood_cam-1",
               "sf_streets_cam-1"]


def plan(use_case):
    """-> {"queries", "cameras", "by"}: presets are fixed; anything else goes to the W&B Inference planner."""
    if use_case in PRESETS:
        return dict(PRESETS[use_case], by="preset")
    from . import planner
    p = planner.plan(use_case)
    return {"queries": p["queries"], "cameras": [p["camera_id"]] if p.get("camera_id") else ALL_CAMERAS[:4],
            "by": p.get("planner", "none")}


def find(p, n):
    """VAST search, round-robin over cameras x queries, until n distinct clips."""
    hits, seen = [], set()
    for q in p["queries"]:
        for cam in p["cameras"]:
            try:
                res = vss.search(q, top_k=max(3, n), min_similarity=0.0, metadata_filters={"camera_id": cam})
            except Exception:  # noqa: BLE001 -- one bad search must not stop the flow
                continue
            for h in res[:max(2, n // max(1, len(p["cameras"])) + 1)]:
                if h["source"] not in seen and h.get("camera_id", cam) in (cam, ""):
                    seen.add(h["source"])
                    hits.append(dict(h, query=q, camera_id=h.get("camera_id") or cam))
    # interleave cameras so a small n still covers them
    by_cam = {}
    for h in hits:
        by_cam.setdefault(h["camera_id"], []).append(h)
    out = []
    while len(out) < n and any(by_cam.values()):
        for c in list(by_cam):
            if by_cam[c] and len(out) < n:
                out.append(by_cam[c].pop(0))
    return out


def suggest(use_case, rep, kept):
    """Cosmos3-Reason (text only) proposes changes to the detector; rules if the call fails."""
    facts = {"use_case": use_case, "clips": len(kept),
             "objects": {o: {"yolo_detection_rate": v["rate"], "in_model": v["has_class"], "clips": v["seen"]}
                         for o, v in rep.get("objects", {}).items()},
             "yolo_said_instead": rep.get("mislabels", {}), "made_up_labels": rep.get("phantoms", [])[:6]}
    if not config.MOCK:
        try:
            from . import segment
            text = ("You reviewed a YOLO11 object detector (COCO classes) against what you saw in the clips of a "
                    f"training dataset. Facts: {json.dumps(facts)}\nGive 3 to 5 concrete changes that would fix "
                    "the detector for this use case (classes to add, data to label, confusions to correct, "
                    'thresholds). Return ONLY JSON: {"changes": [{"change": "<imperative, short>", '
                    '"why": "<the evidence, with numbers>"}]}')
            r = segment.client().chat.completions.create(model=segment.model_id(), temperature=0.2, max_tokens=700,
                                                         messages=[{"role": "user", "content": text}])
            data = segment._loads(re.search(r"\{.*\}", r.choices[0].message.content or "", re.S).group(0))
            ch = [{"change": str(c.get("change", "")), "why": str(c.get("why", ""))}
                  for c in data.get("changes", []) if isinstance(c, dict) and c.get("change")]
            if ch:
                return {"by": "NVIDIA Cosmos3-Reason", "changes": ch[:5]}
        except Exception:  # noqa: BLE001
            pass
    ch = []
    for o, v in rep.get("objects", {}).items():
        inst = ", ".join(f"'{m['yolo_label']}'" for m in rep.get("mislabels", {}).get(o, [])[:2])
        if not v["has_class"] and v["seen"]:
            ch.append({"change": f"Add a '{o}' class and train on these clips",
                       "why": f"YOLO11 has no {o} class; it was in {v['seen']} clips and never detected"
                              + (f", reported as {inst} instead" if inst else "")})
        elif v["rate"] is not None and v["rate"] < 0.8 and v["seen"] >= 2:
            ch.append({"change": f"Label more '{o}' examples from these clips",
                       "why": f"detected in only {round(100 * v['rate'])}% of {v['seen']} clips"})
    for ph in rep.get("phantoms", [])[:2]:
        ch.append({"change": f"Add hard negatives for '{ph['yolo_label']}'",
                   "why": f"YOLO11 reported '{ph['yolo_label']}' in {ph['clips']} of {ph['of']} clips where it is not present"})
    return {"by": "rules (Cosmos unavailable)", "changes": ch[:5]}


def request_text(use_case, spec):
    """What Cosmos is asked to check each clip against."""
    s = use_case
    if spec.get("must"):
        s += f". The clip must clearly show: {', '.join(m.replace('_', ' ') for m in spec['must'])}"
    if spec.get("lighting") and spec["lighting"] != "any":
        s += f". Lighting: {spec['lighting']}"
    return s


def run(use_case, n=6, on_step=None, on_clip=None, spec=None):
    """The whole flow; writes out/flow_<time>/state.json as it goes.
    spec (custom requests): {"cameras": [...], "must": [objects], "lighting": "any|day|night|dim|backlit"}"""
    step = on_step or (lambda k, msg: None)
    spec = spec or {}
    out = config.OUT / f"flow_{time.strftime('%m%d_%H%M%S')}"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    st = {"use_case": use_case, "spec": spec, "started": time.time(), "clips": []}

    def save():
        json.dump(st, open(out / "state.json", "w"), indent=1)

    st["plan"] = plan(use_case)
    if spec.get("cameras"):
        st["plan"]["cameras"] = list(spec["cameras"])
    st["request"] = request_text(use_case, spec)
    step("plan", st["plan"])
    hits = find(st["plan"], n)
    st["found"] = [{"source": h["source"], "camera_id": h["camera_id"], "query": h["query"],
                    "score": h.get("score")} for h in hits]
    st["archive_segments"] = vss.archive_segments()
    save()
    step("found", st["found"])
    for i, h in enumerate(hits):
        cid = f"clip_{i:02d}"
        det = vss.detections(h["source"])
        yolo, yolo_avg = vss.class_counts(det), vss.per_frame(det)
        path = vss.download(h["source"], out / "clips" / f"{cid}.mp4")
        inv, secs, err = audit.inventory(path, request=st["request"])
        checks, phantoms = audit.compare(inv, yolo, yolo_avg, err)
        seen_objs = {o["name"] for o in inv["objects"]}
        lacking = [m for m in spec.get("must", []) if m not in seen_objs]
        light_ok = spec.get("lighting", "any") in ("any", "", None) or inv["conditions"].get("lighting") == spec["lighting"]
        if lacking:
            inv["matches"], inv["match_reason"] = False, f"Cosmos did not see: {', '.join(lacking)}. " + inv.get("match_reason", "")
        elif not light_ok:
            inv["matches"], inv["match_reason"] = False, f"lighting is {inv['conditions'].get('lighting')}, not {spec['lighting']}"
        rec = {"clip_id": cid, "camera_id": h["camera_id"], "source": h["source"], "query": h["query"],
               "file": f"clips/{cid}.mp4", "yolo": yolo, "inventory": inv["objects"],
               "conditions": inv["conditions"], "summary": inv.get("summary", ""), "events": inv.get("events", []),
               "matches": bool(inv.get("matches")) and not err, "match_reason": inv.get("match_reason", ""),
               "duration_s": audit.clip_seconds(path), "checks": checks, "phantoms": phantoms, "error": err,
               "cosmos_s": round(secs, 1)}
        st["clips"].append(rec)
        save()
        if on_clip:
            on_clip(rec, out)
    kept = [r for r in st["clips"] if r["matches"]]
    rep = audit.report(kept) if kept else {"objects": {}, "mislabels": {}, "phantoms": []}
    st["report"] = {k: rep.get(k) for k in ("headline", "objects", "mislabels", "phantoms")}
    step("compared", st["report"])
    st["suggestions"] = suggest(use_case, rep, kept)
    st["seconds"] = round(time.time() - st["started"], 1)
    save()
    step("suggested", st["suggestions"])
    return out, st


def latest():
    runs = sorted([p for p in config.OUT.glob("flow_*") if (p / "state.json").exists()],
                  key=lambda p: -p.stat().st_mtime)
    return runs


def export(out, wandb_log=True):
    """The fixed dataset: kept clips with Cosmos-corrected labels -> zip (+ W&B run with table + artifact)."""
    out = Path(out)
    st = json.load(open(out / "state.json"))
    kept = [r for r in st["clips"] if r["matches"]]
    slug = re.sub(r"[^a-z0-9]+", "_", st["use_case"].lower()).strip("_")[:40]
    ds = out / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    (ds / "clips").mkdir(parents=True)
    (ds / "frames").mkdir()
    rows = []
    for r in kept:
        shutil.copy(out / r["file"], ds / r["file"])
        dur = r.get("duration_s") or 6
        frames = __import__("harvest.datasets", fromlist=["_frames"])._frames(
            out / r["file"], ds / "frames", r["clip_id"], [round(dur * f, 1) for f in (0.2, 0.5, 0.8)])
        rows.append({"clip": r["file"], "use_case": st["use_case"], "camera_id": r["camera_id"],
                     "source": r["source"], "summary": r["summary"], "why_it_fits": r["match_reason"],
                     # the fixed labels: what Cosmos saw, not what YOLO said
                     "objects": r["inventory"], "events": r["events"], "conditions": r["conditions"],
                     "yolo_said": sorted(r["yolo"]), "yolo_missed": [c["object"] for c in r["checks"] if not c["yolo_found"]],
                     "yolo_made_up": r["phantoms"], "frames": frames, "labeller": "NVIDIA Cosmos3-Reason"})
    with open(ds / "annotations.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rows)
    sug = st.get("suggestions", {}).get("changes", [])
    (ds / "README.md").write_text(
        f"# {st['use_case']}: Harvest dataset\n\n{len(rows)} clips kept by NVIDIA Cosmos3-Reason out of "
        f"{len(st['clips'])} found in the VAST archive.\n\nLabels in `annotations.jsonl` are Cosmos's (objects, "
        "events, conditions); `yolo_missed` / `yolo_made_up` show where the deployed YOLO11 was wrong.\n\n"
        "## Suggested model changes\n" + "".join(f"- **{c['change']}**: {c['why']}\n" for c in sug))
    z = shutil.make_archive(str(out / f"{slug}_dataset"), "zip", ds)
    url = None
    if wandb_log:
        import wandb
        wb = wandb.init(project=os.getenv("WANDB_PROJECT") or config.WANDB_PROJECT,
                        entity=os.getenv("WANDB_TEAM") or None, name=f"{slug}-{out.name[5:]}", job_type="dataset",
                        config={"use_case": st["use_case"], "queries": st["plan"]["queries"],
                                "cameras": st["plan"]["cameras"]})
        wb.summary.update({"clips_found": len(st["clips"]), "clips_kept": len(rows),
                           "yolo_missed_objects": sum(len(x["yolo_missed"]) for x in rows),
                           "yolo_made_up_labels": sum(len(x["yolo_made_up"]) for x in rows)})
        t = wandb.Table(columns=["clip", "camera", "kept", "why", "cosmos_saw", "yolo_said", "yolo_missed",
                                 "yolo_made_up", "summary", "video"])
        for r in st["clips"]:
            t.add_data(r["clip_id"], r["camera_id"], r["matches"], r["match_reason"],
                       ", ".join(o["name"] for o in r["inventory"]), ", ".join(sorted(r["yolo"])),
                       ", ".join(c["object"] for c in r["checks"] if not c["yolo_found"]), ", ".join(r["phantoms"]),
                       r["summary"], wandb.Video(str(out / r["file"]), format="mp4"))
        objs = (st.get("report") or {}).get("objects") or {}
        wb.log({"cosmos_vs_yolo": t,
                "suggested_changes": wandb.Table(columns=["change", "why"], data=[[c["change"], c["why"]] for c in sug]),
                "yolo_detection_rate": wandb.plot.bar(
                    wandb.Table(columns=["object", "rate"], data=[[o, v["rate"] or 0] for o, v in objs.items()]),
                    "object", "rate", title="YOLO11 detection rate on this dataset (Cosmos as judge)")})
        art = wandb.Artifact(slug or "dataset", type="dataset", description=f"{st['use_case']}: {len(rows)} clips",
                             metadata={"clips": len(rows), "use_case": st["use_case"]})
        art.add_dir(str(ds))
        logged = wb.log_artifact(art)
        where = f"{wb.entity or os.getenv('WANDB_TEAM') or 'your-team'}/{wb.project}"
        ref = f"{where}/{art.name}:latest"
        try:
            logged.wait()
            ref = f"{where}/{art.name}:{logged.version}"
        except Exception:  # noqa: BLE001 -- offline / slow upload: :latest still resolves
            pass
        url = wb.url
        wb.finish()
    st["export"] = {"zip": z, "wandb_url": url, "clips": len(rows), "at": time.time(),
                    "artifact": ref if wandb_log else None}
    json.dump(st, open(out / "state.json", "w"), indent=1)
    return z, url


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("use_case", nargs="?", default="Close call training")
    ap.add_argument("--clips", type=int, default=6)
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    out, st = run(a.use_case, a.clips, on_step=lambda k, v: print(k, json.dumps(v)[:200]),
                  on_clip=lambda r, o: print(" ", "KEEP" if r["matches"] else "drop", r["camera_id"], "|",
                                             r["summary"][:80], "| yolo missed",
                                             [c["object"] for c in r["checks"] if not c["yolo_found"]]))
    for c in st["suggestions"]["changes"]:
        print("suggest:", c["change"], "--", c["why"])
    print("dataset:", export(out, a.wandb))
