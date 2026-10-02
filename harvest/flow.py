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
    "E-scooter training": {"queries": ["person riding an electric scooter", "scooter rider in a bike lane",
                                       "scooter on the sidewalk"],
                           "cameras": ["sf_streets_cam-1", "pie_cam-3", "neighborhood_cam-1"]},
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


def find(p, n, exclude=(), depth=0, extra_queries=()):
    """VAST search, round-robin over cameras x queries, until n distinct clips not in `exclude`.
    Each deeper round asks VAST for more results per search, so it reaches past the clips already checked."""
    hits, seen = [], set(exclude)
    queries = list(p["queries"]) + [q for q in extra_queries if q and q not in p["queries"]]
    per = max(2, n // max(1, len(p["cameras"])) + 1)
    top_k = min(100, per * (depth + 2) + len(seen))
    for q in queries:
        for cam in p["cameras"]:
            try:
                res = vss.search(q, top_k=top_k, min_similarity=0.0, metadata_filters={"camera_id": cam})
            except Exception:  # noqa: BLE001 -- one bad search must not stop the flow
                continue
            took = 0
            for h in res:
                if took >= per:
                    break
                if h["source"] not in seen and h.get("camera_id", cam) in (cam, ""):
                    seen.add(h["source"])
                    hits.append(dict(h, query=q, camera_id=h.get("camera_id") or cam))
                    took += 1
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
            from . import cosmos
            text = ("You reviewed a YOLO11 object detector (COCO classes) against what you saw in the clips of a "
                    f"training dataset. Facts: {json.dumps(facts)}\nGive 3 to 5 concrete changes that would fix "
                    "the detector for this use case (classes to add, data to label, confusions to correct, "
                    'thresholds). Return ONLY JSON: {"changes": [{"change": "<imperative, short>", '
                    '"why": "<the evidence, with numbers>"}]}')
            r = cosmos.client().chat.completions.create(model=cosmos.model_id(), temperature=0.2, max_tokens=700,
                                                         messages=[{"role": "user", "content": text}])
            data = cosmos._loads(re.search(r"\{.*\}", r.choices[0].message.content or "", re.S).group(0))
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
    if spec.get("objects"):
        s += f". Also look for: {', '.join(o.replace('_', ' ') for o in spec['objects'])}"
    if spec.get("lighting") and spec["lighting"] != "any":
        s += f". Lighting: {spec['lighting']}"
    return s


def _check(h, cid, out, st, spec):
    """Download one clip, Cosmos checks it against the request, compare with YOLO11 -> record."""
    path = vss.download(h["source"], out / "clips" / f"{cid}.mp4")
    det = vss.detections(h["source"])
    yolo, yolo_avg = vss.class_counts(det), vss.per_frame(det)
    extra = spec.get("objects") or []
    inv, secs, err = audit.inventory(path, request=st["request"], extra_objects=extra)
    checks, phantoms = audit.compare(inv, yolo, yolo_avg, err, objects=audit.vocab(extra))
    seen_objs = {o["name"] for o in inv["objects"]}
    lacking = [m for m in spec.get("must", []) if m not in seen_objs]
    light_ok = spec.get("lighting", "any") in ("any", "", None) or inv["conditions"].get("lighting") == spec["lighting"]
    if lacking:
        inv["matches"], inv["match_reason"] = False, f"Cosmos did not see: {', '.join(lacking)}. " + inv.get("match_reason", "")
    elif not light_ok:
        inv["matches"], inv["match_reason"] = False, f"lighting is {inv['conditions'].get('lighting')}, not {spec['lighting']}"
    return {"clip_id": cid, "camera_id": h["camera_id"], "source": h["source"], "query": h["query"],
            "file": f"clips/{cid}.mp4", "yolo": yolo, "inventory": inv["objects"],
            "conditions": inv["conditions"], "summary": inv.get("summary", ""), "events": inv.get("events", []),
            "matches": bool(inv.get("matches")) and not err, "match_reason": inv.get("match_reason", ""),
            "duration_s": audit.clip_seconds(path), "checks": checks, "phantoms": phantoms, "error": err,
            "cosmos_s": round(secs, 1), "round": st["rounds"]}


def _save(out, st):
    json.dump(st, open(Path(out) / "state.json", "w"), indent=1)


def _rounds(out, st, target, n, max_rounds, step, on_clip):
    """Search -> Cosmos-check rounds until `target` clips are kept (Cosmos yes, reviewer did not remove)."""
    spec = st.get("spec") or {}
    seen = {f["source"] for f in st["found"]} | {s["source"] for s in st["skipped"]} | {c["source"] for c in st["clips"]}
    kept_n = sum(1 for r in st["clips"] if r["matches"])
    rounds = 0
    while kept_n < target and rounds < max_rounds:
        rounds += 1
        st["rounds"] = st.get("rounds", 0) + 1
        extra = [st["use_case"]] if st["rounds"] > 1 else []   # widen the search after the first round
        batch = find(st["plan"], n + 4, exclude=seen, depth=st["rounds"] - 1, extra_queries=extra)
        if not batch:
            st["exhausted"] = True
            break
        seen.update(h["source"] for h in batch)
        st["found"] += [{"source": h["source"], "camera_id": h["camera_id"], "query": h["query"],
                         "score": h.get("score"), "round": st["rounds"]} for h in batch[:n]]
        _save(out, st)
        step("found", {"round": st["rounds"], "clips": len(batch[:n]), "kept": kept_n})
        queue, spares, checked = list(batch[:n]), list(batch[n:]), 0
        while queue and kept_n < target:
            h = queue.pop(0)
            cid = f"clip_{len(st['clips']):02d}"
            try:
                rec = _check(h, cid, out, st, spec)
            except Exception as e:  # noqa: BLE001 -- one bad segment must not stop the dataset
                st["skipped"].append({"source": h["source"], "camera_id": h["camera_id"], "error": str(e)[:160]})
                if spares:
                    queue.append(spares.pop(0))
                _save(out, st)
                step("skipped", st["skipped"][-1])
                continue
            checked += 1
            rec["cosmos_matches"] = rec["matches"]
            st["clips"].append(rec)
            kept_n += rec["matches"]
            _save(out, st)
            if on_clip:
                on_clip(rec, out)
        step("round_done", {"round": st["rounds"], "checked": checked, "kept": kept_n, "target": target})
    st["reached_target"] = kept_n >= target


def _finalize(out, st, step, with_suggestions=True):
    kept = [r for r in st["clips"] if r["matches"]]
    rep = audit.report(kept, audit.vocab((st.get("spec") or {}).get("objects"))) if kept else \
        {"objects": {}, "mislabels": {}, "phantoms": []}
    st["report"] = {k: rep.get(k) for k in ("headline", "objects", "mislabels", "phantoms")}
    step("compared", st["report"])
    if with_suggestions:
        st["suggestions"] = suggest(st["use_case"], rep, kept)
        st["suggestions_stale"] = False
    st["seconds"] = round(time.time() - st["started"], 1)
    _save(out, st)
    step("suggested", st.get("suggestions"))


def run(use_case, n=15, on_step=None, on_clip=None, spec=None, target=6, max_rounds=4):
    """The whole flow; writes out/flow_<time>/state.json as it goes.
    Checks n clips per round and keeps searching deeper, round after round, until Cosmos has kept `target`
    clips, the archive has no unseen matches left, or max_rounds is reached.
    spec (custom requests): {"cameras": [...], "must": [objects], "lighting": "any|day|night|dim|backlit"}"""
    step = on_step or (lambda k, msg: None)
    spec = spec or {}
    out = config.OUT / f"flow_{time.strftime('%m%d_%H%M%S')}"
    (out / "clips").mkdir(parents=True, exist_ok=True)
    st = {"use_case": use_case, "spec": spec, "started": time.time(), "clips": [], "found": [], "skipped": [],
          "target": target, "per_round": n, "rounds": 0}
    st["plan"] = plan(use_case)
    if spec.get("cameras"):
        st["plan"]["cameras"] = list(spec["cameras"])
    st["request"] = request_text(use_case, spec)
    step("plan", st["plan"])
    vss.token(refresh=True)   # a fresh VAST login for every run; the app may have been up for hours
    st["archive_segments"] = vss.archive_segments()
    _rounds(out, st, target, n, max_rounds, step, on_clip)
    _finalize(out, st, step)
    return out, st


def review(out, clip_id, verdict):
    """The reviewer overrides Cosmos on one clip: verdict "remove" (not a match) or "undo"."""
    out = Path(out)
    st = json.load(open(out / "state.json"))
    for r in st["clips"]:
        if r["clip_id"] == clip_id:
            if verdict == "remove":
                r["review"], r["matches"] = "removed", False
            else:
                r.pop("review", None)
                r["matches"] = bool(r.get("cosmos_matches", r["matches"]))
    st["reached_target"] = sum(1 for r in st["clips"] if r["matches"]) >= st.get("target", 0)
    st["suggestions_stale"] = True
    _finalize(out, st, lambda k, v: None, with_suggestions=False)
    return st


def replace(out, on_step=None, on_clip=None, max_rounds=3):
    """Find replacements for removed clips: continue the search (never re-checking a clip) until the target is
    met again, then refresh the comparison and Cosmos's suggestions."""
    out = Path(out)
    step = on_step or (lambda k, msg: None)
    st = json.load(open(out / "state.json"))
    st.pop("exhausted", None)
    vss.token(refresh=True)
    _rounds(out, st, st.get("target", 6), st.get("per_round", 15), max_rounds, step, on_clip)
    _finalize(out, st, step)
    return st


def latest():
    runs = sorted([p for p in config.OUT.glob("flow_*") if (p / "state.json").exists()],
                  key=lambda p: -p.stat().st_mtime)
    return runs


def _frames(src, dst_dir, stem, times):
    """Save the frames at `times` (seconds) as JPEGs -> their paths inside the dataset."""
    import cv2
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


def export(out, wandb_log=True):
    """The fixed dataset: kept clips with Cosmos-corrected labels -> zip (+ W&B run with table + artifact)."""
    out = Path(out)
    st = json.load(open(out / "state.json"))
    if st.get("suggestions_stale"):   # the reviewer changed the set since Cosmos last wrote its suggestions
        _finalize(out, st, lambda k, v: None)
    kept = [r for r in st["clips"] if r["matches"]]
    cosmos_kept = [r for r in st["clips"] if r.get("cosmos_matches", r["matches"])]
    removed = [r for r in st["clips"] if r.get("review") == "removed"]
    slug = re.sub(r"[^a-z0-9]+", "_", st["use_case"].lower()).strip("_")[:40]
    ds = out / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    (ds / "clips").mkdir(parents=True)
    (ds / "frames").mkdir()
    rows = []
    for r in kept:
        shutil.copy(out / r["file"], ds / r["file"])
        dur = r.get("duration_s") or 6
        frames = _frames(out / r["file"], ds / "frames", r["clip_id"], [round(dur * f, 1) for f in (0.2, 0.5, 0.8)])
        rows.append({"clip": r["file"], "use_case": st["use_case"], "camera_id": r["camera_id"],
                     "source": r["source"], "summary": r["summary"], "why_it_fits": r["match_reason"],
                     # the fixed labels: what Cosmos saw, not what YOLO said
                     "objects": r["inventory"], "events": r["events"], "conditions": r["conditions"],
                     "yolo_said": sorted(r["yolo"]), "yolo_missed": [c["object"] for c in r["checks"] if not c["yolo_found"]],
                     "yolo_made_up": r["phantoms"], "frames": frames, "labeller": "NVIDIA Cosmos3-Reason",
                     "human_review": "approved (not removed by the reviewer)"})
    with open(ds / "annotations.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in rows)
    sug = st.get("suggestions", {}).get("changes", [])
    (ds / "README.md").write_text(
        f"# {st['use_case']}: Harvest dataset\n\n{len(rows)} clips. NVIDIA Cosmos3-Reason checked "
        f"{len(st['clips'])} clips from the VAST archive and kept {len(cosmos_kept)}; a reviewer then removed "
        f"{len(removed)} that were not a match.\n\nLabels in `annotations.jsonl` are Cosmos's (objects, "
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
                           "cosmos_kept": len(cosmos_kept), "reviewer_removed": len(removed),
                           "cosmos_precision": round((len(cosmos_kept) - len(removed)) / max(1, len(cosmos_kept)), 3),
                           "yolo_missed_objects": sum(len(x["yolo_missed"]) for x in rows),
                           "yolo_made_up_labels": sum(len(x["yolo_made_up"]) for x in rows)})
        t = wandb.Table(columns=["clip", "camera", "kept", "reviewer", "why", "cosmos_saw", "yolo_said", "yolo_missed",
                                 "yolo_made_up", "summary", "video"])
        for r in st["clips"]:
            t.add_data(r["clip_id"], r["camera_id"], r["matches"],
                       "removed: not a match" if r.get("review") == "removed" else "",  r["match_reason"],
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
    ap.add_argument("--clips", type=int, default=15, help="clips checked per round")
    ap.add_argument("--target", type=int, default=6, help="keep searching until this many are kept")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    out, st = run(a.use_case, a.clips, target=a.target, on_step=lambda k, v: print(k, json.dumps(v)[:200]),
                  on_clip=lambda r, o: print(" ", "KEEP" if r["matches"] else "drop", r["camera_id"], "|",
                                             r["summary"][:80], "| yolo missed",
                                             [c["object"] for c in r["checks"] if not c["yolo_found"]]))
    for c in st["suggestions"]["changes"]:
        print("suggest:", c["change"], "--", c["why"])
    print("dataset:", export(out, a.wandb))
