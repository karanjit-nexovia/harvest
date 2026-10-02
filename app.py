"""Harvest: turn unlabeled video into robot training data.   streamlit run app.py"""
import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Blindspot", layout="wide")
from harvest import config  # noqa: E402

_PALETTE = ["#4C9AFF", "#36B37E", "#FFAB00", "#6554C0", "#00B8D9", "#FF5630", "#FF8B00", "#57D9A3",
            "#998DD9", "#E774BB", "#79E2F2", "#8993A4"]
STEP_COLORS = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(config.STEP_NAMES)}
STEP_COLORS["idle"] = "#DFE1E6"


def runs():
    return sorted([p for p in config.OUT.glob("*") if (p / "clips.jsonl").exists()
                   and not p.name.startswith("audit_")],
                  key=lambda p: -p.stat().st_mtime)


def load(run_dir):
    recs = [json.loads(l) for l in open(run_dir / "clips.jsonl")]
    stats = json.load(open(run_dir / "stats.json")) if (run_dir / "stats.json").exists() else {}
    ev = json.load(open(run_dir / "eval.json")) if (run_dir / "eval.json").exists() else {}
    return recs, stats, ev


def timeline(rec):
    d = max(0.1, rec["end"] - rec["start"])
    bars = "".join(
        f'<div title="{s["name"]} {s["start_s"]:.1f}-{s["end_s"]:.1f}s" style="position:absolute;'
        f'left:{100 * max(0, s["start_s"]) / d:.1f}%;width:{100 * max(0.02, s["end_s"] - s["start_s"]) / d:.1f}%;'
        f'top:0;bottom:0;background:{STEP_COLORS.get(s["name"], "#999")};border-right:1px solid #fff"></div>'
        for s in rec["steps"])
    legend = " ".join(f'<span style="color:{STEP_COLORS.get(s["name"])}">■</span>{s["name"]}' for s in rec["steps"])
    return (f'<div style="position:relative;height:14px;background:#eee;border-radius:3px;overflow:hidden">{bars}</div>'
            f'<div style="font-size:12px">{legend}</div>')


page = st.sidebar.radio("Page", ["Blindspot", "Harvest", "Label", "Evaluate", "Train"])
st.sidebar.caption(f"Search: {config.SEARCH_BACKEND} · Cosmos: {'MOCK' if config.MOCK else config.COSMOS_MODEL}")

if page == "Blindspot":
    import pandas as pd
    from harvest import audit
    st.title("Blindspot")
    st.caption("Your perception model is already running on every camera. Where does it fail? Blindspot uses "
               "NVIDIA Cosmos3-Reason as a judge over the VAST archive, grades the YOLO11 detections stored at "
               "ingest, and hands you the clips to retrain on.")
    cams = ["sdg_warehouse_cam-2", "i24_cam-1", "pie_cam-3", "smartspace_cam-1", "neighborhood_cam-1",
            "sf_streets_cam-1"]
    c1, c2 = st.columns([4, 1])
    pick = c1.multiselect("Camera packs to audit", cams, cams[:4])
    per = c2.number_input("clips per camera", 3, 50, 12)
    if st.button("Run audit", type="primary"):
        box = st.empty()
        lines = []

        def progress(msg):
            lines.append(msg)
            box.code("\n".join(lines[-14:]))

        audit.run(pick, int(per), progress=progress)
        st.rerun()
    audits = sorted([p for p in config.OUT.glob("audit_*") if (p / "report.json").exists()],
                    key=lambda p: -p.stat().st_mtime)
    if not audits:
        st.info("No audit yet.")
        st.stop()
    out = st.selectbox("Audit", audits, format_func=lambda p: p.name)
    rep = json.load(open(out / "report.json"))
    recs = [json.loads(l) for l in open(out / "clips.jsonl")]
    run = rep.get("run", {})
    m = st.columns(4)
    m[0].metric("Archive", f"{run.get('archive_segments') or 0:,} segments")
    m[1].metric("Clips audited", run.get("clips", len(recs)))
    m[2].metric("Cameras", len(run.get("cameras", [])))
    m[3].metric("Cosmos time", f"{run.get('cosmos_s', 0):.0f} s")
    st.error(f"Blind spots: {rep['headline']}")
    df = pd.DataFrame([{"object": o, "seen by Cosmos (clips)": v["seen"], "YOLO11 detected": v["detected"],
                        "detection rate": v["rate"], "share found per frame": v.get("count_recall"),
                        "class in model": "yes" if v["has_class"] else "NO"}
                       for o, v in rep["objects"].items()])
    st.subheader("Per object: found at all, and how many of them per frame")
    st.bar_chart(df.set_index("object")[["detection rate", "share found per frame"]].fillna(0))
    st.dataframe(df, use_container_width=True, hide_index=True)
    if rep.get("phantoms"):
        st.subheader("Phantom detections — labels YOLO11 reported that nothing in the clip explains")
        st.dataframe(pd.DataFrame(rep["phantoms_by_camera"]), use_container_width=True, hide_index=True)
    c1, c2 = st.columns(2)
    c1.subheader("By camera pack")
    c1.dataframe(pd.DataFrame(rep["by_camera"]), use_container_width=True, hide_index=True)
    c2.subheader("By condition")
    c2.dataframe(pd.DataFrame(rep["by_condition"]), use_container_width=True, hide_index=True)
    st.subheader("Failures (the retraining set)")
    fails = [r for r in recs if any(not c["yolo_found"] for c in r["checks"]) or r.get("phantoms")]
    cols = st.columns(3)
    for i, r in enumerate(fails[:12]):
        with cols[i % 3]:
            st.video(str(out / r["file"]))
            miss = ", ".join(c["object"] for c in r["checks"] if not c["yolo_found"]) or "—"
            ph = ", ".join(r.get("phantoms", [])) or "—"
            st.markdown(f"**YOLO missed: {miss}** · **phantoms: {ph}** · {r['camera_id']}")
            st.caption(f"Cosmos: {', '.join(o['name'] + ' x' + str(o['count']) for o in r['inventory'])} · "
                       f"YOLO said: {', '.join(list(r['yolo'])[:5]) or 'nothing'} · {json.dumps(r['conditions'])}")
            if r.get("notes"):
                st.caption(r["notes"][:160])
    b1, b2 = st.columns(2)
    if b1.button("Log audit to Weights & Biases"):
        audit.log_wandb(out)
        b1.success("Logged.")
    if b2.button("Export retraining set"):
        z = audit.export_retrain(out)
        b2.download_button("Download retraining set (zip)", open(z, "rb"), file_name=f"{out.name}_retrain.zip")

elif page == "Harvest":
    st.title("Harvest")
    st.caption("Describe what your AI system must learn — a robot, a self-driving stack, a safety or "
               "store-analytics model. Get back segmented, labelled training clips from your video archive.")
    c1, c2 = st.columns([4, 1])
    request = c1.text_input("What should your system learn?", "a warehouse robot that lifts and moves pallets")
    k = c2.number_input("hits per query", 3, 100, 20)
    if c1.button("1 · Plan with W&B Inference"):
        from harvest import planner
        with st.spinner("Planning searches..."):
            st.session_state["plan"] = planner.plan(request)
    plan = st.session_state.get("plan")
    if plan:
        st.caption(f"Planner: {plan['planner']} — {plan['why']}")
        queries = st.text_area("VAST search queries (one per line)", "\n".join(plan["queries"])).splitlines()
        from harvest import planner as _pl
        cams = ["(any)"] + list(_pl.CAMERAS)
        cam = st.selectbox("Camera", cams, index=cams.index(plan["camera_id"]) if plan.get("camera_id") in cams else 0)
        doms = list(config.DOMAINS)
        domain = st.selectbox("Domain (label set)", doms, index=doms.index(plan.get("domain", config.DOMAIN)),
                              format_func=lambda d: f"{d} — {config.DOMAINS[d]['desc']}")
    else:
        queries, cam, domain = [request], "(any)", config.DOMAIN
    with st.expander("Filter settings"):
        os.environ["HAND_ACTIVITY"] = str(st.slider("Hand activity needed (local video only)", 0.1, 1.0, 0.35, 0.05))
    if st.button("2 · Harvest", type="primary"):
        from harvest import pipeline
        box = st.empty()
        lines = []

        def progress(msg):
            lines.append(msg)
            box.code("\n".join(lines[-12:]))

        for q in [q.strip() for q in queries if q.strip()]:
            progress(f"=== {q}")
            pipeline.run(q, k=int(k), progress=progress, camera=None if cam == "(any)" else cam, domain=domain)
        st.rerun()
    all_runs = runs()
    if not all_runs:
        st.info("No runs yet.")
        st.stop()
    run_dir = st.selectbox("Run", all_runs, format_func=lambda p: p.name)
    recs, stats, ev = load(run_dir)
    if stats:
        m = st.columns(5)
        if stats.get("archive_segments"):
            m[0].metric("Archive searched", f"{stats['archive_segments']:,} segments")
        else:
            m[0].metric("Video searched", f"{stats['searched_video_s'] / 60:.1f} min")
        m[1].metric("VAST search hits", stats["ranges"])
        m[2].metric("YOLO kept", stats["kept"])
        m[3].metric("Cosmos verified", stats.get("verified", stats["labelled"]))
        if ev.get("cost", {}).get("saving_x"):
            m[4].metric("GPU saved vs Cosmos-on-all", f"{ev['cost']['saving_x']}x")
    if recs:
        from harvest import blindspots
        bs = blindspots.find(run_dir)
        rows = [(k, v["clips"], v["of"], "no class in the deployed detector (COCO)") for k, v in bs["no_coco_class"].items()]
        rows += [(k, v["clips"], v["of"], "detector missed it") for k, v in bs["missed_by_yolo"].items()]
        if rows:
            st.subheader("Detector blind spots")
            st.caption("Objects Cosmos saw in verified clips that the deployed YOLO11 never reported. "
                       "Each one comes with the verified clips to train it.")
            st.table([{"object": k, "clips": f"{n} of {of}", "why": why} for k, n, of, why in rows[:8]])
    if ev.get("purity"):
        p = ev["purity"]
        st.info(f"Dataset purity ({p['clips_checked']} clips checked by a person): raw VAST search "
                f"{p['raw_search_precision']:.0%} → Harvest-verified {(p['harvest_precision'] or 0):.0%}")
    if ev.get("accuracy"):
        a = ev["accuracy"]
        st.success(f"vs hand labels ({a['clips_compared']} clips): label {a['label_acc']:.0%} · "
                   f"step recall {a['step_recall']:.0%} · precision {a['step_precision']:.0%} · "
                   f"boundary ±{a['boundary_err_s']}s")
    cols = st.columns(3)
    for i, r in enumerate(recs):
        with cols[i % 3]:
            st.video(str(run_dir / r["file"]))
            badge = "✅ verified" if r.get("verified", True) else "❌ rejected"
            st.markdown(f"{badge} · **{r['label']}** · {r.get('camera_id') or r['video']}"
                        + (f" · ⚠ {r['error'][:60]}" if r.get("error") else ""))
            if r.get("evidence"):
                st.caption(r["evidence"][:160])
            st.markdown(timeline(r), unsafe_allow_html=True)
    if st.button("Export dataset"):
        from harvest import export
        zip_path, n = export.export(run_dir)
        st.download_button(f"Download {n} clips (zip)", open(zip_path, "rb"), file_name=f"{run_dir.name}.zip")

elif page == "Label":
    st.title("Hand labels")
    st.caption("Label clips BEFORE looking at Cosmos's answer. These are the ground truth for the eval.")
    all_runs = runs()
    if not all_runs:
        st.stop()
    run_dir = st.selectbox("Run", all_runs, format_func=lambda p: p.name)
    recs, _, _ = load(run_dir)
    lab_path = run_dir / "labels.jsonl"
    done = {}
    if lab_path.exists():
        for l in open(lab_path):
            t = json.loads(l); done[t["clip_id"]] = t
    st.write(f"{len(done)} of {len(recs)} labelled")
    todo = [r for r in recs if r["clip_id"] not in done] or recs
    r = st.selectbox("Clip", todo, format_func=lambda r: r["clip_id"])
    st.video(str(run_dir / r["file"]))
    d = r["end"] - r["start"]
    relevant = st.radio(f'Does this clip really show "{r["query"]}"?', ["Yes", "No"], horizontal=True) == "Yes"
    label = st.selectbox("Label", config.LABELS)
    n = st.number_input("How many steps", 0, 8, 3)
    steps = []
    for j in range(int(n)):
        c = st.columns(3)
        name = c[0].selectbox(f"step {j + 1}", config.STEP_NAMES, key=f"n{j}")
        a = c[1].number_input("start s", 0.0, d, min(d, j * d / max(1, n)), 0.1, key=f"a{j}")
        b = c[2].number_input("end s", 0.0, d, min(d, (j + 1) * d / max(1, n)), 0.1, key=f"b{j}")
        steps.append({"name": name, "start_s": a, "end_s": b})
    if st.button("Save label", type="primary"):
        done[r["clip_id"]] = {"clip_id": r["clip_id"], "relevant": relevant, "label": label, "steps": steps}
        with open(lab_path, "w") as fh:
            fh.writelines(json.dumps(v) + "\n" for v in done.values())
        st.rerun()

elif page == "Train":
    st.title("Train on the harvest")
    st.caption("Close the loop: a small pose model learns the steps from Harvest's labels, scored on clips it never saw.")
    all_runs = runs()
    if not all_runs:
        st.stop()
    run_dir = st.selectbox("Run", all_runs, format_func=lambda p: p.name)
    log = st.checkbox("Log to Weights & Biases")
    if st.button("Train step model", type="primary"):
        from harvest import train
        with st.spinner("Reading poses and training..."):
            train.train(run_dir, use_wandb=log)
    if (run_dir / "train.json").exists():
        t = json.load(open(run_dir / "train.json"))
        c = st.columns(4)
        c[0].metric("Clips (train/test)", f"{t['clips_train']}/{t['clips_test']}")
        c[1].metric("Step accuracy, unseen clips", f"{t['frame_acc']:.0%}", f"{t['frame_acc'] - t['baseline_majority_acc']:+.0%} vs guessing")
        c[2].metric("Macro F1", f"{t['macro_f1']:.2f}")
        c[3].metric("Frames", t["frames_train"] + t["frames_test"])
        recs = {r["clip_id"]: r for r in load(run_dir)[0]}
        cid = st.selectbox("Unseen clip", t["test_clips"])
        r = recs[cid]
        st.video(str(run_dir / r["file"]))
        st.markdown("**Cosmos (teacher)**")
        st.markdown(timeline(r), unsafe_allow_html=True)
        from harvest import train
        seq = train.predict(run_dir, run_dir / r["file"])
        steps, cur = [], None
        for ts, name in seq:
            if cur and cur["name"] == name:
                cur["end_s"] = ts + 0.2
            else:
                cur = {"name": name, "start_s": ts, "end_s": ts + 0.2}
                steps.append(cur)
        st.markdown("**Trained pose model (student)**")
        st.markdown(timeline(dict(r, steps=steps)), unsafe_allow_html=True)

else:
    st.title("Accuracy and cost")
    all_runs = runs()
    if not all_runs:
        st.stop()
    run_dir = st.selectbox("Run", all_runs, format_func=lambda p: p.name)
    log = st.checkbox("Log to Weights & Biases")
    if st.button("Evaluate", type="primary"):
        from harvest import evaluate
        out = evaluate.evaluate(run_dir, use_wandb=log)
        st.json(out)
    elif (run_dir / "eval.json").exists():
        st.json(json.load(open(run_dir / "eval.json")))
