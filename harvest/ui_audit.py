"""The Blindspot audit page: the finished-product view of an audit run."""
import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from harvest import audit, config

CAMERAS = {"sdg_warehouse_cam-2": "Warehouse (synthetic)", "smartspace_cam-1": "Warehouse (indoor)",
           "i24_cam-1": "I-24 highway", "pie_cam-3": "Toronto dashcam", "neighborhood_cam-1": "Neighborhood",
           "sf_streets_cam-1": "SF streets"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
.hv-hero { background: linear-gradient(120deg, #0B1F17 0%, #12372A 55%, #1F5C45 100%); border-radius: 18px;
           padding: 28px 32px; color: #F2FBF6; margin-bottom: 18px; }
.hv-hero h1 { font-size: 34px; font-weight: 800; margin: 0 0 4px 0; color: #F2FBF6; letter-spacing: -0.5px; }
.hv-hero p { font-size: 16px; margin: 0; color: #BFE3D1; max-width: 860px; line-height: 1.5; }
.hv-tag { display: inline-block; background: rgba(255,255,255,.12); color: #DDF5E9; border-radius: 999px;
          padding: 3px 11px; font-size: 12px; margin: 14px 6px 0 0; }
.hv-steps { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; }
.hv-step { border: 1px solid #E3E8E5; border-radius: 14px; padding: 14px 16px; background: #FFFFFF; }
.hv-step b { display: block; font-size: 14px; color: #0B1F17; margin-bottom: 3px; }
.hv-step span { font-size: 13px; color: #5B6B63; line-height: 1.4; }
.hv-step i { font-style: normal; font-size: 11px; font-weight: 700; color: #1F8A5B; letter-spacing: .6px; }
.hv-kpis { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin-bottom: 22px; }
.hv-kpi { background: #F6F8F7; border-radius: 14px; padding: 14px 16px; }
.hv-kpi .v { font-size: 28px; font-weight: 800; color: #0B1F17; line-height: 1.1; }
.hv-kpi .v.bad { color: #C2361F; }
.hv-kpi .l { font-size: 12px; color: #5B6B63; margin-top: 4px; }
.hv-h2 { font-size: 20px; font-weight: 700; color: #0B1F17; margin: 26px 0 4px 0; }
.hv-sub { font-size: 13.5px; color: #5B6B63; margin-bottom: 12px; }
.hv-cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.hv-card { border-radius: 14px; padding: 14px 16px; border: 1px solid #E3E8E5; background: #FFF; }
.hv-card.red { border-left: 5px solid #C2361F; } .hv-card.amber { border-left: 5px solid #D98E04; }
.hv-card.green { border-left: 5px solid #1F8A5B; }
.hv-card .name { font-size: 15px; font-weight: 700; color: #0B1F17; text-transform: capitalize; }
.hv-card .pct { font-size: 30px; font-weight: 800; margin: 2px 0; }
.hv-card.red .pct { color: #C2361F; } .hv-card.amber .pct { color: #D98E04; } .hv-card.green .pct { color: #1F8A5B; }
.hv-card .meta { font-size: 12px; color: #5B6B63; }
.hv-card .instead { font-size: 12px; color: #0B1F17; margin-top: 8px; }
.hv-badge { display: inline-block; font-size: 10.5px; font-weight: 700; padding: 2px 7px; border-radius: 6px;
            background: #FCE9E6; color: #A42B17; margin-left: 6px; vertical-align: middle; letter-spacing: .3px; }
.hv-chip { display: inline-block; font-size: 12px; padding: 2px 9px; border-radius: 999px; margin: 2px 4px 2px 0; }
.hv-chip.c { background: #E6F4EE; color: #145C3D; } .hv-chip.y { background: #EEF0F7; color: #2D3A66; }
.hv-chip.miss { background: #FCE9E6; color: #A42B17; font-weight: 600; }
.hv-chip.ph { background: #FFF3D6; color: #7A4F00; }
.hv-ph { border-radius: 14px; background: #FFF8E8; border: 1px solid #F4E1B0; padding: 14px 16px; }
.hv-ph .big { font-size: 22px; font-weight: 800; color: #7A4F00; }
.hv-ph .meta { font-size: 12.5px; color: #6B5A33; }
.hv-clip { font-size: 12.5px; color: #3D4A44; line-height: 1.55; margin-bottom: 18px; }
.hv-clip .row { margin-top: 3px; }
.hv-clip .lbl { font-weight: 700; color: #0B1F17; display: inline-block; min-width: 54px; }
.hv-note { font-size: 12px; color: #5B6B63; font-style: italic; }
@media (max-width: 900px) { .hv-steps, .hv-kpis, .hv-cards { grid-template-columns: repeat(2, 1fr); } }
</style>
"""


def _e(s):
    return html.escape(str(s))


def _audits():
    return sorted([p for p in config.OUT.glob("audit_*") if (p / "report.json").exists()],
                  key=lambda p: -p.stat().st_mtime)


def _cam(c):
    return CAMERAS.get(c, c)


def _tone(rate):
    return "green" if rate >= 0.8 else "amber" if rate >= 0.4 else "red"


def _new_audit():
    with st.expander("Run a new audit", expanded=not _audits()):
        c1, c2, c3 = st.columns([4, 1, 1])
        pick = c1.multiselect("Cameras", list(CAMERAS), list(CAMERAS)[:4], format_func=_cam)
        per = c2.number_input("Clips per camera", 3, 50, 12)
        c3.write("")
        c3.write("")
        if c3.button("Run audit", type="primary", use_container_width=True):
            box = st.empty()
            lines = []

            def progress(msg):
                lines.append(msg)
                box.code("\n".join(lines[-12:]))

            audit.run(pick, int(per), progress=progress)
            st.rerun()
        st.caption(f"About 5 s of Cosmos time per clip: {len(pick) * per} clips takes about "
                   f"{max(1, round(len(pick) * per * 5 / 60))} min.")


def page():
    st.markdown(CSS, unsafe_allow_html=True)
    audits = _audits()
    out = None
    if audits:
        out = audits[0] if len(audits) == 1 else st.sidebar.selectbox(
            "Audit run", audits, format_func=lambda p: p.name.replace("audit_", "Run "))
    rep = json.load(open(out / "report.json")) if out else {}
    run = rep.get("run", {})
    tags = ""
    if run:
        tags = "".join(f'<span class="hv-tag">{_e(t)}</span>' for t in
                       [f"{run.get('archive_segments') or 0:,} segments indexed in VAST",
                        f"{run.get('clips', 0)} clips judged by Cosmos3-Reason",
                        "model under audit: YOLO11 (COCO)"])
    st.markdown(f"""<div class="hv-hero"><h1>Harvest</h1>
<p>Your vision model already runs on every camera, and nobody checks it. Harvest audits it against the whole
video archive, shows exactly where it is blind, and hands you the clips to retrain it.</p>{tags}</div>""",
                unsafe_allow_html=True)
    st.markdown("""<div class="hv-steps">
<div class="hv-step"><i>1 · VAST</i><b>Sample the archive</b><span>Pull clips per camera from the indexed video, with the YOLO11 detections stored at ingest.</span></div>
<div class="hv-step"><i>2 · NVIDIA COSMOS</i><b>Ask the judge</b><span>Cosmos3-Reason watches each clip and lists what is really there.</span></div>
<div class="hv-step"><i>3 · COMPARE</i><b>Grade the model</b><span>Objects YOLO missed, labels it made up, broken down by camera.</span></div>
<div class="hv-step"><i>4 · W&amp;B + EXPORT</i><b>Fix it</b><span>Failing clips become the retraining set; every audit is logged to Weights &amp; Biases.</span></div>
</div>""", unsafe_allow_html=True)
    _new_audit()
    if not out:
        st.info("No audit yet. Run one above.")
        return

    recs = [json.loads(l) for l in open(out / "clips.jsonl")]
    ok = [r for r in recs if not r.get("error")]
    objs = rep["objects"]
    blind = [o for o, v in objs.items() if (v["rate"] or 0) == 0 and v["seen"] >= 2]
    ph_clips = sum(1 for r in ok if r.get("phantoms"))
    fail_clips = [r for r in ok if any(not c["yolo_found"] for c in r["checks"])]
    kpis = [(f"{len(ok)}", "clips audited", ""), (f"{len(run.get('cameras', []))}", "cameras", ""),
            (f"{len(blind)}", "objects never detected", "bad"),
            (f"{round(100 * ph_clips / max(1, len(ok)))}%", "clips with a made-up label", "bad"),
            (f"{len(fail_clips)}", "clips in the retraining set", "")]
    st.markdown('<div class="hv-kpis">' + "".join(
        f'<div class="hv-kpi"><div class="v {c}">{_e(v)}</div><div class="l">{_e(l)}</div></div>'
        for v, l, c in kpis) + "</div>", unsafe_allow_html=True)

    # report card
    st.markdown('<div class="hv-h2">Report card: every object Cosmos saw</div>'
                '<div class="hv-sub">Share of clips where YOLO11 reported the object, when Cosmos3-Reason saw it. '
                '"No class" means the model was never trained on it, so no threshold tuning can fix it.</div>',
                unsafe_allow_html=True)
    mis = rep.get("mislabels", {})
    cards = []
    for o, v in sorted(objs.items(), key=lambda kv: ((kv[1]["rate"] or 0), -kv[1]["seen"])):
        r = v["rate"] or 0
        badge = "" if v["has_class"] else '<span class="hv-badge">NO CLASS</span>'
        inst = ""
        if mis.get(o):
            inst = ('<div class="instead">YOLO said instead: ' +
                    ", ".join(f"<b>{_e(m['yolo_label'])}</b> ({m['clips']})" for m in mis[o][:3]) + "</div>")
        cards.append(f'<div class="hv-card {_tone(r)}"><div class="name">{_e(o.replace("_", " "))}{badge}</div>'
                     f'<div class="pct">{round(100 * r)}%</div>'
                     f'<div class="meta">detected in {v["detected"]} of {v["seen"]} clips</div>{inst}</div>')
    st.markdown('<div class="hv-cards">' + "".join(cards) + "</div>", unsafe_allow_html=True)

    # phantoms
    per_cam = {}
    for r in ok:
        per_cam[r["camera_id"]] = per_cam.get(r["camera_id"], 0) + 1
    top = {}
    for row in rep.get("phantoms_by_camera", []):
        top.setdefault(row["camera"], row)
    if top:
        st.markdown('<div class="hv-h2">Made-up labels</div><div class="hv-sub">The label YOLO11 reports most '
                    'often on each camera that nothing in the clip explains.</div>', unsafe_allow_html=True)
        cols = st.columns(len(top))
        for col, (cam, row) in zip(cols, sorted(top.items(), key=lambda kv: -kv[1]["clips"] / per_cam.get(kv[0], 1))):
            col.markdown(f'<div class="hv-ph"><div class="big">"{_e(row["yolo_label"])}"</div>'
                         f'<div class="meta">in {row["clips"]} of {per_cam.get(cam, "?")} clips · {_e(_cam(cam))}</div></div>',
                         unsafe_allow_html=True)

    # heatmap + conditions
    st.markdown('<div class="hv-h2">Where it fails</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["By camera", "By condition", "Raw report"])
    with t1:
        bc = pd.DataFrame(rep["by_camera"])
        if not bc.empty:
            import altair as alt
            bc["camera"] = bc["camera"].map(_cam)
            bc["pct"] = (bc["rate"].fillna(0) * 100).round().astype(int)
            base = alt.Chart(bc).encode(x=alt.X("camera:N", title=None, axis=alt.Axis(labelAngle=0)),
                                        y=alt.Y("object:N", title=None,
                                                sort=sorted(rep["objects"], key=lambda o: rep["objects"][o]["rate"] or 0)))
            heat = base.mark_rect(cornerRadius=4).encode(
                color=alt.Color("pct:Q", scale=alt.Scale(domain=[0, 50, 100], range=["#C2361F", "#F2C14E", "#1F8A5B"]),
                                legend=alt.Legend(title="% detected")),
                tooltip=["object", "camera", "detected", "seen", "pct"])
            text = base.mark_text(fontSize=13, fontWeight="bold", color="white").encode(
                text=alt.Text("pct:Q", format="d"))
            st.altair_chart((heat + text).properties(height=max(220, 34 * bc["object"].nunique())),
                            use_container_width=True)
    with t2:
        cond = pd.DataFrame(rep["by_condition"])
        if not cond.empty:
            cond = cond[["object", "condition", "value", "detected", "seen", "rate"]]
            cond["rate"] = (cond["rate"].fillna(0) * 100).round().astype(int).astype(str) + "%"
            st.dataframe(cond, use_container_width=True, hide_index=True)
    with t3:
        st.json(rep, expanded=False)

    # failures gallery
    st.markdown(f'<div class="hv-h2">The retraining set ({len(fail_clips)} clips)</div>'
                '<div class="hv-sub">Each clip YOLO11 got wrong, with what Cosmos saw. These are the clips to '
                'label and retrain on.</div>', unsafe_allow_html=True)
    cams = sorted({r["camera_id"] for r in fail_clips})
    f = st.radio("Camera", ["All"] + cams, horizontal=True, format_func=lambda c: c if c == "All" else _cam(c),
                 label_visibility="collapsed")
    shown = [r for r in fail_clips if f == "All" or r["camera_id"] == f]
    cols = st.columns(3)
    for i, r in enumerate(shown[:9]):
        with cols[i % 3]:
            vid = out / r.get("file", "")
            if r.get("file") and vid.exists():
                st.video(str(vid))
            else:
                st.markdown('<div style="height:150px;border-radius:12px;background:#EEF1EF;display:flex;'
                            'align-items:center;justify-content:center;color:#8A968F;font-size:12px">clip on the VM</div>',
                            unsafe_allow_html=True)
            missed = {c["object"] for c in r["checks"] if not c["yolo_found"]}
            cos = "".join(f'<span class="hv-chip {"miss" if o["name"] in missed else "c"}">{_e(o["name"])} ×{_e(o["count"])}</span>'
                          for o in r["inventory"])
            ph = set(r.get("phantoms", []))
            yol = "".join(f'<span class="hv-chip {"ph" if y in ph else "y"}">{_e(y)}</span>' for y in list(r["yolo"])[:8]) \
                or '<span class="hv-note">nothing</span>'
            note = f'<div class="row hv-note">{_e(r["notes"][:150])}</div>' if r.get("notes") else ""
            st.markdown(f'<div class="hv-clip"><b>{_e(_cam(r["camera_id"]))}</b>'
                        f'<div class="row"><span class="lbl">Cosmos</span>{cos}</div>'
                        f'<div class="row"><span class="lbl">YOLO</span>{yol}</div>{note}</div>',
                        unsafe_allow_html=True)
    st.caption("Red = Cosmos saw it, YOLO missed it. Yellow = YOLO reported it, Cosmos didn't see it.")

    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Export retraining set", type="primary", use_container_width=True):
        z = audit.export_retrain(out)
        b1.download_button("Download zip", open(z, "rb"), file_name=f"{out.name}_retrain.zip", use_container_width=True)
    if b2.button("Log to Weights & Biases", use_container_width=True):
        with st.spinner("Logging..."):
            audit.log_wandb(out)
        b2.success("Logged to W&B.")
    st.markdown('<div class="hv-sub" style="margin-top:18px">Cosmos3-Reason is the judge, not hand labels: '
                'spot-check its calls before acting on a rate.</div>', unsafe_allow_html=True)
