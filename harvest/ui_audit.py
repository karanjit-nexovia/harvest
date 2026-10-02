"""Harvest UI: the Overview page (one audit run) and the Live test page (judge a few clips on stage)."""
import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from harvest import audit, config

CAMERAS = {"sdg_warehouse_cam-2": "Warehouse A (synthetic)", "smartspace_cam-1": "Warehouse B (indoor)",
           "i24_cam-1": "I-24 highway", "pie_cam-3": "Toronto dashcam", "neighborhood_cam-1": "Neighborhood",
           "sf_streets_cam-1": "SF streets"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root { --ink:#0F172A; --muted:#64748B; --line:#E2E8F0; --soft:#F8FAFC; --accent:#2563EB;
        --red:#DC2626; --red-bg:#FEF2F2; --amber:#B45309; --amber-bg:#FFFBEB; --green:#15803D; --green-bg:#F0FDF4; }
html, body, [class*="css"], .stMarkdown, .stButton button, .stRadio, .stSelectbox { font-family:'Inter',sans-serif; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.2rem !important; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"] { display:none !important; }
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { display:flex !important; visibility:visible !important; }
[data-testid="stExpandSidebarButton"] svg, [data-testid="stSidebarCollapsedControl"] svg { color:#0F172A !important; fill:#0F172A; }
section[data-testid="stSidebar"] { background:#0F172A; }
section[data-testid="stSidebar"] * { color:#CBD5E1 !important; }
section[data-testid="stSidebar"] .hv-brand { color:#FFFFFF !important; }
section[data-testid="stSidebar"] .stButton button { background:#1E293B !important; border:1px solid #334155 !important; }
section[data-testid="stSidebar"] .stButton button:hover { border-color:#3B82F6 !important; }
section[data-testid="stSidebar"] .stButton button p { color:#E2E8F0 !important; }
.hv-brand { font-size:20px; font-weight:700; letter-spacing:-.3px; display:flex; align-items:center; gap:10px; }
.hv-mark { width:26px; height:26px; border-radius:7px; background:linear-gradient(135deg,#3B82F6,#1D4ED8);
           display:inline-block; }
.hv-head { display:flex; justify-content:space-between; align-items:flex-start; border-bottom:1px solid var(--line);
           padding-bottom:16px; margin-bottom:20px; gap:20px; flex-wrap:wrap; }
.hv-head h1 { font-size:26px; font-weight:700; color:var(--ink); margin:0; letter-spacing:-.4px; padding:0; }
.hv-head p { font-size:14px; color:var(--muted); margin:4px 0 0 0; max-width:720px; }
.hv-meta { font-size:12px; color:var(--muted); text-align:right; line-height:1.7; }
.hv-meta b { color:var(--ink); font-weight:600; }
.hv-steps { display:grid; grid-template-columns:repeat(4,1fr); border:1px solid var(--line); border-radius:10px;
            margin-bottom:22px; overflow:hidden; }
.hv-step { padding:13px 16px; border-right:1px solid var(--line); background:#FFF; }
.hv-step:last-child { border-right:none; }
.hv-step .n { font-size:11px; font-weight:600; color:var(--accent); letter-spacing:.5px; text-transform:uppercase; }
.hv-step b { display:block; font-size:14px; color:var(--ink); margin:2px 0; font-weight:600; }
.hv-step span { font-size:12.5px; color:var(--muted); line-height:1.45; }
.hv-kpis { display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-bottom:8px; }
.hv-kpi { border:1px solid var(--line); border-radius:10px; padding:14px 16px; background:#FFF; }
.hv-kpi .l { font-size:12px; color:var(--muted); font-weight:500; }
.hv-kpi .v { font-size:26px; font-weight:700; color:var(--ink); margin-top:2px; letter-spacing:-.5px; }
.hv-kpi .v.bad { color:var(--red); }
.hv-h2 { font-size:16px; font-weight:600; color:var(--ink); margin:30px 0 2px 0; }
.hv-sub { font-size:13px; color:var(--muted); margin-bottom:12px; }
.hv-table { width:100%; border-collapse:collapse; font-size:13.5px; border:1px solid var(--line); border-radius:10px;
            overflow:hidden; }
.hv-table th { text-align:left; font-weight:600; color:var(--muted); font-size:12px; background:var(--soft);
               padding:9px 14px; border-bottom:1px solid var(--line); }
.hv-table td { padding:10px 14px; border-bottom:1px solid var(--line); color:var(--ink); vertical-align:middle; }
.hv-table tr:last-child td { border-bottom:none; }
.hv-obj { font-weight:600; text-transform:capitalize; }
.hv-bar { height:8px; background:#F1F5F9; border-radius:4px; overflow:hidden; width:160px; display:inline-block;
          vertical-align:middle; margin-right:10px; }
.hv-bar div { height:100%; border-radius:4px; }
.hv-pill { display:inline-block; font-size:11.5px; font-weight:600; padding:2px 8px; border-radius:999px; }
.hv-pill.red { background:var(--red-bg); color:var(--red); } .hv-pill.amber { background:var(--amber-bg); color:var(--amber); }
.hv-pill.green { background:var(--green-bg); color:var(--green); } .hv-pill.grey { background:#F1F5F9; color:#475569; }
.hv-chip { display:inline-block; font-size:12px; padding:1px 8px; border-radius:6px; margin:2px 4px 2px 0;
           border:1px solid var(--line); background:#FFF; color:#334155; }
.hv-chip.miss { border-color:#FECACA; background:var(--red-bg); color:var(--red); font-weight:600; }
.hv-chip.ph { border-color:#FDE68A; background:var(--amber-bg); color:var(--amber); }
.hv-chip.ok { border-color:#BBF7D0; background:var(--green-bg); color:var(--green); }
.hv-phs { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }
.hv-ph { border:1px solid var(--line); border-radius:10px; padding:13px 16px; background:#FFF; }
.hv-ph .q { font-size:17px; font-weight:700; color:var(--amber); }
.hv-ph .m { font-size:12px; color:var(--muted); margin-top:2px; }
.hv-clip { border:1px solid var(--line); border-top:none; border-radius:0 0 10px 10px; padding:10px 12px 12px;
           font-size:12.5px; margin-bottom:16px; margin-top:-16px; background:#FFF; }
.hv-clip .t { font-weight:600; color:var(--ink); margin-bottom:4px; display:flex; justify-content:space-between; }
.hv-clip .r { margin-top:3px; } .hv-clip .k { display:inline-block; width:52px; color:var(--muted); font-weight:500; }
.hv-noclip { height:150px; border-radius:10px 10px 0 0; background:#F1F5F9; display:flex; align-items:center;
             justify-content:center; color:#94A3B8; font-size:12px; border:1px solid var(--line); margin-bottom:16px; }
.hv-foot { font-size:12px; color:var(--muted); margin-top:26px; padding-top:12px; border-top:1px solid var(--line); }
.hv-live { border:1px solid var(--line); border-radius:10px; padding:14px 16px; background:var(--soft); font-size:13px; color:var(--ink); }
@media (max-width: 900px) { .hv-steps, .hv-kpis, .hv-phs { grid-template-columns:repeat(2,1fr); } }
</style>
"""


def _e(s):
    return html.escape(str(s))


def _cam(c):
    return CAMERAS.get(c, c)


def _tone(rate):
    return "green" if rate >= 0.8 else "amber" if rate >= 0.4 else "red"


_COLOR = {"green": "#16A34A", "amber": "#D97706", "red": "#DC2626"}


def _runs(prefix):
    return sorted([p for p in config.OUT.glob(f"{prefix}_*") if (p / "clips.jsonl").exists()],
                  key=lambda p: -p.stat().st_mtime)


def sidebar_brand():
    st.markdown(CSS, unsafe_allow_html=True)
    st.sidebar.markdown('<div class="hv-brand"><span class="hv-mark"></span>Harvest</div>'
                        '<div style="font-size:12px;margin:4px 0 18px 0">Vision model audit</div>',
                        unsafe_allow_html=True)


def _header(title, sub, meta=""):
    st.markdown(f'<div class="hv-head"><div><h1>{_e(title)}</h1><p>{_e(sub)}</p></div>'
                f'<div class="hv-meta">{meta}</div></div>', unsafe_allow_html=True)


def _clip_card(r, out):
    vid = out / r.get("file", "")
    if r.get("file") and vid.exists():
        st.video(str(vid))
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hv-noclip">video stored on the VM</div>', unsafe_allow_html=True)
    missed = {c["object"] for c in r["checks"] if not c["yolo_found"]}
    ph = set(r.get("phantoms", []))
    cos = "".join(f'<span class="hv-chip {"miss" if o["name"] in missed else "ok"}">{_e(o["name"].replace("_", " "))}'
                  f' ×{_e(o["count"])}</span>' for o in r["inventory"]) or '<span class="hv-chip">nothing</span>'
    yol = "".join(f'<span class="hv-chip {"ph" if y in ph else ""}">{_e(y)}</span>' for y in list(r["yolo"])[:8]) \
        or '<span class="hv-chip">nothing</span>'
    if r.get("error"):
        verdict = '<span class="hv-pill grey">judge error</span>'
    elif missed:
        verdict = f'<span class="hv-pill red">{len(missed)} missed</span>'
    elif ph:
        verdict = f'<span class="hv-pill amber">{len(ph)} made up</span>'
    else:
        verdict = '<span class="hv-pill green">correct</span>'
    summ = f'<div class="r" style="color:#334155;margin-bottom:4px">{_e(r["summary"])}</div>' if r.get("summary") else ""
    st.markdown(f'<div class="hv-clip"><div class="t"><span>{_e(_cam(r["camera_id"]))}</span>{verdict}</div>{summ}'
                f'<div class="r"><span class="k">Cosmos</span>{cos}</div>'
                f'<div class="r"><span class="k">YOLO11</span>{yol}</div></div>', unsafe_allow_html=True)


LEGEND = ("Green = Cosmos saw it and YOLO11 found it. Red = Cosmos saw it, YOLO11 missed it. "
          "Amber = YOLO11 reported it, Cosmos did not see it.")


def _run_name(p):
    n = sum(1 for _ in open(p / "clips.jsonl"))
    t = p.name.split("_", 1)[-1]
    if len(t) >= 9 and t[:4].isdigit():
        t = f"{t[:2]}/{t[2:4]} {t[5:7]}:{t[7:9]}"
    return f"Audit run {t} · {n} clips"


def overview(out=None):
    """A full camera audit. `out` given: show that run (the Audit report page picks it)."""
    runs = [p for p in _runs("audit") if (p / "report.json").exists()]
    head = st.container()
    if out is None and runs:
        out = runs[0] if len(runs) == 1 else st.selectbox(
            "Audit run", runs, format_func=_run_name, label_visibility="collapsed")
    rep = json.load(open(out / "report.json")) if out else {}
    run = rep.get("run", {})
    meta = ""
    if run:
        meta = (f'Archive <b>{run.get("archive_segments") or 0:,}</b> segments · VAST<br>'
                f'Judge <b>NVIDIA Cosmos3-Reason</b><br>Model under audit <b>YOLO11 (COCO)</b>')
    with head:
        _header("Model audit", "Where the deployed vision model is blind, measured against the video archive, "
                "with the clips to retrain it.", meta)
    st.markdown("""<div class="hv-steps">
<div class="hv-step"><div class="n">1 · VAST Data</div><b>Sample</b><span>Clips per camera, with the YOLO11 detections stored at ingest</span></div>
<div class="hv-step"><div class="n">2 · NVIDIA Cosmos</div><b>Judge</b><span>Cosmos3-Reason lists what each clip really contains</span></div>
<div class="hv-step"><div class="n">3 · Harvest</div><b>Compare</b><span>Missed objects and made-up labels, by camera</span></div>
<div class="hv-step"><div class="n">4 · Weights &amp; Biases</div><b>Fix</b><span>Failing clips become the retraining set; every audit is logged</span></div>
</div>""", unsafe_allow_html=True)
    if not out:
        st.info("No audit yet. Run one from the terminal: python3 -m harvest.audit --wandb")
        return
    recs = [json.loads(l) for l in open(out / "clips.jsonl")]
    ok = [r for r in recs if not r.get("error")]
    objs = rep.get("objects", {})
    blind = [o for o, v in objs.items() if (v["rate"] or 0) == 0 and v["seen"] >= 2]
    ph_clips = sum(1 for r in ok if r.get("phantoms"))
    fails = [r for r in ok if any(not c["yolo_found"] for c in r["checks"])]
    kpis = [("Clips audited", f"{len(ok)}", ""), ("Cameras", f"{len(run.get('cameras', []))}", ""),
            ("Objects never detected", f"{len(blind)}", "bad"),
            ("Clips with a made-up label", f"{round(100 * ph_clips / max(1, len(ok)))}%", "bad"),
            ("Clips to retrain on", f"{len(fails)}", "")]
    st.markdown('<div class="hv-kpis">' + "".join(
        f'<div class="hv-kpi"><div class="l">{_e(l)}</div><div class="v {c}">{_e(v)}</div></div>'
        for l, v, c in kpis) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="hv-h2">Detection rate by object</div><div class="hv-sub">Clips where YOLO11 reported '
                'the object, out of clips where Cosmos3-Reason saw it. "Not in model" = the class does not exist in '
                'COCO; only retraining fixes it.</div>', unsafe_allow_html=True)
    mis = rep.get("mislabels", {})
    rows = []
    for o, v in sorted(objs.items(), key=lambda kv: ((kv[1]["rate"] or 0), -kv[1]["seen"])):
        r = v["rate"] or 0
        t = _tone(r)
        status = ('<span class="hv-pill red">Not in model</span>' if not v["has_class"] else
                  f'<span class="hv-pill {t}">{"Reliable" if t == "green" else "Unreliable" if t == "amber" else "Failing"}</span>')
        inst = ", ".join(f"{_e(m['yolo_label'])} ({m['clips']})" for m in mis.get(o, [])[:3]) if not v["has_class"] else ""
        rows.append(f'<tr><td class="hv-obj">{_e(o.replace("_", " "))}</td>'
                    f'<td><span class="hv-bar"><div style="width:{max(2, round(100 * r))}%;background:{_COLOR[t]}"></div></span>'
                    f'<b>{round(100 * r)}%</b></td><td>{v["detected"]} / {v["seen"]}</td><td>{status}</td>'
                    f'<td style="color:#64748B">{inst or "—"}</td></tr>')
    st.markdown('<table class="hv-table"><tr><th>Object</th><th>Detected</th><th>Clips</th><th>Status</th>'
                '<th>YOLO11 reported instead</th></tr>' + "".join(rows) + "</table>", unsafe_allow_html=True)

    per_cam = {}
    for r in ok:
        per_cam[r["camera_id"]] = per_cam.get(r["camera_id"], 0) + 1
    top = {}
    for row in rep.get("phantoms_by_camera", []):
        top.setdefault(row["camera"], row)
    if top:
        st.markdown('<div class="hv-h2">Most frequent made-up label per camera</div><div class="hv-sub">Labels '
                    'YOLO11 reported that nothing in the clip explains.</div>', unsafe_allow_html=True)
        cards = sorted(top.items(), key=lambda kv: -kv[1]["clips"] / per_cam.get(kv[0], 1))
        st.markdown('<div class="hv-phs">' + "".join(
            f'<div class="hv-ph"><div class="q">“{_e(row["yolo_label"])}”</div>'
            f'<div class="m">{row["clips"]} of {per_cam.get(cam, "?")} clips · {_e(_cam(cam))}</div></div>'
            for cam, row in cards) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="hv-h2">Breakdown</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["By camera", "By condition", "Report JSON"])
    with t1:
        bc = pd.DataFrame(rep.get("by_camera", []))
        if not bc.empty:
            import altair as alt
            bc["camera"] = bc["camera"].map(_cam)
            bc["pct"] = (bc["rate"].fillna(0) * 100).round().astype(int)
            order = sorted(objs, key=lambda o: objs[o]["rate"] or 0)
            base = alt.Chart(bc).encode(x=alt.X("camera:N", title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
                                        y=alt.Y("object:N", title=None, sort=order, axis=alt.Axis(labelFontSize=12)))
            heat = base.mark_rect(cornerRadius=3, stroke="white", strokeWidth=2).encode(
                color=alt.Color("pct:Q", scale=alt.Scale(domain=[0, 50, 100], range=["#DC2626", "#F59E0B", "#16A34A"]),
                                legend=alt.Legend(title="% detected")),
                tooltip=["object", "camera", "detected", "seen", "pct"])
            text = base.mark_text(fontSize=12, fontWeight=600, color="white").encode(text=alt.Text("pct:Q", format="d"))
            st.altair_chart((heat + text).properties(height=max(240, 32 * bc["object"].nunique()))
                            .configure_view(strokeWidth=0), use_container_width=True)
    with t2:
        cond = pd.DataFrame(rep.get("by_condition", []))
        if not cond.empty:
            cond = cond[["object", "condition", "value", "detected", "seen", "rate"]].copy()
            cond["rate"] = (cond["rate"].fillna(0) * 100).round().astype(int).astype(str) + "%"
            st.dataframe(cond, use_container_width=True, hide_index=True)
    with t3:
        st.json(rep, expanded=False)

    st.markdown(f'<div class="hv-h2">Retraining set · {len(fails)} clips</div>'
                f'<div class="hv-sub">{LEGEND}</div>', unsafe_allow_html=True)
    cams = sorted({r["camera_id"] for r in fails})
    f = st.radio("Camera", ["All"] + cams, horizontal=True, format_func=lambda c: c if c == "All" else _cam(c),
                 label_visibility="collapsed")
    shown = [r for r in fails if f == "All" or r["camera_id"] == f]
    cols = st.columns(3)
    for i, r in enumerate(shown[:9]):
        with cols[i % 3]:
            _clip_card(r, out)
    b1, b2, _ = st.columns([1, 1, 2])
    if b1.button("Export retraining set", type="primary", use_container_width=True):
        z = audit.export_retrain(out)
        b1.download_button("Download .zip", open(z, "rb"), file_name=f"{out.name}_retrain.zip", use_container_width=True)
    if b2.button("Log to Weights & Biases", use_container_width=True):
        with st.spinner("Logging to W&B..."):
            audit.log_wandb(out)
        b2.success("Logged.")
    st.markdown('<div class="hv-foot">Cosmos3-Reason acts as the judge, not hand labels: spot-check its calls '
                'before acting on a rate. YOLO11 results are read from VAST, so the audit adds no detector cost.</div>',
                unsafe_allow_html=True)
