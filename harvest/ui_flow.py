"""Build a dataset: the one-page Harvest flow. Use case -> clips -> Cosmos check -> vs YOLO -> suggestions -> W&B."""
import json
import time

import streamlit as st

from harvest import flow
from harvest.ui_audit import _cam, _e, _header

STEPS = ["Find clips", "Cosmos check", "Compare with YOLO11", "Suggestions", "Export"]

FLOW_CSS = """
<style>
.fl-steps { display:flex; gap:0; margin:4px 0 22px; border:1px solid #E2E8F0; border-radius:10px; overflow:hidden; }
.fl-step { flex:1; padding:10px 14px; font-size:13px; color:#94A3B8; background:#FFF; border-right:1px solid #E2E8F0; }
.fl-step:last-child { border-right:none; }
.fl-step b { display:block; font-size:11px; letter-spacing:.5px; }
.fl-step.done { color:#15803D; background:#F0FDF4; } .fl-step.now { color:#1D4ED8; background:#EFF6FF; font-weight:600; }
.fl-sec { display:flex; gap:12px; align-items:baseline; margin:30px 0 4px; }
.fl-num { width:26px; height:26px; border-radius:50%; background:#0F172A; color:#FFF; font-size:13px; font-weight:700;
          display:inline-flex; align-items:center; justify-content:center; flex-shrink:0; }
.fl-title { font-size:17px; font-weight:600; color:#0F172A; }
.fl-sub { font-size:13px; color:#64748B; margin:0 0 12px 38px; }
.fl-box { border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px; background:#F8FAFC; font-size:13.5px;
          color:#0F172A; margin-left:38px; }
.fl-q { display:inline-block; background:#FFF; border:1px solid #E2E8F0; border-radius:6px; padding:1px 8px; margin:2px 4px 2px 0; font-size:12.5px; }
.fl-sug { border:1px solid #E2E8F0; border-left:4px solid #2563EB; border-radius:8px; padding:11px 14px; margin:0 0 8px 38px; background:#FFF; }
.fl-sug b { color:#0F172A; font-size:14px; } .fl-sug div { color:#64748B; font-size:12.5px; margin-top:2px; }
.fl-card { border:1px solid #E2E8F0; border-top:none; border-radius:0 0 10px 10px; padding:10px 12px 12px; font-size:12.5px;
           margin:-16px 0 16px; background:#FFF; }
.fl-card.rej { background:#FAFAFA; opacity:.75; }
.fl-card .t { display:flex; justify-content:space-between; font-weight:600; color:#0F172A; margin-bottom:3px; }
.fl-card .why { color:#334155; margin-bottom:6px; } .fl-card .r { margin-top:3px; }
.fl-card .k { display:inline-block; width:52px; color:#64748B; font-weight:500; }
.fl-indent { margin-left:38px; }
</style>
"""


def _sec(n, title, sub=""):
    st.markdown(f'<div class="fl-sec"><span class="fl-num">{n}</span><span class="fl-title">{_e(title)}</span></div>'
                + (f'<div class="fl-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def _stepper(done, now=None):
    st.markdown('<div class="fl-steps">' + "".join(
        f'<div class="fl-step {"done" if i < done else "now" if i == now else ""}"><b>STEP {i + 2}</b>'
        f'{"✓ " if i < done else ""}{_e(s)}</div>' for i, s in enumerate(STEPS)) + "</div>", unsafe_allow_html=True)


def _card(r, out):
    vid = out / r["file"]
    if vid.exists():
        st.video(str(vid))
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="hv-noclip">video stored on the VM</div>', unsafe_allow_html=True)
    missed = {c["object"] for c in r["checks"] if not c["yolo_found"]}
    ph = set(r.get("phantoms", []))
    cos = "".join(f'<span class="hv-chip {"miss" if o["name"] in missed else "ok"}">{_e(o["name"].replace("_", " "))}</span>'
                  for o in r["inventory"]) or '<span class="hv-chip">nothing</span>'
    yol = "".join(f'<span class="hv-chip {"ph" if y in ph else ""}">{_e(y)}</span>' for y in sorted(r["yolo"])[:8]) \
        or '<span class="hv-chip">nothing</span>'
    pill = ('<span class="hv-pill green">✓ Kept</span>' if r["matches"] else
            '<span class="hv-pill grey">✗ Rejected</span>')
    st.markdown(f'<div class="fl-card {"" if r["matches"] else "rej"}"><div class="t"><span>{_e(_cam(r["camera_id"]))}</span>'
                f'{pill}</div><div class="why">{_e(r.get("match_reason") or r.get("summary") or "")}</div>'
                f'<div class="r"><span class="k">Cosmos</span>{cos}</div>'
                f'<div class="r"><span class="k">YOLO11</span>{yol}</div></div>', unsafe_allow_html=True)


def _results(state, out):
    clips = state.get("clips", [])
    kept = [r for r in clips if r["matches"]]
    p = state.get("plan", {})
    done = 1 + bool(clips) + bool(state.get("report")) + bool(state.get("suggestions")) + bool(state.get("export"))
    st.markdown(f'<div class="hv-sub" style="margin:14px 0 6px">Results for <b style="color:#0F172A">'
                f'{_e(state.get("use_case", ""))}</b> · {len(clips)} clips · '
                f'{state.get("seconds") or 0:.0f} s</div>', unsafe_allow_html=True)
    _stepper(min(done, 5), now=min(done, 4))

    _sec(2, "Clips found in the VAST archive",
         f"Searched {state.get('archive_segments') or 0:,} indexed segments. Search plan by "
         f"{_e(p.get('by', ''))}.")
    st.markdown('<div class="fl-box">' + "".join(f'<span class="fl-q">“{_e(q)}”</span>' for q in p.get("queries", []))
                + ' on ' + ", ".join(_e(_cam(c)) for c in p.get("cameras", []))
                + f' → <b>{len(state.get("found", []))} clips</b>'
                + (f'<br>Cosmos checks each clip against: <i>“{_e(state["request"])}”</i>' if state.get("request") else "")
                + (f'<br><span style="color:#B45309">{len(state["skipped"])} clip(s) skipped: the VAST video server did not '
                   'return them (replaced with spares)</span>' if state.get("skipped") else "")
                + '</div>', unsafe_allow_html=True)

    _sec(3, f"Cosmos check: {len(kept)} of {len(clips)} clips kept",
         "NVIDIA Cosmos3-Reason watched every clip and kept only the ones that really show the use case.")
    if clips:
        cols = st.columns(3)
        for i, r in enumerate(sorted(clips, key=lambda r: not r["matches"])):
            with cols[i % 3]:
                _card(r, out)
        st.markdown('<div class="hv-sub fl-indent">Green = Cosmos saw it and YOLO11 found it · Red = YOLO11 missed it · '
                    'Amber = YOLO11 made it up</div>', unsafe_allow_html=True)

    rep = state.get("report") or {}
    objs = rep.get("objects") or {}
    if state.get("report") is not None:
        _sec(4, "Cosmos vs YOLO11 on the kept clips",
             "For each object Cosmos saw: how often the deployed YOLO11 detected it, and what it said instead.")
        if objs:
            mis = rep.get("mislabels") or {}
            rows = []
            for o, v in sorted(objs.items(), key=lambda kv: (kv[1]["rate"] or 0)):
                r = v["rate"] or 0
                tone = "green" if r >= .8 else "amber" if r >= .4 else "red"
                status = "Not in model" if not v["has_class"] else "Reliable" if tone == "green" else "Unreliable" if tone == "amber" else "Failing"
                inst = ", ".join(f"{_e(m['yolo_label'])} ({m['clips']})" for m in mis.get(o, [])[:3])
                rows.append(f'<tr><td class="hv-obj">{_e(o.replace("_", " "))}</td><td>{v["seen"]}</td>'
                            f'<td>{v["detected"]}</td><td><b>{round(100 * r)}%</b></td>'
                            f'<td><span class="hv-pill {"red" if not v["has_class"] else tone}">{status}</span></td>'
                            f'<td style="color:#64748B">{inst or "—"}</td></tr>')
            st.markdown('<div class="fl-indent"><table class="hv-table"><tr><th>Object</th><th>Cosmos saw it in</th>'
                        '<th>YOLO11 found it in</th><th>Rate</th><th>Status</th><th>YOLO11 said instead</th></tr>'
                        + "".join(rows) + "</table></div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="fl-box">No clips were kept, so there is nothing to compare.</div>',
                        unsafe_allow_html=True)

    sug = state.get("suggestions")
    if sug:
        _sec(5, "What Cosmos suggests changing in the model", f"Written by {_e(sug.get('by', ''))} from the comparison above.")
        for c in sug.get("changes", []):
            st.markdown(f'<div class="fl-sug"><b>{_e(c["change"])}</b><div>{_e(c["why"])}</div></div>',
                        unsafe_allow_html=True)

    if state.get("suggestions"):
        n_miss = sum(1 for r in kept for c in r["checks"] if not c["yolo_found"])
        n_ph = sum(len(r["phantoms"]) for r in kept)
        _sec(6, "The fixed dataset",
             f"{len(kept)} clips, labelled by Cosmos: {n_miss} objects YOLO11 missed are added and {n_ph} labels it "
             "made up are removed. Includes clips, frames, annotations.jsonl and the suggested changes.")
        exp = state.get("export")
        c1, c2, _ = st.columns([1.3, 1, 2])
        with c1:
            st.markdown('<div class="fl-indent">', unsafe_allow_html=True)
            go = st.button("Export to Weights & Biases", type="primary", use_container_width=True,
                           disabled=not kept, key=f"exp_{out.name}")
        if go:
            with st.spinner("Building the dataset and logging it to W&B..."):
                try:
                    flow.export(out, wandb_log=True)
                except Exception as e:  # noqa: BLE001 -- still give them the zip
                    flow.export(out, wandb_log=False)
                    st.warning(f"W&B logging failed ({type(e).__name__}); the dataset zip is ready below.")
            st.rerun()
        if exp:
            c2.download_button("Download .zip", open(exp["zip"], "rb"), file_name=exp["zip"].split("/")[-1].split("\\")[-1],
                               use_container_width=True, key=f"dl_{out.name}")
            if exp.get("wandb_url"):
                st.markdown(f'<div class="fl-box">✓ Logged to W&B: <a href="{_e(exp["wandb_url"])}" target="_blank">'
                            f'{_e(exp["wandb_url"])}</a> · table of every clip (Cosmos vs YOLO11, with video), '
                            'detection-rate chart, suggested changes, and the dataset as a versioned artifact.</div>',
                            unsafe_allow_html=True)
            if exp.get("artifact"):
                ref = exp["artifact"]
                st.markdown(f'<div class="fl-sec" style="margin-top:22px"><span class="fl-num">→</span>'
                            f'<span class="fl-title">Use this dataset</span></div><div class="fl-sub">Versioned in '
                            f'W&amp;B as <b>{_e(ref)}</b>. Every re-export of this use case becomes a new version, so '
                            'each model can be traced to the exact clips it was trained on. Browser: run page → '
                            'Artifacts → Files.</div>', unsafe_allow_html=True)
                st.markdown('<div class="fl-indent">', unsafe_allow_html=True)
                st.code(f'import wandb\nrun = wandb.init(project="{ref.split("/")[1]}", job_type="train")\n'
                        f'data_dir = run.use_artifact("{ref}").download()\n'
                        '# data_dir/annotations.jsonl, data_dir/clips/, data_dir/frames/', language="python")
                st.code(f"wandb artifact get {ref}", language="bash")


def page():
    st.markdown(FLOW_CSS, unsafe_allow_html=True)
    _header("Build a training dataset",
            "Describe what you want to train. Harvest finds matching clips in VAST, NVIDIA Cosmos3-Reason checks "
            "each one, compares it with the deployed YOLO11, and exports the fixed dataset to Weights & Biases.",
            "VAST Data · NVIDIA Cosmos<br>Weights &amp; Biases · CoreWeave")
    _sec(1, "What do you want to train?")
    mode = st.radio("Mode", ["Pick a use case", "Describe your own training data"], horizontal=True,
                    label_visibility="collapsed")
    spec = {}
    if mode == "Pick a use case":
        cols = st.columns(len(flow.PRESETS))
        for c, name in zip(cols, flow.PRESETS):
            if c.button(name, use_container_width=True,
                        type="primary" if st.session_state.get("preset") == name else "secondary"):
                st.session_state["preset"] = name
                st.rerun()
        uc = st.session_state.get("preset", "")
        if uc:
            p = flow.PRESETS[uc]
            st.markdown('<div class="fl-box" style="margin-left:0">Searches: ' + "".join(
                f'<span class="fl-q">“{_e(q)}”</span>' for q in p["queries"]) + " on " +
                ", ".join(_e(_cam(c)) for c in p["cameras"]) + "</div>", unsafe_allow_html=True)
    else:
        uc = st.text_area("Describe the training data you need", key="custom_uc", height=80,
                          placeholder="e.g. A forklift reversing while a worker walks behind it in an indoor warehouse")
        from harvest import audit as _audit
        a1, a2, a3 = st.columns([2, 2, 1])
        spec["cameras"] = a1.multiselect("Cameras (empty = Harvest chooses)", flow.ALL_CAMERAS, format_func=_cam)
        spec["must"] = a2.multiselect("Every clip must show", list(_audit.OBJECTS),
                                      format_func=lambda o: o.replace("_", " "))
        spec["lighting"] = a3.selectbox("Lighting", ["any"] + _audit.CONDITIONS["lighting"])
        st.caption("Your description searches the archive (planned by W&B Inference) and is what Cosmos checks every "
                   "clip against. Clips missing a must-show object, or in the wrong lighting, are rejected.")
    b0, b, c = st.columns([5, 1, 1.4])
    n = b.number_input("Clips", 2, 15, 6, label_visibility="collapsed")
    run = c.button("Build dataset", type="primary", use_container_width=True, disabled=not (uc or "").strip())

    if run:
        _stepper(0, now=0)
        status = st.empty()
        grid = st.container()
        cols3 = grid.columns(3)
        seen = []
        status.markdown('<div class="hv-live">Planning the search and searching VAST…</div>', unsafe_allow_html=True)

        def on_step(k, v):
            if k == "found":
                status.markdown(f'<div class="hv-live">Found <b>{len(v)}</b> clips. Cosmos is checking them…</div>',
                                unsafe_allow_html=True)
            elif k == "compared":
                status.markdown('<div class="hv-live">Compared with YOLO11. Cosmos is writing suggestions…</div>',
                                unsafe_allow_html=True)

        def on_clip(rec, out):
            seen.append(rec)
            with cols3[(len(seen) - 1) % 3]:
                _card(rec, out)
            status.markdown(f'<div class="hv-live">Cosmos checked <b>{len(seen)}</b> clips…</div>', unsafe_allow_html=True)

        out, _ = flow.run(uc.strip(), int(n), on_step=on_step, on_clip=on_clip, spec=spec)
        st.session_state["flow_dir"] = str(out)
        st.rerun()

    runs = flow.latest()
    if not runs:
        st.markdown('<div class="fl-box" style="margin-top:18px">Pick a use case above and press <b>Build dataset</b>.'
                    '</div>', unsafe_allow_html=True)
        return
    names = {str(p): p for p in runs}
    cur = st.session_state.get("flow_dir")
    if cur not in names:
        cur = str(runs[0])

    def label(p):
        s = json.load(open(p / "state.json"))
        return f"{s['use_case']} · {time.strftime('%H:%M', time.localtime(s.get('started', 0)))} · {len(s.get('clips', []))} clips"

    if len(runs) > 1:
        cur = st.selectbox("Run", list(names), index=list(names).index(cur), format_func=lambda k: label(names[k]),
                           label_visibility="collapsed")
    out = names[cur]
    _results(json.load(open(out / "state.json")), out)
