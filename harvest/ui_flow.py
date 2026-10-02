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


def _card(r, out, review=False):
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
    removed = r.get("review") == "removed"
    pill = ('<span class="hv-pill red">✗ Removed by you</span>' if removed else
            '<span class="hv-pill green">✓ Kept</span>' if r["matches"] else
            '<span class="hv-pill grey">✗ Rejected by Cosmos</span>')
    why = ("You marked this clip as not a match. Cosmos had said: " if removed else "") + (
        r.get("match_reason") or r.get("summary") or "")
    st.markdown(f'<div class="fl-card {"" if r["matches"] else "rej"}"><div class="t"><span>{_e(_cam(r["camera_id"]))}</span>'
                f'{pill}</div><div class="why">{_e(why)}</div>'
                f'<div class="r"><span class="k">Cosmos</span>{cos}</div>'
                f'<div class="r"><span class="k">YOLO11</span>{yol}</div></div>', unsafe_allow_html=True)
    if review and (r["matches"] or removed):
        key = f"rv_{out.name}_{r['clip_id']}"
        if removed:
            if st.button("↺ Undo", key=key, use_container_width=True):
                flow.review(out, r["clip_id"], "undo")
                st.rerun()
        elif st.button("✗ Not a match: remove", key=key, use_container_width=True,
                       help="Take this clip out of the dataset. Harvest can then find a replacement."):
            flow.review(out, r["clip_id"], "remove")
            st.rerun()


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
                + (f' over <b>{state["rounds"]} rounds</b>' if state.get("rounds", 1) > 1 else "")
                + (f'<br>Cosmos checks each clip against: <i>“{_e(state["request"])}”</i>' if state.get("request") else "")
                + ('<br>Your own objects: ' + "".join(f'<span class="fl-q">{_e(o.replace("_", " "))}</span>'
                   for o in (state.get("spec") or {}).get("objects", [])) if (state.get("spec") or {}).get("objects") else "")
                + (f'<br><span style="color:#B45309">{len(state["skipped"])} clip(s) skipped: the VAST video server did not '
                   'return them (replaced with spares)</span>' if state.get("skipped") else "")
                + '</div>', unsafe_allow_html=True)

    _sec(3, f"Cosmos check: {len(kept)} of {len(clips)} clips kept",
         "NVIDIA Cosmos3-Reason watched every clip and kept only the ones that really show the use case."
         + (f" Target: {state['target']} good clips." if state.get("target") else ""))
    cosmos_kept = sum(1 for r in clips if r.get("cosmos_matches", r["matches"]))
    if state.get("target") and not state.get("reached_target") and cosmos_kept < state["target"]:
        why = ("the archive has no more unseen matches" if state.get("exhausted")
               else f"stopped after {state.get('rounds')} round{'s' if state.get('rounds', 0) != 1 else ''}")
        st.markdown(f'<div class="fl-box" style="border-color:#FDE68A;background:#FFFBEB">Only <b>{cosmos_kept}</b> of '
                    f'the {state["target"]} good clips needed: {why}. Cosmos rejected the rest as not showing '
                    f'“{_e(state.get("use_case", ""))}”. Try a broader description or other cameras.</div>',
                    unsafe_allow_html=True)
    removed = [r for r in clips if r.get("review") == "removed"]
    need = max(0, (state.get("target") or 0) - len(kept))
    if clips:
        st.markdown(f'<div class="fl-box">Review the kept clips. If one is not a match, press <b>Not a match</b> and '
                    f'it leaves the dataset. <b>{len(kept)}</b> kept'
                    + (f' · <b style="color:#DC2626">{len(removed)}</b> removed by you' if removed else "")
                    + (f' · <b>{need}</b> more needed for {state["target"]}' if need else " · target met")
                    + '</div>', unsafe_allow_html=True)
        if need and removed:
            r1, _ = st.columns([1.4, 3])
            with r1:
                st.markdown('<div class="fl-indent">', unsafe_allow_html=True)
                go = st.button(f"Find {need} replacement{'s' if need > 1 else ''}", type="primary",
                               use_container_width=True, key=f"repl_{out.name}")
            if go:
                status = st.empty()
                live = st.columns(3)
                got = []

                def on_clip(rec, o):
                    got.append(rec)
                    with live[(len(got) - 1) % 3]:
                        _card(rec, o)
                    status.markdown(f'<div class="hv-live">Cosmos checked <b>{len(got)}</b> new clips · kept '
                                    f'<b>{sum(x["matches"] for x in got)}</b> of {need} needed</div>',
                                    unsafe_allow_html=True)

                status.markdown('<div class="hv-live">Searching the archive for clips not seen yet…</div>',
                                unsafe_allow_html=True)
                flow.replace(out, on_clip=on_clip)
                st.rerun()
        cols = st.columns(3)
        for i, r in enumerate(sorted(clips, key=lambda r: (not r["matches"], r.get("review") != "removed"))):
            with cols[i % 3]:
                _card(r, out, review=True)
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
        _sec(5, "What Cosmos suggests changing in the model", f"Written by {_e(sug.get('by', ''))} from the comparison above."
             + (" Your review changed the dataset: these refresh on export or after replacements." if state.get("suggestions_stale") else ""))
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
        c1, c2, c3, _ = st.columns([1.3, 1, 1.2, 0.8])
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
        def _to_report(path=str(out)):
            st.session_state["nav"] = "Audit report"
            st.session_state["report_pick"] = path

        c3.button("See dataset report →", key=f"rep_{out.name}", on_click=_to_report, use_container_width=True,
                  help="Before vs after Harvest: search results + YOLO11 labels vs this dataset")
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
    ss = st.session_state
    # Streamlit drops a widget's value while it is off screen (e.g. on the launch page): keep a copy
    for k, v in {"mode": "Pick a use case", "target": 6, "per_round": 15, "custom_light": "any",
                 "custom_objs": ""}.items():
        if ss.get(k) is None:
            ss[k] = ss.get(f"_keep_{k}", v)
    mode = st.radio("Mode", ["Pick a use case", "Describe your own training data"], horizontal=True,
                    label_visibility="collapsed", key="mode")
    spec = {}
    from harvest import audit as _audit
    extra_txt = st.text_input("Also have Cosmos look for (your own objects, comma-separated)", key="custom_objs",
                              placeholder="e.g. can, tree, fire extinguisher, ladder")
    extra = [k for k in dict.fromkeys(_audit.object_key(x) for x in extra_txt.split(",")) if k and k not in _audit.OBJECTS]
    if extra:
        spec["objects"] = extra
        st.markdown('<div class="hv-sub" style="margin:-6px 0 10px">' + " ".join(
            f'<span class="hv-pill {"green" if _audit.coco_for(k) else "red"}">{_e(k.replace("_", " "))}: '
            f'{"YOLO11 class " + _e(_audit.coco_for(k)) if _audit.coco_for(k) else "not in YOLO11"}</span>' for k in extra)
            + '</div>', unsafe_allow_html=True)
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
        a1, a2, a3 = st.columns([2, 2, 1])
        must_opts = list(_audit.OBJECTS) + extra
        ss["custom_must"] = [m for m in (ss.get("custom_must") or []) if m in must_opts]
        spec["cameras"] = a1.multiselect("Cameras (empty = Harvest chooses)", flow.ALL_CAMERAS, format_func=_cam,
                                         key="custom_cams")
        spec["must"] = a2.multiselect("Every clip must show", must_opts,
                                      format_func=lambda o: o.replace("_", " "), key="custom_must")
        spec["lighting"] = a3.selectbox("Lighting", ["any"] + _audit.CONDITIONS["lighting"], key="custom_light")
        st.caption("Your description searches the archive (planned by W&B Inference) and is what Cosmos checks every "
                   "clip against. Clips missing a must-show object, or in the wrong lighting, are rejected.")
    b0, t, b, c = st.columns([3.2, 1.3, 1.3, 1.4])
    b0.markdown('<div class="hv-sub" style="margin-top:30px">Harvest keeps checking new clips, round after round '
                '(up to 4), until Cosmos has kept enough good ones.</div>', unsafe_allow_html=True)
    target = t.number_input("Good clips needed", 1, 30, key="target")
    n = b.number_input("Clips per round", 3, 30, key="per_round")
    c.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
    run = c.button("Build dataset", type="primary", use_container_width=True, disabled=not (uc or "").strip())
    for k in ("mode", "target", "per_round", "custom_light", "custom_objs"):
        ss[f"_keep_{k}"] = ss.get(k)
    # arriving from the launch page: start straight away
    run = (ss.pop("autorun", False) and bool((uc or "").strip())) or run

    if run:
        _stepper(0, now=0)
        status = st.empty()
        grid = st.container()
        cols3 = grid.columns(3)
        seen = []
        status.markdown('<div class="hv-live">Planning the search and searching VAST…</div>', unsafe_allow_html=True)

        prog = {"kept": 0, "round": 1}

        def on_step(k, v):
            if k == "found":
                prog["round"] = v["round"]
                status.markdown(f'<div class="hv-live">Round <b>{v["round"]}</b>: found <b>{v["clips"]}</b> new clips in '
                                f'VAST. Cosmos is checking them… (kept {v["kept"]} of {int(target)} needed)</div>',
                                unsafe_allow_html=True)
            elif k == "round_done" and v["kept"] < v["target"]:
                status.markdown(f'<div class="hv-live">Round {v["round"]}: only <b>{v["kept"]}</b> of {v["target"]} good '
                                'clips so far. Searching deeper in the archive…</div>', unsafe_allow_html=True)
            elif k == "compared":
                status.markdown('<div class="hv-live">Compared with YOLO11. Cosmos is writing suggestions…</div>',
                                unsafe_allow_html=True)

        def on_clip(rec, out):
            seen.append(rec)
            with cols3[(len(seen) - 1) % 3]:
                _card(rec, out)
            prog["kept"] += rec["matches"]
            status.markdown(f'<div class="hv-live">Round <b>{prog["round"]}</b> · Cosmos checked <b>{len(seen)}</b> clips · '
                            f'kept <b>{prog["kept"]}</b> of {int(target)} needed</div>', unsafe_allow_html=True)

        out, _ = flow.run(uc.strip(), int(n), on_step=on_step, on_clip=on_clip, spec=spec, target=int(target))
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
