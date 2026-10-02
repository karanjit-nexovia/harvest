"""Audit report: pick a dataset you built (how much better it is than raw search + YOLO11's own labels) or a
full camera audit (where the deployed model is blind)."""
import collections
import json
import time
from pathlib import Path

import streamlit as st

from harvest import config, flow, ui_audit
from harvest.ui_audit import _cam, _e, _header

REPORT_CSS = """
<style>
.rp-ba { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin:6px 0 8px; }
.rp-col { border:1px solid #E2E8F0; border-radius:14px; padding:18px 20px; background:#FFF; }
.rp-col.before { background:#FBFBFC; } .rp-col.after { border-color:#BBF7D0; background:#F7FEF9; }
.rp-col h4 { font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:#64748B; margin:0 0 12px; font-weight:600; }
.rp-col.after h4 { color:#15803D; }
.rp-m { display:flex; align-items:baseline; gap:10px; margin:8px 0; }
.rp-m b { font-size:28px; font-weight:700; color:#0F172A; letter-spacing:-.5px; min-width:84px; }
.rp-col.before .rp-m b.bad { color:#DC2626; } .rp-col.after .rp-m b { color:#15803D; }
.rp-m span { font-size:13.5px; color:#475569; line-height:1.35; }
.rp-funnel { display:flex; align-items:stretch; gap:0; margin:4px 0 6px; border-radius:12px; overflow:hidden; border:1px solid #E2E8F0; }
.rp-f { flex:1; padding:12px 16px; background:#FFF; border-right:1px solid #E2E8F0; }
.rp-f:last-child { border-right:none; background:#F0FDF4; }
.rp-f b { display:block; font-size:24px; font-weight:700; color:#0F172A; }
.rp-f span { font-size:12px; color:#64748B; }
.rp-chips span { display:inline-block; font-size:12.5px; border:1px solid #E2E8F0; border-radius:999px; padding:3px 10px; margin:3px 4px 3px 0; background:#FFF; }
.rp-chips span b { color:#2563EB; margin-right:4px; }
.fl-box { border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px; background:#F8FAFC; font-size:13.5px; color:#0F172A; }
</style>
"""


def _flows():
    return [p for p in flow.latest() if (p / "state.json").exists()]


def _audits():
    return sorted([p for p in config.OUT.glob("audit_*") if (p / "report.json").exists()],
                  key=lambda p: -p.stat().st_mtime)


def _label(path):
    p = Path(path)
    if p.name.startswith("flow_"):
        s = json.load(open(p / "state.json"))
        kept = sum(1 for r in s.get("clips", []) if r["matches"])
        return (f"Dataset · {s.get('use_case', '')} · {kept} clips · "
                f"{time.strftime('%H:%M', time.localtime(s.get('started', 0)))}")
    return ui_audit._run_name(p).replace("Audit run", "Camera audit (all cameras)")


def _baseline():
    """Per-object detection rate from the newest full camera audit (the 'before' for the whole archive)."""
    a = _audits()
    if not a:
        return {}, 0
    rep = json.load(open(a[0] / "report.json"))
    return rep.get("objects", {}), sum(1 for _ in open(a[0] / "clips.jsonl"))


def dataset_report(out):
    out = Path(out)
    s = json.load(open(out / "state.json"))
    clips = s.get("clips", [])
    cosmos_kept = [r for r in clips if r.get("cosmos_matches", r["matches"])]
    approved = [r for r in clips if r["matches"]]
    removed = [r for r in clips if r.get("review") == "removed"]
    correct = sum(1 for r in approved for c in r["checks"] if c["yolo_found"])
    missed = sum(1 for r in approved for c in r["checks"] if not c["yolo_found"])
    made_up = sum(len(r.get("phantoms", [])) for r in approved)
    yolo_acc = correct / max(1, correct + missed + made_up)
    search_rel = len(approved) / max(1, len(clips))
    exp = s.get("export") or {}

    _header(f"Dataset report: {s.get('use_case', '')}",
            "How much better this dataset is than what you would get without Harvest: raw search results "
            "labelled by the deployed YOLO11.",
            f"Built {time.strftime('%H:%M', time.localtime(s.get('started', 0)))} · "
            f"{len(clips)} clips checked<br>Judge <b>NVIDIA Cosmos3-Reason</b> + your review")

    st.markdown('<div class="hv-h2">From search results to a clean dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="rp-funnel">'
                f'<div class="rp-f"><b>{s.get("archive_segments") or 0:,}</b><span>segments in the VAST archive</span></div>'
                f'<div class="rp-f"><b>{len(clips)}</b><span>clips Cosmos checked</span></div>'
                f'<div class="rp-f"><b>{len(cosmos_kept)}</b><span>Cosmos kept</span></div>'
                f'<div class="rp-f"><b>{len(removed)}</b><span>you removed</span></div>'
                f'<div class="rp-f"><b>{len(approved)}</b><span>in the dataset</span></div></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="hv-h2">Before and after Harvest</div><div class="hv-sub">Same clips, two ways of getting '
                'a dataset. "Before" is what a team gets today: search results, labelled by the model it already runs.'
                '</div>', unsafe_allow_html=True)
    acc_tone = "bad" if yolo_acc < .8 else ""
    rel_tone = "bad" if search_rel < .8 else ""
    st.markdown(f'''<div class="rp-ba">
<div class="rp-col before"><h4>Before: search + YOLO11 labels</h4>
 <div class="rp-m"><b class="{rel_tone}">{round(100 * search_rel)}%</b><span>of searched clips actually show “{_e(s.get("use_case", ""))}”
   ({len(approved)} of {len(clips)}); the rest would pollute the training set</span></div>
 <div class="rp-m"><b class="{acc_tone}">{round(100 * yolo_acc)}%</b><span>of YOLO11's labels on the kept clips are right
   ({correct} right, {missed} objects missed, {made_up} made up)</span></div>
 <div class="rp-m"><b class="bad">{missed + made_up}</b><span>label errors someone would have to find and fix by hand</span></div>
</div>
<div class="rp-col after"><h4>After: the Harvest dataset</h4>
 <div class="rp-m"><b>{len(approved)}</b><span>clips, every one checked by Cosmos{" and reviewed by you" if clips else ""}
   {f"({len(removed)} removed by you)" if removed else ""}</span></div>
 <div class="rp-m"><b>+{missed}</b><span>missing objects added to the labels (YOLO11 never saw them)</span></div>
 <div class="rp-m"><b>−{made_up}</b><span>made-up labels removed</span></div>
</div></div>''', unsafe_allow_html=True)

    # per object: this dataset vs the camera-audit baseline
    objs = (s.get("report") or {}).get("objects") or {}
    mis = (s.get("report") or {}).get("mislabels") or {}
    base, base_n = _baseline()
    if objs:
        st.markdown('<div class="hv-h2">What this dataset teaches the model</div><div class="hv-sub">For each object '
                    'in the dataset: how the deployed YOLO11 does today. The last column fills in when you retrain on '
                    'this dataset and run Harvest again.</div>', unsafe_allow_html=True)
        rows = []
        for o, v in sorted(objs.items(), key=lambda kv: (kv[1]["rate"] or 0)):
            r = v["rate"] or 0
            tone = "green" if r >= .8 else "amber" if r >= .4 else "red"
            b = base.get(o)
            btxt = (f'{round(100 * (b["rate"] or 0))}% <span style="color:#94A3B8">({b["detected"]}/{b["seen"]})</span>'
                    if b else '<span style="color:#94A3B8">not in the audit</span>')
            inst = ", ".join(f"{_e(m['yolo_label'])} ({m['clips']})" for m in mis.get(o, [])[:3])
            rows.append(f'<tr><td class="hv-obj">{_e(o.replace("_", " "))}</td><td><b>{v["seen"]}</b> clips</td>'
                        f'<td>{btxt}</td><td><span class="hv-pill {"red" if not v["has_class"] else tone}">'
                        f'{"Not in model" if not v["has_class"] else str(round(100 * r)) + "% detected"}</span></td>'
                        f'<td style="color:#64748B">{inst or "—"}</td>'
                        f'<td style="color:#94A3B8">retrain, then re-run</td></tr>')
        st.markdown('<table class="hv-table"><tr><th>Object</th><th>Examples in this dataset</th>'
                    f'<th>YOLO11 today, camera audit ({base_n} clips)</th><th>YOLO11 on these clips</th>'
                    '<th>YOLO11 said instead</th><th>After retraining</th></tr>' + "".join(rows) + "</table>",
                    unsafe_allow_html=True)

    # coverage
    cams = collections.Counter(_cam(r["camera_id"]) for r in approved)
    light = collections.Counter((r.get("conditions") or {}).get("lighting", "?") for r in approved)
    crowd = collections.Counter((r.get("conditions") or {}).get("crowding", "?") for r in approved)
    events = collections.Counter(e["type"].replace("_", " ") for r in approved for e in r.get("events", []))
    if approved:
        st.markdown('<div class="hv-h2">Coverage</div><div class="hv-sub">A model trained on one camera in daylight '
                    'fails everywhere else. What this dataset spans:</div>', unsafe_allow_html=True)

        def chips(title, c):
            return (f'<div class="rp-chips" style="margin-bottom:6px"><span style="border:none;background:none;'
                    f'color:#64748B;padding-left:0">{title}</span>' +
                    "".join(f'<span><b>{n}</b>{_e(k)}</span>' for k, n in c.most_common()) + "</div>")

        st.markdown(chips("Cameras", cams) + chips("Lighting", light) + chips("Crowding", crowd)
                    + (chips("Events Cosmos labelled", events) if events else ""), unsafe_allow_html=True)

    st.markdown('<div class="hv-h2">Where it lives</div>', unsafe_allow_html=True)
    if exp.get("artifact"):
        st.markdown(f'<div class="fl-box" style="margin-left:0">Exported to Weights &amp; Biases as <b>{_e(exp["artifact"])}</b>'
                    + (f' · <a href="{_e(exp["wandb_url"])}" target="_blank">open the run</a>' if exp.get("wandb_url") else "")
                    + '. Retrain on it, then run Harvest again on the new model to fill in the last column above.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="fl-box" style="margin-left:0">Not exported yet. Go back to <b>Build dataset</b> and press '
                    '<b>Export to Weights &amp; Biases</b> (step 6).</div>', unsafe_allow_html=True)
    st.markdown('<div class="hv-foot">“Right” and “missed” are judged by NVIDIA Cosmos3-Reason plus your review, not by '
                'hand-drawn boxes. Labels are per clip and per event, with extracted frames.</div>', unsafe_allow_html=True)


def page():
    st.markdown(ui_audit.CSS + REPORT_CSS, unsafe_allow_html=True)
    options = [str(p) for p in _flows()] + [str(p) for p in _audits()]
    if not options:
        _header("Audit report", "Build a dataset first, or run a camera audit.")
        return
    ss = st.session_state
    if ss.get("report_pick") not in options:
        ss["report_pick"] = options[0]
    st.selectbox("Report", options, format_func=_label, key="report_pick", label_visibility="collapsed")
    pick = Path(ss["report_pick"])
    if pick.name.startswith("flow_"):
        dataset_report(pick)
    else:
        ui_audit.overview(out=pick)
