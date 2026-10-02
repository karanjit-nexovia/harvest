"""The launch page: a full-screen intro (loader -> story -> 'What would you like to train today?') that hands the
choice to the Build dataset page and starts it."""
import json
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from harvest import audit, config, flow
from harvest.ui_audit import CAMERAS

_landing = components.declare_component("harvest_landing", path=str(Path(__file__).parent / "landing"))

FULLSCREEN = """
<style>
section[data-testid="stSidebar"], header[data-testid="stHeader"], [data-testid="stSidebarCollapsedControl"],
[data-testid="stExpandSidebarButton"] { display: none !important; }
.stApp { background: #04060C !important; }
iframe[title*="harvest_landing"] { position: fixed !important; inset: 0 !important; width: 100vw !important;
  height: 100vh !important; z-index: 999999 !important; border: 0 !important; }
</style>
"""

DESCRIPTIONS = {
    "Forklift training": "Forklifts driving and lifting. The deployed YOLO11 has no forklift class at all.",
    "Close call training": "A forklift, vehicle or person comes dangerously close to another. Trains near-miss alerts.",
    "Break-in training": "Someone trying doors, climbing fences or looking into windows. Trains security alerts.",
    "Traffic analysis": "Dense, merging and heavy traffic. Trains traffic analytics and driving models.",
    "E-scooter training": "Electric scooter riders on streets and sidewalks. YOLO11 has no scooter class.",
}


def _stats():
    """Real numbers from the newest full audit, for the story."""
    runs = sorted([p for p in config.OUT.glob("audit_*") if (p / "report.json").exists()],
                  key=lambda p: -p.stat().st_mtime)
    s = {"archive": 2352}
    if not runs:
        return s
    try:
        rep = json.load(open(runs[0] / "report.json"))
        objs = rep.get("objects", {})
        s["archive"] = (rep.get("run") or {}).get("archive_segments") or 2352
        blind = sorted([(o, v) for o, v in objs.items() if not v["has_class"]], key=lambda kv: -kv[1]["seen"])
        if blind:
            s.update(blind_obj=blind[0][0], blind_seen=blind[0][1]["seen"], blind_detected=blind[0][1]["detected"])
        s["never"] = sum(1 for v in objs.values() if (v["rate"] or 0) == 0 and v["seen"] >= 2)
        per_cam = {}
        for l in open(runs[0] / "clips.jsonl"):
            c = json.loads(l)["camera_id"]
            per_cam[c] = per_cam.get(c, 0) + 1
        hw = [r for r in rep.get("phantoms_by_camera", []) if r["camera"] == "i24_cam-1"]
        if hw:
            s.update(ph_label=hw[0]["yolo_label"], ph_n=hw[0]["clips"], ph_of=per_cam.get("i24_cam-1", 12))
    except Exception:  # noqa: BLE001 -- the story falls back to the numbers in the page
        pass
    return s


def show(skip_loader=False):
    st.markdown(FULLSCREEN, unsafe_allow_html=True)
    presets = [{"name": k, "desc": DESCRIPTIONS.get(k, ""), "queries": v["queries"],
                "cameras": [CAMERAS.get(c, c) for c in v["cameras"]]} for k, v in flow.PRESETS.items()]
    return _landing(skip_loader=skip_loader, stats=_stats(), presets=presets,
                    objects=list(audit.OBJECTS), lighting=audit.CONDITIONS["lighting"],
                    cameras=[{"id": c, "name": CAMERAS.get(c, c)} for c in flow.ALL_CAMERAS],
                    key="landing", default=None)


def apply(v):
    """Turn the landing page's choice into the Build dataset page's inputs, and start it."""
    ss = st.session_state
    ss["entered"] = True
    ss["nav"] = "Build dataset"
    if v.get("target"):
        ss["target"] = ss["_keep_target"] = int(v["target"])
    ss["per_round"] = ss.get("_keep_per_round") or 15
    if v.get("action") == "preset":
        ss["mode"] = "Pick a use case"
        ss["preset"] = v["preset"]
        ss["autorun"] = True
    elif v.get("action") == "describe":
        ss["mode"] = "Describe your own training data"
        ss["custom_uc"] = v.get("text", "")
        own = [k for k in (audit.object_key(o) for o in v.get("objects", [])) if k and k not in audit.OBJECTS]
        ss["custom_objs"] = ss["_keep_custom_objs"] = ", ".join(o.replace("_", " ") for o in own)
        ss["custom_must"] = [m for m in (audit.object_key(x) for x in v.get("must", [])) if m in audit.OBJECTS or m in own]
        ss["custom_cams"] = [c for c in v.get("cameras", []) if c in flow.ALL_CAMERAS]
        ss["custom_light"] = v.get("lighting") if v.get("lighting") in ["any"] + audit.CONDITIONS["lighting"] else "any"
        ss["autorun"] = True


def gate():
    """Show the landing page until the visitor makes a choice. Call before drawing the app."""
    if st.session_state.get("entered") or st.query_params.get("intro") == "0":
        return
    v = show(skip_loader=st.session_state.get("seen_loader", False))
    if v and v.get("nonce") != st.session_state.get("landing_nonce"):
        st.session_state["landing_nonce"] = v["nonce"]
        apply(v)
        st.rerun()
    st.stop()
