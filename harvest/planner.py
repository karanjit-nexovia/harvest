"""The search planner, on Weights & Biases Inference.

A request in your own words ("forklift reversing near a worker") becomes 2-3 VAST search phrases and the
camera to search. No W&B key, or the call fails: the request itself is the one search.

    python -m harvest.planner "people carrying boxes"
"""
import json
import os
import re
import sys

from . import config

CAMERAS = {"sdg_warehouse_cam-2": "warehouse aisles, forklifts, pallets, workers (ceiling cam)",
           "smartspace_cam-1": "indoor facility, people walking, carrying things",
           "i24_cam-1": "highway traffic, multi-camera", "pie_cam-3": "dashcam driving in Toronto",
           "neighborhood_cam-1": "residential street, cars", "sf_streets_cam-1": "San Francisco streets"}

PROMPT = """You plan searches over an indexed video archive to collect training clips for a vision model.
Cameras: {cams}
Request: "{request}"
Return ONLY JSON: {{"queries": [2-3 short visual search phrases describing the moment on camera],
 "camera_id": one camera id from the list or null, "why": "<one sentence>"}}"""


def model():
    return os.getenv("WANDB_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def plan(request):
    """-> {"queries": [...], "camera_id": str|None, "why": str, "planner": str}"""
    fallback = {"queries": [request], "camera_id": None, "why": "planner off", "planner": "none"}
    key = os.getenv("WANDB_API_KEY")
    if not key:
        return fallback
    try:
        from openai import OpenAI
        team, proj = os.getenv("WANDB_TEAM"), os.getenv("WANDB_PROJECT", config.WANDB_PROJECT)
        client = OpenAI(base_url="https://api.inference.wandb.ai/v1", api_key=key,
                        project=f"{team}/{proj}" if team else None)
        r = client.chat.completions.create(
            model=model(), temperature=0.2, max_tokens=300,
            messages=[{"role": "user", "content": PROMPT.format(cams=json.dumps(CAMERAS), request=request)}])
        data = json.loads(re.search(r"\{.*\}", r.choices[0].message.content or "", re.S).group(0))
        queries = [str(q) for q in data.get("queries", []) if str(q).strip()][:3] or [request]
        cam = data.get("camera_id")
        return {"queries": queries, "camera_id": cam if cam in CAMERAS else None,
                "why": str(data.get("why", "")), "planner": f"W&B Inference ({model()})"}
    except Exception as e:  # noqa: BLE001 -- never block a run on the planner
        return dict(fallback, why=f"planner failed: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    print(json.dumps(plan(" ".join(sys.argv[1:]) or "people carrying boxes"), indent=1))
