"""The request planner, on Weights & Biases serverless inference (the event's LLM for app logic).

A robotics team writes what their robot must learn ("a robot that moves pallets in a warehouse").
The LLM turns that into 2-3 VAST search queries, the camera to search, and the action labels to
expect. No W&B key, or the call fails: the request itself is the one query.

    python -m harvest.planner "data for a robot that pushes carts"
"""
import json
import os
import re
import sys

from . import config

CAMERAS = {"sdg_warehouse_cam-2": "warehouse aisles, forklifts, pallets, workers (ceiling cam)",
           "smartspace_cam-1": "indoor facility, people walking, carrying things",
           "i24_cam-1": "highway traffic, multi-camera", "pie_cam-3": "dashcam driving in Toronto",
           "neighborhood_cam-1": "residential street, cars"}

PROMPT = """You plan searches over an indexed video archive to collect training examples for a vision
AI system (robots, autonomous vehicles, safety or retail analytics).
Cameras: {cams}
Domains: {domains}
Request: "{request}"
Return ONLY JSON: {{"domain": one domain key, "queries": [2-3 short visual search phrases describing the
 moment on camera], "camera_id": one camera id from the list or null, "why": "<one sentence>"}}"""


def model():
    return os.getenv("WANDB_LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


def plan(request):
    """-> {"queries": [...], "camera_id": str|None, "labels": [...], "why": str, "planner": str}"""
    fallback = {"queries": [request], "camera_id": None, "domain": config.DOMAIN, "labels": [],
                "why": "planner off", "planner": "none"}
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
            messages=[{"role": "user", "content": PROMPT.format(
                cams=json.dumps(CAMERAS), request=request,
                domains=json.dumps({k: v["desc"] for k, v in config.DOMAINS.items()}))}])
        text = r.choices[0].message.content or ""
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        queries = [str(q) for q in data.get("queries", []) if str(q).strip()][:3] or [request]
        cam = data.get("camera_id")
        dom = data.get("domain") if data.get("domain") in config.DOMAINS else config.DOMAIN
        return {"queries": queries, "camera_id": cam if cam in CAMERAS else None, "domain": dom,
                "labels": config.DOMAINS[dom]["labels"],
                "why": str(data.get("why", "")), "planner": f"W&B Inference ({model()})"}
    except Exception as e:  # noqa: BLE001 -- never block the harvest on the planner
        return dict(fallback, why=f"planner failed: {type(e).__name__}: {str(e)[:120]}")


if __name__ == "__main__":
    print(json.dumps(plan(" ".join(sys.argv[1:]) or "a robot that moves pallets"), indent=1))
