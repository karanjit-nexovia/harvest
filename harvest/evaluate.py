"""Step 4: how good and how cheap. Compares Cosmos steps to hand labels, and prices the cascade.

    python -m harvest.evaluate out/<run> [--labels labels.jsonl] [--wandb]

labels.jsonl: one {"clip_id", "label", "steps":[{"name","start_s","end_s"}]} per hand-labelled clip
(make them with the Label page of the app, BEFORE looking at Cosmos's answers).

Cost: A = Cosmos on every searched range (cosmos seconds per clip x all ranges);
      B = YOLO on every range + Cosmos only on the kept ones (what Harvest does)."""
import argparse
import os
import json
from pathlib import Path

from . import config


def iou(a, b):
    inter = max(0.0, min(a["end_s"], b["end_s"]) - max(a["start_s"], b["start_s"]))
    union = max(a["end_s"], b["end_s"]) - min(a["start_s"], b["start_s"])
    return inter / union if union > 0 else 0.0


def score(pred, truth):
    """Label accuracy, step recall / precision at IoU >= 0.5 (same name), mean boundary error."""
    n = lab = tp = n_true = n_pred = 0
    errs = []
    for cid, t in truth.items():
        p = pred.get(cid)
        if p is None:
            continue
        n += 1
        lab += p["label"] == t["label"]
        n_true += len(t["steps"])
        n_pred += len(p["steps"])
        used = set()
        for ts in t["steps"]:
            best = max(((iou(ps, ts), j) for j, ps in enumerate(p["steps"])
                        if j not in used and ps["name"] == ts["name"]), default=(0, None))
            if best[1] is not None and best[0] >= 0.5:
                used.add(best[1]); tp += 1
                ps = p["steps"][best[1]]
                errs += [abs(ps["start_s"] - ts["start_s"]), abs(ps["end_s"] - ts["end_s"])]
    return {"clips_compared": n, "label_acc": round(lab / n, 3) if n else None,
            "step_recall": round(tp / n_true, 3) if n_true else None,
            "step_precision": round(tp / n_pred, 3) if n_pred else None,
            "boundary_err_s": round(sum(errs) / len(errs), 2) if errs else None}


def purity(pred, truth):
    """Of the clips a person checked: how many raw search hits truly show the request, and how many
    of the clips Harvest verified do. Plus how well the verifier agrees with the person."""
    checked = [(pred[c], t) for c, t in truth.items() if c in pred and "relevant" in t]
    if not checked:
        return {}
    raw = sum(1 for _p, t in checked if t["relevant"]) / len(checked)
    kept = [(p, t) for p, t in checked if p.get("verified", True)]
    agree = sum(1 for p, t in checked if bool(p.get("verified", True)) == bool(t["relevant"])) / len(checked)
    tp = sum(1 for p, t in checked if p.get("verified", True) and t["relevant"])
    true_n = sum(1 for _p, t in checked if t["relevant"])
    return {"clips_checked": len(checked), "raw_search_precision": round(raw, 3),
            "harvest_precision": round(sum(1 for _p, t in kept if t["relevant"]) / len(kept), 3) if kept else None,
            "verifier_agreement": round(agree, 3),
            "verifier_recall": round(tp / true_n, 3) if true_n else None, "kept": len(kept)}


def cost(stats):
    per_cosmos = stats["cosmos_s"] / max(1, stats["kept"])
    # A = Cosmos on every segment of the archive to find the same examples (the brute-force way);
    # without an archive count, on every search hit. B = Harvest: search + reused YOLO + Cosmos on
    # the kept hits only.
    a = per_cosmos * (stats.get("archive_segments") or stats["ranges"])
    b = stats["yolo_s"] + stats["cosmos_s"]
    usable = max(1, stats["labelled"])
    d = config.GPU_DOLLARS_PER_HOUR / 3600
    return {"A_cosmos_everything_gpu_s": round(a, 1), "B_harvest_gpu_s": round(b, 1),
            "saving_x": round(a / b, 1) if b else None,
            "A_usd_per_usable_clip": round(a * d / usable, 4), "B_usd_per_usable_clip": round(b * d / usable, 4),
            "video_searched_s": stats["searched_video_s"], "kept_of_ranges": f"{stats['kept']}/{stats['ranges']}",
            "archive_segments": stats.get("archive_segments"),
            "cosmos_s_per_clip": round(per_cosmos, 2)}


def evaluate(run_dir, labels=None, use_wandb=False):
    run_dir = Path(run_dir)
    recs = [json.loads(l) for l in open(run_dir / "clips.jsonl")]
    stats = json.load(open(run_dir / "stats.json"))
    pred = {r["clip_id"]: r for r in recs}
    labels = Path(labels) if labels else run_dir / "labels.jsonl"
    truth = {}
    if labels.exists():
        for l in open(labels):
            t = json.loads(l); truth[t["clip_id"]] = t
    acc = score(pred, {k: v for k, v in truth.items() if v.get("relevant", True)}) if truth else {}
    pur = purity(pred, truth) if truth else {}
    c = cost(stats)
    out = {"purity": pur, "accuracy": acc, "cost": c, "stats": stats}
    json.dump(out, open(run_dir / "eval.json", "w"), indent=1)
    if use_wandb:
        import wandb
        wb = wandb.init(project=config.WANDB_PROJECT, entity=os.getenv("WANDB_TEAM") or None, name=run_dir.name, config=stats)
        wb.log({**{f"purity/{k}": v for k, v in pur.items() if v is not None},
                **{f"acc/{k}": v for k, v in acc.items() if v is not None},
                **{f"cost/{k}": v for k, v in c.items() if isinstance(v, (int, float))}})
        table = wandb.Table(columns=["clip", "harvest_verified", "person_says_relevant", "evidence", "label",
                                     "steps", "truth_label", "video"])
        for r in recs[:50]:
            t = truth.get(r["clip_id"], {})
            table.add_data(r["clip_id"], r.get("verified"), t.get("relevant"), r.get("evidence", ""), r["label"],
                           " > ".join(s["name"] for s in r["steps"]), t.get("label"),
                           wandb.Video(str(run_dir / r["file"]), format="mp4"))
        if pur:
            wb.log({"purity_bar": wandb.plot.bar(wandb.Table(
                data=[["Raw VAST search", pur["raw_search_precision"]],
                      ["Harvest (Cosmos-verified)", pur["harvest_precision"] or 0]], columns=["dataset", "precision"]),
                "dataset", "precision", title="Share of clips that truly show the request")})
        wb.log({"clips": table, "cost_bar": wandb.plot.bar(
            wandb.Table(data=[["Cosmos on everything", c["A_cosmos_everything_gpu_s"]],
                              ["Harvest (YOLO -> Cosmos)", c["B_harvest_gpu_s"]]], columns=["pipeline", "gpu_s"]),
            "pipeline", "gpu_s", title="GPU seconds")})
        wb.finish()
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--labels")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    evaluate(a.run_dir, a.labels, a.wandb)
