"""Step 6: close the loop -- train a small model on what Harvest produced.

Per frame of each exported clip, the main person's pose (YOLO-pose keypoints, normalised to the
body: relative to the shoulders, in body heights) plus how the wrists moved since the last frame.
Each frame takes the step name Cosmos gave that moment (reach / grasp / lift_or_pull / ...). A
small classifier learns step-from-pose over a short window; clips are split train/test by CLIP, so
the score is on clips the model never saw. Writes model.joblib + train.json, logs to W&B.

    python -m harvest.train out/<run> [--wandb]
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from . import config

FPS = 5.0
WIN = 3                       # frames either side in the feature window
_POSE = {}


def pose_model():
    if "m" not in _POSE:
        from ultralytics import YOLO
        _POSE["m"] = YOLO("yolo11n-pose.pt")
    return _POSE["m"]


def frame_poses(path):
    """[(t, 17x2 normalised keypoints or None)] for the largest person, at FPS."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step, i, out = max(1, int(round(fps / FPS))), 0, []
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            r = pose_model().predict(f, verbose=False, conf=0.3, device=config.YOLO_DEVICE or "cpu")[0]
            kp = None
            if r.keypoints is not None and len(r.boxes):
                b = r.boxes.xyxy.cpu().numpy()
                j = int(np.argmax((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])))
                k = r.keypoints.data.cpu().numpy()[j]
                h = max(1.0, b[j, 3] - b[j, 1])
                sh = k[[5, 6]]
                if (sh[:, 2] > 0.3).any():
                    c = sh[sh[:, 2] > 0.3, :2].mean(0)
                    kp = np.where(k[:, 2:3] > 0.3, (k[:, :2] - c) / h, 0.0)
            out.append((i / fps, kp))
        i += 1
    cap.release()
    return out


def features(poses):
    """Per frame: keypoints (34) + wrist velocity (4), averaged over +-WIN frames, + window std."""
    base = []
    prev = None
    for _t, kp in poses:
        if kp is None:
            v = np.zeros(38)
        else:
            wr = kp[[9, 10]].ravel()
            vel = wr - prev if prev is not None else np.zeros(4)
            prev = wr
            v = np.concatenate([kp.ravel(), vel])
        base.append(v)
    base = np.array(base) if base else np.zeros((0, 38))
    feats = []
    for i in range(len(base)):
        w = base[max(0, i - WIN):i + WIN + 1]
        feats.append(np.concatenate([w.mean(0), w.std(0)]))
    return np.array(feats)


def step_at(steps, t):
    for s in steps:
        if s["start_s"] <= t <= s["end_s"]:
            return s["name"]
    return "idle"


def build(run_dir):
    run_dir = Path(run_dir)
    recs = [json.loads(l) for l in open(run_dir / "clips.jsonl")]
    recs = [r for r in recs if not r.get("error") and r["steps"]]
    data = {}
    for r in recs:
        poses = frame_poses(run_dir / r["file"])
        X = features(poses)
        y = [step_at(r["steps"], t) for t, _ in poses]
        data[r["clip_id"]] = (X, y)
        print(f"{r['clip_id']}: {len(y)} frames", flush=True)
    return data


def train(run_dir, use_wandb=False, seed=7):
    from joblib import dump
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    data = build(run_dir)
    ids = sorted(data)
    random.Random(seed).shuffle(ids)
    n_test = max(1, len(ids) // 4)
    test, tr = ids[:n_test], ids[n_test:]
    Xtr = np.concatenate([data[i][0] for i in tr]); ytr = sum((data[i][1] for i in tr), [])
    Xte = np.concatenate([data[i][0] for i in test]); yte = sum((data[i][1] for i in test), [])
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    majority = max(set(ytr), key=ytr.count)
    out = {"clips_train": len(tr), "clips_test": len(test), "frames_train": len(ytr), "frames_test": len(yte),
           "frame_acc": round(accuracy_score(yte, pred), 3),
           "macro_f1": round(f1_score(yte, pred, average="macro"), 3),
           "baseline_majority_acc": round(sum(1 for y in yte if y == majority) / max(1, len(yte)), 3),
           "report": classification_report(yte, pred, output_dict=True, zero_division=0),
           "test_clips": test}
    dump(clf, Path(run_dir) / "step_model.joblib")
    json.dump(out, open(Path(run_dir) / "train.json", "w"), indent=1)
    if use_wandb:
        import wandb
        wb = wandb.init(project=config.WANDB_PROJECT, name=f"train-{Path(run_dir).name}", job_type="train")
        wb.log({"train/frame_acc": out["frame_acc"], "train/macro_f1": out["macro_f1"],
                "train/baseline_acc": out["baseline_majority_acc"],
                "train/confusion": wandb.plot.confusion_matrix(y_true=yte, preds=list(pred),
                                                               class_names=sorted(set(yte) | set(pred)))})
        wb.finish()
    print(json.dumps({k: v for k, v in out.items() if k != "report"}, indent=1))
    return out


def predict(run_dir, clip_path):
    """Step timeline for a new clip from the trained model: [(t, step)]."""
    from joblib import load
    clf = load(Path(run_dir) / "step_model.joblib")
    poses = frame_poses(clip_path)
    if not poses:
        return []
    return list(zip([t for t, _ in poses], clf.predict(features(poses))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--wandb", action="store_true")
    a = ap.parse_args()
    train(a.run_dir, a.wandb)
