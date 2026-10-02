"""Step 2: the cheap filter. YOLO-pose tracks people and their wrists in each candidate range; a
range is kept if a person is there >= 1.5 s and their hands are working: wrist travel relative to
the body (reaching, grabbing, pulling) above HAND_ACTIVITY, or an object detected at the hands.
Kept ranges are cut to clip files. GPU (or CPU) seconds are timed for the cost table."""
import os
import subprocess
import time

import numpy as np

import cv2

from . import config

OBJECTS = {39: "bottle", 41: "cup", 67: "cell phone", 26: "handbag", 24: "backpack", 73: "book",
           46: "banana", 47: "apple", 64: "mouse", 76: "scissors", 25: "umbrella", 28: "suitcase"}
_MODEL = {}


def model():
    if "m" not in _MODEL:
        from ultralytics import YOLO
        _MODEL["m"] = YOLO(config.YOLO_MODEL)
        _MODEL["pose"] = YOLO(os.getenv("YOLO_POSE_MODEL", "yolo11n-pose.pt"))
    return _MODEL["m"]


def pose_model():
    model()
    return _MODEL["pose"]


def _hand_zone(p):
    """Where hands are: the person box's lower 2/3, widened 25% each side."""
    x1, y1, x2, y2 = p
    w, h = x2 - x1, y2 - y1
    return x1 - 0.25 * w, y1 + h / 3, x2 + 0.25 * w, y2


def _overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def screen(rng):
    """Run YOLO on one candidate range. Returns (keep, stats)."""
    path = config.RAW / rng["video"]
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / config.YOLO_FPS)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(rng["start"] * fps))
    m, pm = model(), pose_model()
    for mm in (m, pm):
        for tr in getattr(getattr(mm, "predictor", None), "trackers", None) or []:
            tr.reset()                               # fresh track ids per range
    t0 = time.time()
    person_time, contact, people, frames = {}, 0, set(), 0
    wrists = {}                                      # track -> [(wrist offset from shoulders / body height)]
    i = int(rng["start"] * fps)
    while i <= rng["end"] * fps:
        ok = cap.grab()
        if not ok:
            break
        if (i - int(rng["start"] * fps)) % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            frames += 1
            pr = pm.track(f, persist=True, verbose=False, conf=0.3, device=config.YOLO_DEVICE or None)[0]
            if pr.keypoints is not None and pr.boxes.id is not None:
                for t, b, k in zip(pr.boxes.id.int().tolist(), pr.boxes.xyxy.tolist(), pr.keypoints.data.tolist()):
                    h = max(1.0, b[3] - b[1])
                    sh = [p for p in (k[5], k[6]) if p[2] > 0.4]
                    for w in (k[9], k[10]):
                        if w[2] > 0.4 and sh:
                            cx, cy = sum(p[0] for p in sh) / len(sh), sum(p[1] for p in sh) / len(sh)
                            wrists.setdefault(t, []).append(((w[0] - cx) / h, (w[1] - cy) / h))
            r = m.track(f, persist=True, verbose=False, classes=[0] + list(OBJECTS), conf=0.25,
                        device=config.YOLO_DEVICE or None)[0]
            boxes = r.boxes
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            persons = [(b, t) for b, c, t in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist(), ids) if c == 0]
            objs = [b for b, c in zip(boxes.xyxy.tolist(), boxes.cls.int().tolist()) if c in OBJECTS]
            for b, t in persons:
                people.add(t)
                person_time[t] = person_time.get(t, 0) + step / fps
            if any(_overlaps(_hand_zone(p), o) for p, _ in persons for o in objs):
                contact += 1
        i += 1
    cap.release()
    secs = time.time() - t0
    longest = max(person_time.values(), default=0.0)
    # hand activity: how far the wrists range relative to the shoulders, in body heights (the
    # widest-ranging person) -- standing still ~0.1, reaching to a shelf and back ~0.4+
    activity = max((float(np.ptp(np.array(v), axis=0).max()) for v in wrists.values() if len(v) >= 4), default=0.0)
    keep = longest >= 1.5 and (activity >= float(os.getenv("HAND_ACTIVITY", "0.35"))
                               or contact >= int(os.getenv("HARVEST_MIN_CONTACT", "5")))
    return keep, {"people": len(people), "person_s": round(longest, 1), "hand_object_frames": contact,
                  "hand_activity": round(activity, 2), "frames": frames, "yolo_s": round(secs, 2)}


def cut(rng, out_path):
    """Cut [start, end] of the source video to an mp4 (re-encoded: exact cuts, small, browser-safe)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{rng['start']:.2f}", "-i", str(config.RAW / rng["video"]),
                    "-t", f"{rng['end'] - rng['start']:.2f}", "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "26", "-an", "-movflags", "+faststart", str(out_path)],
                   check=True)
    return out_path
