"""Blind-spot finder: where the detector in production can't see what's there.

The VAST pipeline runs YOLO11 (COCO, 80 classes) at ingest. COCO has no forklift, pallet or cart, so in
warehouse footage those objects are missed or mislabelled ("truck", "suitcase", "boat"). For every
clip Cosmos verified, compare the objects Cosmos saw with the classes YOLO reported: an object Cosmos
saw that YOLO never named is a blind spot -- and the verified clip is training data to fix it.

    python -m harvest.blindspots out/<run>
"""
import collections
import json
import re
import sys
from pathlib import Path

# Cosmos's words -> the COCO class that would cover them (None: COCO has no such class)
COCO_FOR = {"person": "person", "worker": "person", "man": "person", "woman": "person", "people": "person",
            "forklift": None, "pallet": None, "cart": None, "trolley": None, "box": None, "boxes": None,
            "crate": None, "shelf": None, "shelves": None, "rack": None, "barrel": None, "cone": None,
            "car": "car", "vehicle": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
            "bicycle": "bicycle", "bike": "bicycle", "pedestrian": "person", "traffic light": "traffic light",
            "bag": "handbag", "backpack": "backpack", "suitcase": "suitcase", "chair": "chair", "bench": "bench"}


def _canon(name):
    n = re.sub(r"[^a-z ]", "", str(name).lower()).strip()
    for k in sorted(COCO_FOR, key=len, reverse=True):
        if re.search(rf"\b{k}\b", n):
            return k
    return n or None


def find(run_dir):
    run_dir = Path(run_dir)
    recs = [json.loads(l) for l in open(run_dir / "clips.jsonl")]
    recs = [r for r in recs if not r.get("error") and r.get("verified", True)]
    missed = collections.Counter()
    no_class = collections.Counter()
    seen = collections.Counter()
    examples = collections.defaultdict(list)
    for r in recs:
        yolo = {c.lower() for c in (r.get("tracks") or {})}
        for obj in {_canon(o) for o in r.get("objects", []) if _canon(o)}:
            seen[obj] += 1
            coco = COCO_FOR.get(obj, "?")
            if coco is None:
                no_class[obj] += 1
                examples[obj].append(r["clip_id"])
            elif coco != "?" and coco not in yolo:
                missed[obj] += 1
                examples[obj].append(r["clip_id"])
    out = {"clips": len(recs),
           "no_coco_class": {k: {"clips": v, "of": seen[k], "examples": examples[k][:10]} for k, v in no_class.most_common()},
           "missed_by_yolo": {k: {"clips": v, "of": seen[k], "examples": examples[k][:10]} for k, v in missed.most_common()},
           "yolo_labels_seen": collections.Counter(c for r in recs for c in (r.get("tracks") or {})).most_common(10)}
    json.dump(out, open(run_dir / "blindspots.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(find(sys.argv[1]), indent=1))
