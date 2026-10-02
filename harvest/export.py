"""Step 5: one folder (and zip) that a robotics team can train on.

    python -m harvest.export out/<run>   ->  out/<run>/dataset/ + dataset.zip"""
import json
import shutil
import sys
from pathlib import Path


def export(run_dir):
    run_dir = Path(run_dir)
    recs = [json.loads(l) for l in open(run_dir / "clips.jsonl")]
    good = [r for r in recs if not r.get("error") and r["label"] != "other" and r["steps"]]
    ds = run_dir / "dataset"
    shutil.rmtree(ds, ignore_errors=True)
    (ds / "clips").mkdir(parents=True)
    with open(ds / "annotations.jsonl", "w") as fh:
        for r in good:
            shutil.copy(run_dir / r["file"], ds / r["file"])
            fh.write(json.dumps({k: r[k] for k in ("clip_id", "file", "label", "steps", "objects",
                                                   "video", "start", "end", "query")}) + "\n")
    ev = {}
    if (run_dir / "eval.json").exists():
        ev = json.load(open(run_dir / "eval.json"))
    (ds / "README.md").write_text(
        f"# Harvest dataset: {recs[0]['query'] if recs else ''}\n\n"
        f"{len(good)} clips with step annotations (from {len(recs)} candidates).\n\n"
        f"Accuracy vs hand labels: {json.dumps(ev.get('accuracy', {}))}\n\n"
        f"Cost: {json.dumps(ev.get('cost', {}))}\n\n"
        "Steps: reach, grasp, lift_or_pull, move, place, conceal, release, idle (seconds from clip start).\n"
        "Source videos: see `video` per record; check their licence before redistribution.\n")
    zip_path = shutil.make_archive(str(run_dir / "dataset"), "zip", ds)
    print(f"{len(good)} clips -> {zip_path}")
    return zip_path, len(good)


if __name__ == "__main__":
    export(sys.argv[1])
