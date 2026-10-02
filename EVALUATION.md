# Harvest: inputs, outputs and evaluation

## Inputs
- **What to train:** a use case (Forklift, Close call, Break-in, E-scooter, Traffic) or your own description.
- **Optional:** cameras, objects every clip must show (built-in or your own words), lighting, how many good
  clips you need (default 6) and how many to check per round (default 15).
- **Credentials:** VAST and Cosmos from the event VM (`/config/<team>.config` or its environment); W&B from
  `WANDB_API_KEY`, `WANDB_TEAM`, `WANDB_PROJECT`. Nothing is hard-coded or printed.

## Outputs

A build writes `out/flow_<time>/`:

| File | What |
|---|---|
| `state.json` | the plan (searches, cameras), every clip checked (Cosmos's verdict and reason, objects, conditions, events, YOLO11's labels, missed and made-up objects, your review), the comparison, Cosmos's suggestions |
| `clips/*.mp4` | every clip checked |
| `dataset/` + `<use_case>_dataset.zip` | the exported dataset: `clips/`, `frames/` (3 per clip), `annotations.jsonl` (one row per kept clip: Cosmos's objects, events and conditions, YOLO11's labels, `yolo_missed`, `yolo_made_up`, frame paths), `README.md` with the suggested model changes |
| **W&B run + artifact** | `cosmos_vs_yolo` table (every clip with video, kept or removed, reason), detection-rate chart, suggested changes, the dataset as a versioned artifact |

A camera audit writes `out/audit_<time>/` (`clips.jsonl`, `report.json`, `retrain.zip`).

## Metrics

| Metric | Definition | Where |
|---|---|---|
| **Search precision** | clips kept after Cosmos and your review ÷ clips checked: how much of raw search would have polluted a dataset | Dataset report |
| **Cosmos precision** | (clips Cosmos kept − clips you removed) ÷ clips Cosmos kept | W&B run summary |
| **YOLO11 label accuracy** | YOLO11 labels Cosmos agrees with ÷ (agreed + missed + made up), on the kept clips | Dataset report |
| **Detection rate** (per object) | clips where YOLO11 reported the object ÷ clips where Cosmos saw it | Build page step 4, Audit report |
| **Made-up rate** | clips where YOLO11 reported a label nothing in the clip explains | Audit report |
| **Not in model** | the object has no YOLO11 (COCO) class: only retraining can fix it | everywhere |

## Limits (stated honestly)
- Cosmos is the judge, not ground truth. Harvest's review step exists for that, and Cosmos's precision is
  logged on every export.
- Labels are per clip and per event, with extracted frames. There are no bounding boxes yet.
- Clips come through search, so a dataset leans toward what the searches match; rounds widen the search.
- The “after retraining” column of the Dataset report is empty until you retrain on the dataset and run
  Harvest again. Harvest does not claim an improvement it has not measured.

## Reproduce
```bash
python3 -m harvest.flow "Close call training" --target 6 --wandb
python3 -m harvest.audit --cameras sdg_warehouse_cam-2,i24_cam-1,pie_cam-3,smartspace_cam-1 --per-camera 12 --wandb
python3 -m streamlit run app.py --server.port 8501
```
