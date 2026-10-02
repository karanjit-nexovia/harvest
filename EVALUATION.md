# Blindspot: inputs, outputs, evaluation

## Sponsor tools
| Tool | Role in Blindspot | Code |
|---|---|---|
| **VAST Data** (DataEngine, VastDB, VSS API) | the archive, the search used to sample it, and the YOLO11 detections being audited | `harvest/vss.py` |
| **NVIDIA Cosmos3-Reason** | the judge: what is really in each clip | `harvest/audit.py: inventory()` |
| **YOLO11** (ran at ingest) | the model under audit | read through `/api/v1/videos/detections` |
| **Weights & Biases** | each audit logged as a run: detection rate, count recall, failures table | `audit.log_wandb()` |
| **W&B Inference** (Llama-3.1-8B) | turns a plain-English request into queries and a camera (Harvest page) | `harvest/planner.py` |
| **CoreWeave GPUs** | hosts Cosmos3-Reason, YOLO11 and Cosmos Embed | event stack |

## Inputs
- The camera packs to audit (e.g. `sdg_warehouse_cam-2, i24_cam-1, pie_cam-3, smartspace_cam-1`) and the clips per camera.
- Team credentials from `/config/<team>.config` or the VM env. Nothing is hard-coded or printed.

## Outputs (`out/audit_MMDD_HHMM/`)
| File | What |
|---|---|
| `clips/*.mp4` | the audited clips |
| `clips.jsonl` | per clip: camera, Cosmos inventory + conditions, YOLO classes and objects per frame, per-object checks (`yolo_found`, `yolo_avg_count`, `count_recall`), `phantoms` |
| `report.json` | headline; per object: seen, detected, rate, `count_recall`, has_class; phantoms; by camera; by condition; retrain list |
| `retrain.zip` | the failing clips plus their Cosmos labels, ready for annotation and retraining |

## Metrics
- **Detection rate** (per object) = clips where YOLO named the object ÷ clips where Cosmos saw it.
- **Count recall** (per object) = mean over clips of min(1, YOLO objects per frame ÷ Cosmos count). It catches the case where the detection rate looks perfect but YOLO only sees half the cars.
- **Phantom rate** = clips where YOLO reported a class Cosmos did not see.
- **Has class**: if COCO has no class for the object (forklift, pallet, box, cart, rack, cone), no threshold change will fix it, only retraining will.

## Limits (stated honestly)
- Cosmos is the judge, not ground truth. Spot-check a sample of its inventories by eye before trusting a rate.
- Count recall compares a per-frame average with Cosmos's "typically visible at once" count, so it is approximate. Use it to rank failures, not as an exact figure.
- Sampling goes through search, so clips are biased toward what the queries match.

## Reproduce
```bash
python3 -m harvest.audit --cameras sdg_warehouse_cam-2,i24_cam-1,pie_cam-3,smartspace_cam-1 --per-camera 12 --wandb
python3 -m streamlit run app.py --server.port 8501   # Blindspot page: charts, failures, export
```

---

# Harvest page: inputs, outputs, evaluation

## Sponsor tools used
| Tool | Where |
|---|---|
| **VAST Data** (DataEngine index + VastDB + search API) | `harvest/vss.py`: login, `POST /api/v1/search`, segment stream, YOLO sidecars, archive size |
| **NVIDIA Cosmos3-Reason** (video reasoning) | `harvest/segment.py`: each clip → action label + timestamped steps (guided JSON, schema-checked) |
| **YOLO11** (detections, computed at ingest) | reused as the free filter in front of Cosmos (`HARVEST_NEED=person`) |
| **W&B Inference** (serverless LLM) | `harvest/planner.py`: the robot-skill request → 2–3 search queries + camera |
| **Weights & Biases** (experiments) | `harvest/evaluate.py`, `harvest/train.py`: accuracy, cost, clip table, confusion matrix |
| **CoreWeave GPUs** | Cosmos3-Reason, YOLO11 and Cosmos Embed run there (the event stack) |

## Inputs
- `data/raw/*.mp4` — the videos to mine (the event's sample videos).
- A plain-English query, e.g. `"person takes an item from a shelf"`.
- `.env` — endpoints and keys (see `.env.example`). No keys: `HARVEST_MOCK=1`.

## Outputs (`out/<query>/`)
| File | What |
|---|---|
| `clips/*.mp4` | One cut clip per kept range |
| `clips.jsonl` | One record per clip: source video, start/end, search score, YOLO stats, Cosmos `label`, `steps` [{name, start_s, end_s, conf}], GPU seconds |
| `stats.json` | The funnel: video searched → candidate ranges → YOLO kept → Cosmos labelled; total YOLO and Cosmos seconds |
| `labels.jsonl` | Hand labels (ground truth), made on the Label page |
| `eval.json` | Accuracy vs hand labels + cost table |
| `dataset/` + `dataset.zip` | The exported training set: clips + `annotations.jsonl` + README |
| `step_model.joblib`, `train.json` | The model trained on the harvest, and its score on unseen clips |

Step vocabulary: `reach, grasp, lift_or_pull, move, place, conceal, release, idle`.
Labels: `take_item, return_item, conceal_item, open_drawer, pour, other`.

## Evaluation steps (reproduce)
```bash
python -m harvest.pipeline "person takes an item from a shelf" --k 50   # 1. harvest
streamlit run app.py   # 2. Label page: hand-label >= 20 clips BEFORE viewing Cosmos's answers
python -m harvest.evaluate out/person_takes_an_item_from_a_shelf --wandb # 3. accuracy + cost
python -m harvest.train    out/person_takes_an_item_from_a_shelf --wandb # 4. close the loop
python -m harvest.export   out/person_takes_an_item_from_a_shelf         # 5. dataset.zip
```

## Metrics
**Label quality (Cosmos vs hand labels):**
- `label_acc` — clip label matches.
- `step_recall` / `step_precision` — a hand step counts as found when Cosmos gives a step of the same
  name overlapping it with IoU ≥ 0.5 (each Cosmos step used once).
- `boundary_err_s` — mean |difference| of matched step start/end times, seconds.

**Cost (the cascade):**
- A = brute force: Cosmos on every segment in the archive (Cosmos seconds per clip × archive size).
- B = Harvest: VAST search + YOLO detections already in the index + Cosmos only on the kept hits.
- Reported as GPU seconds, `saving_x = A / B`, and $ per usable clip (`GPU_DOLLARS_PER_HOUR`).

**Does the data teach?** (`harvest/train.py`)
- A small pose-based step classifier trained on the harvested labels, split **by clip**; reported
  as frame accuracy and macro-F1 on unseen clips, against always guessing the most common step.

## Known limits
- Cosmos labels are the teacher; the hand-labelled set is small (≈20 clips), so accuracy has wide
  error bars. More hand labels tighten it.
- The YOLO filter (`HAND_ACTIVITY`) trades recall for cost; tune it per video set.
