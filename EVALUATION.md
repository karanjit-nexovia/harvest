# Harvest: inputs, outputs, and how we evaluate it

## Sponsor tools used
| Tool | Where |
|---|---|
| **VAST Data** (ingest + semantic search) | `harvest/search.py` (`SEARCH_BACKEND=vast`): natural-language query → candidate time ranges |
| **NVIDIA Cosmos** (video reasoning) | `harvest/segment.py`: each clip → action label + timestamped steps (JSON, schema-checked) |
| **YOLO** (detection + tracking + pose) | `harvest/detect.py`: cheap filter in front of Cosmos (person present + hands working) |
| **Weights & Biases** | `harvest/evaluate.py`, `harvest/train.py`: accuracy, cost, clip table, confusion matrix |
| **CoreWeave GPUs** (if used) | `YOLO_DEVICE=0` and Cosmos inference |

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
- A = Cosmos run on every candidate range (Cosmos seconds per clip × all ranges).
- B = YOLO on every range + Cosmos only on the kept ones (what Harvest does).
- Reported as GPU seconds, `saving_x = A / B`, and $ per usable clip (`GPU_DOLLARS_PER_HOUR`).

**Does the data teach?** (`harvest/train.py`)
- A small pose-based step classifier trained on the harvested labels, split **by clip**; reported
  as frame accuracy and macro-F1 on unseen clips, against always guessing the most common step.

## Known limits
- Cosmos labels are the teacher; the hand-labelled set is small (≈20 clips), so accuracy has wide
  error bars. More hand labels tighten it.
- The YOLO filter (`HAND_ACTIVITY`) trades recall for cost; tune it per video set.
