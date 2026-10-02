# Harvest

**Turn unlabeled video into robot training data.** Ask for an action in plain English, get back
segmented, labelled clips: semantic search → YOLO filter → NVIDIA Cosmos step segmentation →
dataset export, with accuracy and GPU-cost numbers in Weights & Biases.

```
query ─► search (VAST / CLIP / motion) ─► YOLO + pose filter (cheap, edge-able)
      ─► cut clips ─► Cosmos: reach / grasp / pull / … with timestamps (JSON)
      ─► eval vs hand labels + cost vs "Cosmos on everything" ─► export dataset.zip
      ─► train: a small pose model learns the steps from the harvest (scored on unseen clips)
```

## Run it (5 min)

```bash
git clone https://github.com/karanjit-nexovia/harvest && cd harvest
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # put the event's keys in
# drop the sample videos in data/raw/
streamlit run app.py
```

No keys yet? `HARVEST_MOCK=1 streamlit run app.py` runs everything with fake Cosmos answers.

CLI:
```bash
python -m harvest.pipeline "person takes an item from a shelf" --k 50
python -m harvest.evaluate out/person_takes_an_item_from_a_shelf --wandb
python -m harvest.export   out/person_takes_an_item_from_a_shelf
python -m harvest.train    out/person_takes_an_item_from_a_shelf --wandb
```

## Wiring the event's stack (.env)

| Piece | Setting |
|---|---|
| Cosmos | `COSMOS_BASE_URL`, `COSMOS_API_KEY`, `COSMOS_MODEL` — any OpenAI-compatible endpoint. `COSMOS_INPUT=frames` (works with any vision model) or `video` (sends the mp4). |
| Semantic search | `SEARCH_BACKEND=vast` + `VAST_SEARCH_URL` (POST `{"query","k"}` → `[{"video","start","end","score"}]`). Adapt `harvest/search.py::_vast` to the event API. `clip` = local CLIP, `motion` = no model (always works). |
| YOLO | `YOLO_MODEL`, `YOLO_DEVICE` (`0` on a GPU box), `HAND_ACTIVITY` (filter strictness) |
| W&B | `wandb login`, then Evaluate page → "Log to Weights & Biases" |
| Cost | `GPU_DOLLARS_PER_HOUR` |

## Demo checklist
1. Precompute 2–3 queries before demos (the Harvest page lists past runs instantly).
2. Label 20 clips on the **Label** page *before* looking at Cosmos's steps.
3. **Evaluate** → W&B report: label accuracy, step recall/precision, boundary error, GPU saving ×.
4. **Train** page → "Train step model": the student model's step accuracy on unseen clips, with
   Cosmos (teacher) and student timelines side by side. That's "data → working model in an afternoon".
5. Record a backup screen video.

Test videos in `data/raw/` are Intel IoT DevKit samples (CC BY 4.0). Use the event's videos for the demo.
