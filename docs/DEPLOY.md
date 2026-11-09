# Deploying the clinical CDSS demo (Render + Qdrant Cloud)

Live site: **https://clinical-cdss.onrender.com**

## Why this architecture

| Target | Verdict | Reason |
|---|---|---|
| Vercel | ❌ cannot work | Serverless: no long-running Python process, no embedded Qdrant, ~100 MB index can't live in a function bundle |
| **Render web service** | ✅ chosen | Runs `python web_demo.py` as a normal process, injects `$PORT`, binds `0.0.0.0` automatically |
| **Qdrant Cloud** | ✅ chosen | The local index (34,206 chunks) is git-ignored; the hosted service needs a hosted vector store — `config.py` switches to it automatically when `QDRANT_URL` is set |

## One-time setup

### 1. Qdrant Cloud (index)

```powershell
# after .env has QDRANT_URL + QDRANT_API_KEY (never commit .env):
python scripts/sync_qdrant_cloud.py                # all 3 collections
python scripts/sync_qdrant_cloud.py --only symptom_disease   # resume a single one
```

The script copies every point local → cloud, prints per-collection progress, and
verifies counts, payload keys and a search round-trip. Idempotent (upserts by id).
**Never run `build_index.py` while `QDRANT_URL` is set** — it deletes and rebuilds
its collection on the target client.

### 2. Render (web service)

In the Render dashboard → your Web Service → **Environment**:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | from `.env` (paste in the dashboard, never in chat/git) |
| `SERPER_API_KEY` | optional but recommended — enables web-search evidence |
| `QDRANT_URL` | from `.env` |
| `QDRANT_API_KEY` | from `.env` |

**Build command:** `pip install -r requirements.txt`
**Start command:** `python web_demo.py`

Then **Deploy latest commit** (Render also redeploys on every push to `main`).

## Verifying a deploy

1. Open the URL (free instances sleep — first hit takes ~50 s; leave a tab open 2 min before presenting).
2. `GET /api/version` → JSON with `llm_module`.
3. **Saved demonstrations** tab → replays `medqa_347` / `medqa_107` instantly (no API calls, no index).
4. **New analysis** → paste a case → this runs the full pipeline: cloud Qdrant retrieval + Gemini + PubMed/NICE/Serper. Takes 30–150 s.
5. Check Render logs: no tracebacks; the run report panel shows sources utilized + confidence tier.

## Failure playbook

| Symptom | Cause | Fix |
|---|---|---|
| 503 on first visit | free instance asleep | wait, retry — cold start ~50 s |
| Deploy fails in `pip install torch` | CUDA wheels too heavy for the instance | switch build to CPU wheels (`--extra-index-url https://download.pytorch.org/whl/cpu`) |
| Service restarts on first analysis | 512 MB OOM (torch + BioBERT ≈ 1 GB) | move to a paid instance (≥1 GB), or set `OMP_NUM_THREADS=1` and retry |
| Empty results / collection errors | `QDRANT_URL`/key missing on Render | re-check the 4 env vars, redeploy |
| Live analysis error, replay works | Gemini key missing/invalid | check `GEMINI_API_KEY` |

## Demo insurance (do this before presentation day)

- Record a screen capture of a full local run (`python main.py --case medqa_347`).
- Keep localhost as fallback: `python web_demo.py` → http://127.0.0.1:8765
- Saved-demonstration tab works even if every external service is down.
