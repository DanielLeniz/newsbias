# News Bias

Detects article-level political bias (Left/Center/Right), summarizes the article, and (optionally) displays a **source prior** from AllSides for context.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
# python3 -m venv .venv -- use this if you have python3
# .venv/Scripts/activate -- use this instead if .venv/bin/activate does not work with your machine
pip install -r requirements.txt
cp .env.example .env  # fill in API keys
```

## Start server with hosted model
```bash
uvicorn app.api:app --reload --port 8000
```
## Health & model
```bash
curl -s http://127.0.0.1:8000/healthz | jq
curl -s http://127.0.0.1:8000/model | jq
```


## Classify this Fox News article
```bash
curl -s -X POST http://127.0.0.1:8000/predict_url \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.foxnews.com/live-news/anti-trump-no-kings-protests-october-18-2025"}' | jq
  ```
For Powershell Users run
```bash
$body = @{ url = "https://www.foxnews.com/live-news/anti-trump-no-kings-protests-october-18-2025" } |
         ConvertTo-Json

Invoke-RestMethod -Method POST `
  -Uri http://127.0.0.1:8000/predict_url `
  -ContentType 'application/json' `
  -Body $body | ConvertTo-Json -Depth 10
```


## What it does

- **Fetch** article text from a URL (`/fetch`, `/predict_url`)
- **Summarize** with GPT (OpenAI) when enabled — falls back to extractive summary of the first few sentences if not available
- **Source prior** (optional): Look up the outlet’s AllSides media-bias rating (display only; does **not** affect the bias rating)
- **Article bias**: Run `bucketresearch/politicalBiasBERT` to predict **Left / Center / Right**
  - Returns `label`, `confidence`, and full per-class `probs`
  - Adds important keywords for user transparency -- **does not affect the model**


## Endpoints

### Health & Info
- `GET /healthz` → `{"ok": true, "time": "...Z"}`
- `GET /env` / `GET /env_debug` device, torch, summarizer status/model
- `GET /model` minimal model card (name, labels, optional temperature/metrics)

### Extraction
- `GET /fetch?url=...`  
  Fetch & parse article `{url, source (domain), title, text}`

### Inference
- `POST /predict`  
  Body: `{"text": "...", "title": "Optional"}`  
  Classifies raw text `summary`, `bias`, `explain`

- `POST /batch_predict`  
  Body: `[{"text":"..."}, {"text":"..."}]`  
  Returns a list of `PredictResponse` items.

- `POST /predict_url`  
  Body: `{"url":"<article url>"}`  
  Pipeline: **fetch, summarize, classify, (source_prior if available)**


---

## Example usage

```bash
# Extract the text only
curl -s "http://127.0.0.1:8000/fetch?url=https://www.foxnews.com/us/harvard-faculty-expressed-support-potential-left-wing-political-violence-during-2018-panel" | jq

# URL with full pipeline
curl -s -X POST http://127.0.0.1:8000/predict_url   -H 'Content-Type: application/json'   -d '{"url":"https://www.foxnews.com/us/harvard-faculty-expressed-support-potential-left-wing-political-violence-during-2018-panel"}' | jq

# Raw text then classify
curl -s -X POST http://127.0.0.1:8000/predict   -H 'Content-Type: application/json'   -d '{"text":"Lawmakers reached a bipartisan deal after extended negotiations."}' | jq

```

**Response shape (example)**

```json
{
  "summary": "...",
  "bias": {
    "label": "Right",
    "confidence": 0.664,
    "probs": {
      "Left": 0.181,
      "Center": 0.155,
      "Right": 0.664
    }
  },
  "explain": {
    "spans": [
      {
        "start": 20,
        "end": 30,
         "text": "bipartisan"
      }
    ]
  },
  "source_prior": {
    "source": "Fox News",
    "domain": "foxnews.com",
    "rating": "Right",
    "origin": "AllSides"
    "notes": "Outlet generally right-leaning"
  }
}
```

---

## Configuration

Environment variables:

| `OPENAI_API_KEY` | Enables GPT summarization | `sk-...` |

| `SUMMARY_ENABLED` | `1` to use OpenAI, `0` to disable | `1` |

| `SUMMARY_MODEL` | OpenAI model for summaries | `gpt-4o-mini` |

| `BIAS_MODEL_NAME` | HF repo id or local path | `bucketresearch/politicalBiasBERT` |

| `ALLSIDES_PRIORS_PATH` | Path to AllSides priors CSV | `./data/allsides_priors.csv` |

| `DATA_DIR` | Data dir (defaults to `data`) | `data` |

**AllSides CSV**  
CSV should include at least an outlet **name** (e.g., `source_name`) and **rating** (e.g., `allsides_bias`). A domain column is optional; the loader also maps common domains to names (e.g., `cnn.com -> CNN`). If no match is found, `source_prior` is `null`.

---

## How it works (high level)

1. **Fetch**: `httpx` downloads the page HTML; extractor gets the main text using `trafilatura`
2. **Summarize**: If OpenAI is enabled, call the summarization model; otherwise, return an extractive summary based off the first few sentences of the article.
3. **Source prior**: Lookup outlet in the AllSides CSV by domain and/or name
4. **Classify**: Tokenize and run the bias model. Compute softmax:
   - `label` = Left/Center/Right
   - `confidence` 
   - `probs` = prob for each L/C/R
5. **Explain**: Spans to show political phrasing within article

---

## Testing


```bash
PYTHONPATH=. pytest -q
```

If you see `ModuleNotFoundError: app`, run:

```bash
BIAS_MODEL_NAME=bucketresearch/politicalBiasBERT PYTHONPATH=. pytest -q
```

---

## Troubleshooting

- **Summaries look like the first paragraph only**  
  Check `/env_debug`. If `summary_llm:false`, set `SUMMARY_ENABLED=1` and `OPENAI_API_KEY`

- **`source_prior` is null**  
  Ensure `ALLSIDES_PRIORS_PATH` points to your CSV and that the CSV has your outlet (by domain or name). Common domains are mapped to names (e.g., `foxnews.com → Fox News`). If still null, verify the CSV headers (e.g., `source_name`, `allsides_bias`)

---
