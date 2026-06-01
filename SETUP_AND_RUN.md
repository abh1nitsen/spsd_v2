# SPSD v4.3 — Setup and Run Guide
## Complete from-scratch replication in Google Colab (free tier)

---

## What This Proves

This pipeline tests 11 hypotheses across five dimensions to prove that **SPSD is worth deploying**:

| Dimension | Hypotheses | What It Proves |
|---|---|---|
| PRIMARY | H1, H2, H3 | SPSD saves meaningful tokens on every eligible call |
| QUALITY | H4, H5 | Response quality is preserved (cosine similarity + independent LLM judge) |
| SAFETY | H6, H7 | Medical and legal prompts are never distilled |
| EFFICIENCY | H8, H9 | SPSD is most effective on verbose social prompts (its target) |
| ECONOMY | H10, H11 | Total cost reduction is real and system prompt caching amplifies savings |

---

## Files in This Package

```
fetch_corpus_v3.py      Step 1 — Fetch 200+ real prompts from HuggingFace
run_spsd_v3.py          Step 2 — Run SPSD distillation on every prompt
run_llm_v3.py           Step 3 — Get paired LLM responses (raw + distilled)
score_and_judge_v3.py   Step 4 — Cosine similarity + LLM-as-judge scoring
hypothesis_v3.py        Step 5 — Run all 11 hypothesis tests + Excel report
spsd_v4.py              Core SPSD pipeline (required, upload to Colab)
ale_prompt.py           ALE packet builder (required, upload to Colab)
```

---

## Prerequisites

### Accounts (both free)
| Service | Purpose | Sign up at |
|---|---|---|
| Google Colab | Run the pipeline | colab.research.google.com |
| Groq | LLM API calls (free tier) | console.groq.com |

### Groq free tier limits
- 14,400 requests/day
- ~20,000 tokens/minute for llama-3.1-8b-instant (eval model)
- ~6,000 tokens/minute for llama-3.3-70b-versatile (judge model)
- The pipeline handles rate limiting automatically (90s backoff on 429)

---

## Step-by-Step Instructions

### 1. Open a new Google Colab notebook
Go to colab.research.google.com → New notebook

### 2. Add your Groq API key as a secret
Left sidebar → key icon (🔑) → Add new secret  
Name: `GROQ_API_KEY`  
Value: your key from console.groq.com

### 3. Upload all pipeline files to Colab
In the left sidebar → Files tab → Upload icon  
Upload these files to `/content/`:
```
spsd_v4.py
ale_prompt.py
fetch_corpus_v3.py
run_spsd_v3.py
run_llm_v3.py
score_and_judge_v3.py
hypothesis_v3.py
```

### 4. Run Cell 1 — Install dependencies
```python
!pip install -q llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
!pip install -q datasets groq sentence-transformers huggingface_hub \
    scipy openpyxl matplotlib requests
print("Dependencies installed.")
```

### 5. Run Cell 2 — Download the SLM model (~986 MB, one-time)
```python
import sys
sys.path.insert(0, '/content')
import spsd_v4
MODEL_PATH = spsd_v4.download_model(dest_dir='/content/models')
print(f"Model saved at: {MODEL_PATH}")

# Optional: save to Google Drive so you don't re-download next session
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd/models', exist_ok=True)
shutil.copy(MODEL_PATH, '/content/drive/MyDrive/spsd/models/')
print("Model saved to Drive.")
```

**If you already have the model on Drive**, load it instead:
```python
MODEL_PATH = '/content/drive/MyDrive/spsd/models/qwen2.5-1.5b-instruct-q4_k_m.gguf'
import spsd_v4
spsd_v4.load_model(model_path=MODEL_PATH)
print("Model loaded from Drive.")
```

### 6. Run Cell 3 — Reload modules (ALWAYS run after uploading new files)
```python
import sys, importlib
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key:
        del sys.modules[key]
import spsd_v4, ale_prompt
importlib.reload(spsd_v4)
importlib.reload(ale_prompt)
print(f"spsd_v4 loaded | SHORT_LIMIT={spsd_v4.SHORT_PROMPT_WORD_LIMIT}")

# Quick verification
test = ("A 23-year-old pregnant woman at 22 weeks gestation presents to her "
        "physician with burning upon urination that started one day ago")
pt, reason = spsd_v4.tier1_check(test)
assert pt and reason == "domain_medical", f"FAILED: {pt} {reason}"
print("Medical gate: OK")
print("Ready to run pipeline.")
```

---

## Run the Pipeline (Steps 1–5)

Each step is resume-safe. If Colab disconnects, just re-run the same step and it continues from where it left off.

### Step 1 — Fetch corpus (~5–10 minutes)
```python
%run /content/fetch_corpus_v3.py
```
**Output:** `/content/spsd_corpus_v3.csv`  
**What to expect:** ~205 prompts across 6 categories. All full text, no truncation.

### Step 2 — SPSD distillation (~30–60 minutes on CPU)
```python
%run /content/run_spsd_v3.py
```
**Output:** `/content/spsd_results_v3.csv`  
**What to expect:** ~120 distilled, ~85 passthrough. Medical prompts all passthrough.  
Checkpoints every 25 rows. Safe to interrupt and resume.

### Step 3 — LLM calls (~20–30 minutes)
```python
%run /content/run_llm_v3.py
```
**Output:** Updates `/content/spsd_results_v3.csv` in place  
**What to expect:** 2 Groq API calls per distilled row (raw + distilled).  
Done tracker: `/content/llm_v3_done.json`. Resume safe.

**Groq rate limit troubleshooting:**  
If you see repeated 429 errors, increase `CALL_INTERVAL` at the top of `run_llm_v3.py`:
```python
CALL_INTERVAL = 10.0  # increase from 6.0
```

### Step 4 — Quality scoring + LLM judge (~30–60 minutes)
```python
%run /content/score_and_judge_v3.py
```
**Output:** Updates `spsd_results_v3.csv` + creates `/content/judge_results_v3.json`  
**What to expect:**
- Cosine similarity scores for all pairs (fast, ~2 minutes)
- LLM judge calls at 8s per pair — for ~120 pairs, ~16 minutes  
- Judge uses `llama-3.3-70b-versatile` (different model from evaluation)

**Note:** Judge calls use Groq's 70B model which has lower TPM. If rate limited, the script handles it automatically with 90s backoff.

### Step 5 — Hypothesis tests + Excel report (~2 minutes)
```python
%run /content/hypothesis_v3.py
```
**Output:** `/content/spsd_hypothesis_v3.xlsx`  
**What to expect:** All 11 hypotheses tested. Excel has 3 sheets: Hypothesis Results, Full Data, Judge Detail.

---

## Save All Results to Drive
```python
from google.colab import drive
import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd/results_v3', exist_ok=True)

for fname in ['spsd_corpus_v3.csv', 'spsd_results_v3.csv',
              'spsd_hypothesis_v3.xlsx', 'judge_results_v3.json']:
    src = f'/content/{fname}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/drive/MyDrive/spsd/results_v3/{fname}')
        print(f"Saved: {fname}")
```

---

## Expected Results

Based on v4.2 corpus results (150 prompts), the new v3 corpus should show:

| Hypothesis | Expected Result |
|---|---|
| H1 Token saving > 0 | PROVEN — t > 15, p < 0.001, d > 1.5 |
| H2 Compression ratio < 1.0 | PROVEN — t < -30, p < 0.001 |
| H3 100% positive | PROVEN — all distilled calls net positive |
| H4 Quality sim > 0.70 | PROVEN — t > 4, p < 0.001 |
| H5 Judge dist >= raw | EXPECTED PROVEN — depends on corpus |
| H6 Medical 100% passthrough | PROVEN — by gate design |
| H7 Legal 100% passthrough | PROVEN — by gate design |
| H8 verbose > general savings | PROVEN — Mann-Whitney p < 0.01 |
| H9 Category variation | LIKELY NOT SIGNIFICANT — SPSD works broadly |
| H10 Total saving > 0 | PROVEN — input saving dominates |
| H11 Cache saving > 80% | PROVEN — ~90% saving on system prompt |

---

## Troubleshooting

### "Medical prompts getting distilled"
The module has stale state. Run the reload cell (Cell 3) again.

### "GROQ_API_KEY not found"
Make sure you added it via left sidebar → key icon, NOT by pasting it in code.

### "Groq 429 after every call"
Increase `CALL_INTERVAL = 10.0` in `run_llm_v3.py` and `JUDGE_INTERVAL = 12.0` in `score_and_judge_v3.py`.

### "run_spsd_v3.py stuck / Colab disconnected"
Re-run the step. It reads the existing output and skips already-processed rows.

### "run_llm_v3.py calling the wrong rows"
Delete the done tracker and re-run:
```python
import os
if os.path.exists('/content/llm_v3_done.json'):
    os.remove('/content/llm_v3_done.json')
```

### "Prompts look cut off"
This is a display issue — the actual `raw_prompt` column in the CSV is never truncated. Verify with:
```python
import csv
with open('/content/spsd_results_v3.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
lens = [len(r['raw_prompt']) for r in rows if r['passthrough']=='False']
print(f"Min: {min(lens)} chars | Max: {max(lens)} chars")
# Max should be well above 400 (old cap was 400 chars)
```

---

## File Sizes (approximate)

| File | Size | Notes |
|---|---|---|
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | ~986 MB | SLM model, download once |
| `spsd_corpus_v3.csv` | ~1–2 MB | 200+ prompts, full text |
| `spsd_results_v3.csv` | ~3–5 MB | SPSD + LLM + scoring results |
| `spsd_hypothesis_v3.xlsx` | ~1 MB | 3-sheet Excel report with charts |
| `judge_results_v3.json` | ~200 KB | Raw judge scores per pair |

---

## Complete Notebook Template

Paste this into a new Colab notebook for a clean single-session run:

```python
# Cell 1 — Install
!pip install -q llama-cpp-python --extra-index-url \
    https://abetlen.github.io/llama-cpp-python/whl/cpu
!pip install -q datasets groq sentence-transformers huggingface_hub \
    scipy openpyxl matplotlib requests

# Cell 2 — Upload files
# (Upload spsd_v4.py, ale_prompt.py, and all 5 pipeline scripts via Files tab)

# Cell 3 — Load model (first time) OR load from Drive (subsequent runs)
import sys; sys.path.insert(0, '/content')
import spsd_v4
# First time: MODEL_PATH = spsd_v4.download_model(dest_dir='/content/models')
# Subsequent: MODEL_PATH = '/content/drive/MyDrive/spsd/models/qwen2.5-1.5b-instruct-q4_k_m.gguf'
MODEL_PATH = spsd_v4.download_model(dest_dir='/content/models')
spsd_v4.load_model(model_path=MODEL_PATH)

# Cell 4 — Reload modules (run after every file upload)
import importlib
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key: del sys.modules[key]
import spsd_v4, ale_prompt
importlib.reload(spsd_v4); importlib.reload(ale_prompt)
pt, r = spsd_v4.tier1_check("A 23-year-old pregnant woman at 22 weeks gestation presents to her physician with symptoms")
assert pt and r == "domain_medical"
print("All checks passed. Ready.")

# Cell 5 — Step 1: Fetch corpus
%run /content/fetch_corpus_v3.py

# Cell 6 — Step 2: SPSD distillation
%run /content/run_spsd_v3.py

# Cell 7 — Step 3: LLM calls
%run /content/run_llm_v3.py

# Cell 8 — Step 4: Score + judge
%run /content/score_and_judge_v3.py

# Cell 9 — Step 5: Hypothesis tests
%run /content/hypothesis_v3.py

# Cell 10 — Save to Drive
from google.colab import drive; import shutil, os
drive.mount('/content/drive')
os.makedirs('/content/drive/MyDrive/spsd/results_v3', exist_ok=True)
for f in ['spsd_corpus_v3.csv','spsd_results_v3.csv',
          'spsd_hypothesis_v3.xlsx','judge_results_v3.json']:
    if os.path.exists(f'/content/{f}'):
        shutil.copy(f'/content/{f}',f'/content/drive/MyDrive/spsd/results_v3/{f}')
        print(f"Saved {f}")
```
