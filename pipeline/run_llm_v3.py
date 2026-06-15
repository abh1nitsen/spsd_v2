"""
SPSD v4.3 — Step 3: LLM Calls (Raw + Distilled)
=================================================
Issues two paired calls per distilled row:
  RAW call    : original prompt with RAW_SYSTEM_PROMPT
  DISTILLED   : [annotation] compressed_prompt with ALE_SYSTEM_PROMPT

Both calls use identical model, temperature, and no max_tokens cap.
SPSD targets INPUT compression only. Output length is the model's choice.

Changes from previous version:
  - Uses updated ALE_SYSTEM_PROMPT (AI assistant framing, not support agent)
  - Uses ale_user_turn from spsd_results_v3.csv directly
  - JSON extraction fix: re.search(r'{.*}') catches preamble before JSON
  - Batch-aware: Drive checkpoint every 10 pairs + batch boundary saves
  - Self-healing: model auto-reloads if client session expires

Input:  /content/spsd_results_v3.csv (updated in place)
Done:   /content/llm_v3_done.json
Run:    %run /content/run_llm_v3.py
"""

import csv, json, os, time, sys, re
sys.path.insert(0, '/content')

# ── Drive mount ───────────────────────────────────────────────
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

# ── Groq key ──────────────────────────────────────────────────
try:
    from google.colab import userdata
    GROQ_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "In Colab: left sidebar → key icon → Secrets → add GROQ_API_KEY\n"
        "Free key: https://console.groq.com")

import ale_prompt

# ── Config ────────────────────────────────────────────────────
GROQ_MODEL     = "llama-3.1-8b-instant"
TEMPERATURE    = 0.4
CALL_INTERVAL  = 6.0    # seconds between calls
RETRY_WAIT     = 90     # seconds on rate limit
MAX_RETRIES    = 4
CHECKPOINT_N   = 10     # checkpoint every N pairs
INPUT_FILE     = "/content/spsd_results_v3.csv"
DONE_FILE      = "/content/llm_v3_done.json"
DRIVE_BACKUP   = "/content/drive/MyDrive/spsd/v3_run/spsd_results_afterLLM.csv"

# ── Load data ─────────────────────────────────────────────────
with open(INPUT_FILE, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
by_id = {r['id']: r for r in rows}

done_ids = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        done_ids = set(json.load(f))
    print(f"Resuming: {len(done_ids)} pairs already done")

dist_rows     = [r for r in rows if r['passthrough'] == 'False']
remaining     = [r for r in dist_rows if r['id'] not in done_ids]
n_total       = len(dist_rows)
print(f"Distilled rows: {n_total} | Remaining: {len(remaining)}")
print(f"Model: {GROQ_MODEL} | No output cap | interval: {CALL_INTERVAL}s\n")


def make_client():
    from groq import Groq
    return Groq(api_key=GROQ_KEY)


def call_groq(client, user_turn: str, system: str, label: str = ""):
    msgs = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_turn},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            resp  = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=msgs,
                temperature=TEMPERATURE,
                # max_tokens intentionally omitted — uncapped
            )
            text  = resp.choices[0].message.content or ""
            out_t = resp.usage.completion_tokens or 0
            in_t  = resp.usage.prompt_tokens or 0
            return text, out_t, in_t, None
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = RETRY_WAIT * (attempt + 1)
                print(f"\n  [{label}] rate limit — waiting {wait}s...")
                time.sleep(wait)
                client = make_client()  # fresh client after rate limit
            elif attempt == MAX_RETRIES - 1:
                return "", 0, 0, err
            else:
                time.sleep(10)
    return "", 0, 0, "max retries exceeded"


def _write_checkpoint():
    updated = list(by_id.values())
    ak = list(dict.fromkeys(k for r in updated for k in r.keys()))
    with open(INPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                           extrasaction='ignore', restval='')
        w.writeheader(); w.writerows(updated)
    with open(DONE_FILE, 'w') as f:
        json.dump(list(done_ids), f)
    # Drive backup
    os.makedirs(os.path.dirname(DRIVE_BACKUP), exist_ok=True)
    with open(DRIVE_BACKUP, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                           extrasaction='ignore', restval='')
        w.writeheader(); w.writerows(updated)


client = make_client()
n_done = len(done_ids)

for row in remaining:
    pid      = row['id']
    category = row['category']
    n_done  += 1

    save_in = int(row.get('token_saving_input', '0') or '0')
    print(f"[{n_done:3d}/{n_total}] {pid} [{category[:22]:22s}]  "
          f"save_in={save_in:+d}t | ", end='', flush=True)

    # Raw call — original prompt, RAW system prompt
    raw_resp, raw_out, _, raw_err = call_groq(
        client, row['raw_prompt'], ale_prompt.RAW_SYSTEM_PROMPT, f"{pid}-raw")
    time.sleep(CALL_INTERVAL)

    if raw_err:
        print(f"RAW_ERROR: {raw_err}")
        by_id[pid].update({'raw_response': f"ERROR:{raw_err}",
                           'llm_model': GROQ_MODEL})
        done_ids.add(pid)
        continue

    # Distilled call — ALE user turn, ALE system prompt
    dist_resp, dist_out, _, dist_err = call_groq(
        client, row['ale_user_turn'], ale_prompt.ALE_SYSTEM_PROMPT, f"{pid}-dist")
    time.sleep(CALL_INTERVAL)

    print(f"raw_out={raw_out}t  dist_out={dist_out}t")
    if not dist_err:
        print(f"         {repr(dist_resp[:80])}")

    if dist_err:
        print(f"  DIST_ERROR: {dist_err}")
        by_id[pid].update({'dist_response': f"ERROR:{dist_err}",
                           'llm_model': GROQ_MODEL})
        done_ids.add(pid)
        continue

    raw_in    = int(row.get('raw_input_tokens',  0) or 0)
    dist_in   = int(row.get('dist_input_tokens', 0) or 0)
    raw_total = raw_in  + raw_out
    dist_total= dist_in + dist_out
    total_save= raw_total - dist_total

    by_id[pid].update({
        'raw_response':       raw_resp,
        'raw_output_tokens':  str(raw_out),
        'dist_response':      dist_resp,
        'dist_output_tokens': str(dist_out),
        'raw_total_tokens':   str(raw_total),
        'dist_total_tokens':  str(dist_total),
        'total_token_saving': str(total_save),
        'llm_model':          GROQ_MODEL,
    })
    done_ids.add(pid)

    if n_done % CHECKPOINT_N == 0:
        _write_checkpoint()
        print(f"  ── checkpoint ({n_done} done) ──")

_write_checkpoint()

print(f"\n{'='*60}")
print(f"LLM COMPLETE: {n_done} pairs processed")
print(f"Model: {GROQ_MODEL} | No output cap | temp={TEMPERATURE}")
print(f"Next: %run /content/score_and_judge_v3.py")
