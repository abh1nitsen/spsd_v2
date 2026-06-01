"""
SPSD v4.3 — Step 3: LLM Calls (Raw + Distilled)
=================================================
Issues two paired calls per distilled row:
  RAW call    : original prompt in passthrough format (P\\n<prompt>)
  DISTILLED   : ALE compressed packet (D|tone|urgency|aux\\n<compressed>)

Both calls use identical model, system prompt, and temperature.
No max_tokens cap — the model generates until its natural endpoint.
SPSD targets INPUT compression only. Output length is the model's decision.

WHAT THE PROGRESS OUTPUT MEANS:
────────────────────────────────
  [  1/114] P001 [verbose_social]  save_in=+62t | raw_out=187t dist_out=143t
  save_in      = input token saving from SPSD (key metric for H1, H10)
  raw_out      = tokens the LLM generated for the RAW call (uncapped)
  dist_out     = tokens the LLM generated for the DISTILLED call (uncapped)
  Both output numbers are informational — H2 quality is measured by
  cosine similarity in score_and_judge_v3.py, not by output length.

Model: Groq llama-3.1-8b-instant (free tier)
No max_tokens: model stops naturally. Typical: 100-400t for support,
               200-600t for code. Groq TPM budget: ~4,500t/min used
               vs 20,000t/min available — well within free tier limits.

Input:  /content/spsd_results_v3.csv  (updated in place)
Done:   /content/llm_v3_done.json
Run:    %run /content/run_llm_v3.py
"""
import csv, json, os, time, sys

try:
    from google.colab import userdata
    GROQ_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "In Colab: left sidebar → key icon (🔑) → Secrets → add GROQ_API_KEY\n"
        "Free key: https://console.groq.com")

from groq import Groq
client = Groq(api_key=GROQ_KEY)

sys.path.insert(0, '/content')
import ale_prompt
SYSTEM_TEXT = ale_prompt.ALE_SYSTEM_PROMPT

GROQ_MODEL    = "llama-3.1-8b-instant"
TEMPERATURE   = 0.4
# No MAX_TOKENS — model generates until natural stop.
# SPSD compresses input only. Output length is the model's choice.
CALL_INTERVAL = 6.0   # seconds between calls (Groq TPM buffer)
RETRY_WAIT    = 90    # seconds on HTTP 429
MAX_RETRIES   = 4
INPUT_FILE    = "/content/spsd_results_v3.csv"
DONE_FILE     = "/content/llm_v3_done.json"

with open(INPUT_FILE, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
by_id = {r['id']: r for r in rows}

done_ids = set()
if os.path.exists(DONE_FILE):
    with open(DONE_FILE) as f:
        done_ids = set(json.load(f))
    print(f"Resuming: {len(done_ids)} pairs already done")

dist_rows = [r for r in rows if r['passthrough'] == 'False']
remaining  = [r for r in dist_rows if r['id'] not in done_ids]
n_total    = len(dist_rows)
print(f"Distilled rows: {n_total} | Remaining: {len(remaining)}")
print(f"Model: {GROQ_MODEL} | No output cap | interval: {CALL_INTERVAL}s\n")

def call_groq(user_turn, label=""):
    msgs = [
        {"role": "system", "content": SYSTEM_TEXT},
        {"role": "user",   "content": user_turn},
    ]
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=msgs,
                temperature=TEMPERATURE,
                # max_tokens intentionally omitted — uncapped output
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
            elif attempt == MAX_RETRIES - 1:
                return "", 0, 0, err
            else:
                time.sleep(5)
    return "", 0, 0, "max retries exceeded"

def _write(by_id, INPUT_FILE, DONE_FILE, done_ids):
    updated = list(by_id.values())
    ak = list(dict.fromkeys(k for r in updated for k in r.keys()))
    with open(INPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                           extrasaction='ignore', restval='')
        w.writeheader(); w.writerows(updated)
    with open(DONE_FILE, 'w') as f:
        json.dump(list(done_ids), f)

n_done = len(done_ids)
for row in remaining:
    pid      = row['id']
    category = row['category']
    n_done  += 1

    save_in = row.get('token_saving_input', '0') or '0'
    print(f"[{n_done:3d}/{n_total}] {pid} [{category[:22]:22s}]  "
          f"save_in={int(save_in):+d}t | ", end='', flush=True)

    # ── Raw call — original prompt, no compression ────────────────────────────
    raw_resp, raw_out, _, raw_err = call_groq(
        f"P\n{row['raw_prompt']}", f"{pid}-raw")
    time.sleep(CALL_INTERVAL)

    if raw_err:
        print(f"RAW_ERROR: {raw_err}")
        by_id[pid].update({'raw_response': f"ERROR:{raw_err}",
                           'llm_model': GROQ_MODEL})
        continue

    # ── Distilled call — ALE compressed packet ────────────────────────────────
    dist_resp, dist_out, _, dist_err = call_groq(
        row['ale_user_turn'], f"{pid}-dist")
    time.sleep(CALL_INTERVAL)

    print(f"raw_out={raw_out}t  dist_out={dist_out}t")
    if not dist_err:
        # Show first 80 chars of distilled response for real-time QA
        print(f"         {dist_resp[:80]!r}")

    if dist_err:
        print(f"  DIST_ERROR: {dist_err}")
        by_id[pid].update({'dist_response': f"ERROR:{dist_err}",
                           'llm_model': GROQ_MODEL})
        done_ids.add(pid)
        continue

    # ── Compute total token savings (input + output delta) ────────────────────
    raw_in     = int(row.get('raw_input_tokens',  0) or 0)
    dist_in    = int(row.get('dist_input_tokens', 0) or 0)
    raw_total  = raw_in  + raw_out
    dist_total = dist_in + dist_out
    total_save = raw_total - dist_total

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

    # Checkpoint every 10 pairs
    if n_done % 10 == 0:
        _write(by_id, INPUT_FILE, DONE_FILE, done_ids)
        print(f"  ── checkpoint ({n_done} done) ──")

_write(by_id, INPUT_FILE, DONE_FILE, done_ids)

print(f"\n{'='*60}")
print(f"LLM COMPLETE: {n_done} pairs processed")
print(f"Model: {GROQ_MODEL} | No output cap | temp={TEMPERATURE}")
print(f"Next: %run /content/score_and_judge_v3.py")
