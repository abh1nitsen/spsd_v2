"""
SPSD v4.3 — Step 4: Quality Scoring + LLM-as-Judge
====================================================
Two independent quality signals:

Method A — Cosine similarity (all-MiniLM-L6-v2)
  Pair-wise embedding similarity between raw and distilled responses.
  Threshold: 0.70

Method B — LLM-as-judge (llama-3.3-70b-versatile)
  Different model from eval model — no contamination.
  Blind A/B: judge never knows which response is raw or distilled.
  Scores INFORMATION EQUIVALENCE (1-5).
  JSON extraction: re.search catches preamble text before JSON object.

Changes from previous version:
  - JSON extraction uses re.search(r'{.*}') not json.loads on full response
  - Drive backup after similarity scoring
  - Judge checkpoint to Drive every 10 pairs
  - Countdown before judge removed (no longer needed)
  - Cleaner exclusion reporting

Input:  /content/spsd_results_v3.csv (updated in place)
Output: /content/judge_results_v3.json
Run:    %run /content/score_and_judge_v3.py
"""

import csv, json, os, time, sys, random, re
import numpy as np
sys.path.insert(0, '/content')

# ── Drive mount ───────────────────────────────────────────────
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

INPUT_FILE  = "/content/spsd_results_v3.csv"
JUDGE_FILE  = "/content/judge_results_v3.json"
SIM_THRESH  = 0.70
DRIVE_DIR   = "/content/drive/MyDrive/spsd/v3_run"

# ── Load data ─────────────────────────────────────────────────
with open(INPUT_FILE, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for r in rows:
    pt = r.get('passthrough', '')
    if pt.upper() == 'FALSE': r['passthrough'] = 'False'
    elif pt.upper() == 'TRUE': r['passthrough'] = 'True'

by_id = {r['id']: r for r in rows}

def sf(v):
    try: return float(v)
    except: return None


# ── Cosine similarity ─────────────────────────────────────────
print("Loading sentence-transformers/all-MiniLM-L6-v2...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Loaded.")

paired = [r for r in rows
          if r['passthrough'] == 'False'
          and r.get('raw_response', '').strip()
          and not r.get('raw_response', '').startswith('ERROR')
          and r.get('dist_response', '').strip()
          and not r.get('dist_response', '').startswith('ERROR')]

to_score = [r for r in paired if not sf(r.get('semantic_similarity'))]
print(f"\nEligible pairs: {len(paired)} | To score now: {len(to_score)}")

if to_score:
    raw_texts  = [r['raw_response'][:600]  for r in to_score]
    dist_texts = [r['dist_response'][:600] for r in to_score]
    raw_embs   = st_model.encode(raw_texts,  convert_to_numpy=True,
                                  show_progress_bar=True, batch_size=64)
    dist_embs  = st_model.encode(dist_texts, convert_to_numpy=True,
                                  show_progress_bar=True, batch_size=64)
    for row, re_, de in zip(to_score, raw_embs, dist_embs):
        sim  = float(np.dot(re_, de) /
                     (np.linalg.norm(re_) * np.linalg.norm(de)))
        flag = ('OK'         if sim >= SIM_THRESH else
                'BORDERLINE' if sim >= 0.50       else 'LOW')
        by_id[row['id']]['semantic_similarity'] = f"{sim:.4f}"
        by_id[row['id']]['quality_flag']        = flag

all_sims = [sf(by_id[r['id']].get('semantic_similarity'))
            for r in paired
            if sf(by_id[r['id']].get('semantic_similarity'))]
n_ok  = sum(1 for s in all_sims if s >= SIM_THRESH)
n_low = sum(1 for s in all_sims if s  < 0.50)
print(f"\nCosine similarity — mean={np.mean(all_sims):.4f} "
      f">=0.70: {n_ok}/{len(all_sims)}")

# Save after similarity
def _write_csv():
    updated = list(by_id.values())
    ak = list(dict.fromkeys(k for r in updated for k in r.keys()))
    with open(INPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                           extrasaction='ignore', restval='')
        w.writeheader(); w.writerows(updated)
    # Drive backup
    os.makedirs(DRIVE_DIR, exist_ok=True)
    backup = os.path.join(DRIVE_DIR, 'spsd_results_afterJUDGE.csv')
    with open(backup, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                           extrasaction='ignore', restval='')
        w.writeheader(); w.writerows(updated)

_write_csv()
print("Similarity scores saved.")

# ── LLM-as-judge ─────────────────────────────────────────────
try:
    from google.colab import userdata
    GROQ_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_KEY:
    print("No GROQ_API_KEY — skipping judge. Add key to Colab Secrets.")
else:
    from groq import Groq
    JUDGE_MODEL    = "llama-3.3-70b-versatile"
    JUDGE_INTERVAL = 12.0
    RETRY_WAIT     = 60
    MAX_RETRIES    = 4
    JUDGE_TRUNC    = 800

    JUDGE_SYSTEM = (
        "You are evaluating whether two AI responses convey equivalent "
        "information to a user query.\n\n"
        "You will receive:\n"
        "  ORIGINAL QUERY: the user's question or request\n"
        "  RESPONSE A: one AI response\n"
        "  RESPONSE B: another AI response\n\n"
        "Your task is to assess information equivalence — not which response "
        "is better, but whether both responses would leave the user equally "
        "informed and able to act.\n\n"
        "CRITICAL CALIBRATION RULE:\n"
        "Format and approach differences do NOT lower the equivalence score. "
        "If Response A writes a letter template and Response B gives bullet "
        "point steps, but both would allow the user to take the same action, "
        "score equivalence as 4 or 5. Only lower the score when one response "
        "is missing information the other has that would materially change "
        "what the user does next.\n\n"
        "Score EACH response independently on:\n"
        "  intent_coverage (1-5): Does this response address what the user asked?\n"
        "    5=fully addresses all parts  1=misses the main point\n\n"
        "  factual_quality (1-5): Are the facts/recommendations correct and useful?\n"
        "    5=accurate, specific, actionable  1=vague, incorrect, unhelpful\n\n"
        "  information_completeness (1-5): Does this contain the key information?\n"
        "    5=everything the user needs  1=missing critical information\n\n"
        "Then assess EQUIVALENCE:\n"
        "  equivalence (1-5): How interchangeable are these two responses?\n"
        "    5=user receiving either would be equally informed and able to act\n"
        "    4=minor differences in detail or format, same core information\n"
        "    3=one response has noticeably MORE useful information — not just "
        "a different format but genuinely more actionable content\n"
        "    2=significant information gap — one response is missing something "
        "the user clearly needs\n"
        "    1=responses give substantially different information or one "
        "refuses to help while the other does\n\n"
        'Respond ONLY with valid JSON, no other text:\n'
        '{"response_a":{"intent_coverage":N,"factual_quality":N,'
        '"information_completeness":N,"total":N},'
        '"response_b":{"intent_coverage":N,"factual_quality":N,'
        '"information_completeness":N,"total":N},'
        '"equivalence":N,'
        '"what_differs":"one sentence on key difference, or: responses are equivalent"}'
    )

    # Pre-specified exclusions
    _TASK_REQ = re.compile(
        r'\b(provide (a |the )?code|'
         r'can you (code|write (a |the )?(code|function|script|program|tool|system|app))|'
         r'write (the |a )?(code|function|script|program|tool|system|app|algorithm|query|class)|'
         r'code (in|for|using) [a-z#+]+|'
         r'\bin (python|java|c#|c\+\+|javascript|typescript|sql|matlab|php|ruby|swift)\b|'
         r'using (python|java|c#|c\+\+|javascript|typescript|sql|matlab)|'
         r'error (in|for|with) my code|IndentationError|SyntaxError|'
         r'build (a |the )?(tool|system|app|application|database|model))\b', re.I)

    EXCL_CATS   = {'general_conversational', 'multi_intent_linked'}
    coding_excl = {r['id'] for r in paired
                   if r['category'] == 'verbose_social'
                   and _TASK_REQ.search(r.get('raw_prompt', ''))}
    low_excl    = {r['id'] for r in paired
                   if r.get('quality_flag', '') == 'LOW'}
    sim_excl    = {r['id'] for r in paired
                   if r['category'] in EXCL_CATS
                   and (sf(r.get('semantic_similarity')) or 1.0) < 0.40}
    excl_ids    = coding_excl | low_excl | sim_excl

    judge_results = {}
    if os.path.exists(JUDGE_FILE):
        with open(JUDGE_FILE) as f:
            judge_results = json.load(f)
        print(f"Judge: resuming ({len(judge_results)} already scored)")

    to_judge = [r for r in paired
                if r['id'] not in judge_results
                and r['id'] not in excl_ids]

    print(f"Judge scoring: {len(to_judge)} pairs "
          f"(excluded {len(excl_ids)}: "
          f"{len(coding_excl)} coding-misclassified, "
          f"{len(low_excl)} low-flag, "
          f"{len(sim_excl - coding_excl - low_excl)} low-sim)")

    random.seed(42)
    judge_client = Groq(api_key=GROQ_KEY)

    for i, row in enumerate(to_judge):
        pid    = row['id']
        orig   = row['raw_prompt']
        raw_r  = row['raw_response']
        dist_r = row['dist_response']

        if random.random() > 0.5:
            resp_a, resp_b, a_is = raw_r, dist_r, 'raw'
        else:
            resp_a, resp_b, a_is = dist_r, raw_r, 'dist'

        # Truncate to prevent empty judge responses on long inputs
        a_trunc = resp_a[:JUDGE_TRUNC] + ("..." if len(resp_a) > JUDGE_TRUNC else "")
        b_trunc = resp_b[:JUDGE_TRUNC] + ("..." if len(resp_b) > JUDGE_TRUNC else "")

        prompt = (f"ORIGINAL QUERY:\n{orig}\n\n---\n\n"
                  f"RESPONSE A:\n{a_trunc}\n\n---\n\nRESPONSE B:\n{b_trunc}")

        for attempt in range(MAX_RETRIES):
            try:
                resp = judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                               {"role": "user",   "content": prompt}],
                    max_tokens=200,
                    temperature=0.1)
                raw_json = resp.choices[0].message.content or ""
                clean_j  = re.sub(r'```[a-z]*', '', raw_json).strip()
                # Extract JSON object from anywhere in response
                m = re.search(r'\{.*\}', clean_j, re.DOTALL)
                if not m:
                    raise ValueError(f"No JSON in response: {raw_json[:80]}")
                parsed = json.loads(m.group())

                sa          = parsed['response_a']['total']
                sb          = parsed['response_b']['total']
                equivalence = int(parsed.get('equivalence', 3))
                what_differs= parsed.get('what_differs', '')

                raw_s  = sa if a_is == 'raw' else sb
                dist_s = sb if a_is == 'raw' else sa

                judge_results[pid] = {
                    'raw_score':   raw_s,
                    'dist_score':  dist_s,
                    'equivalence': equivalence,
                    'what_differs': what_differs,
                    'a_is':        a_is,
                }
                by_id[pid].update({
                    'judge_raw_score':    str(raw_s),
                    'judge_dist_score':   str(dist_s),
                    'judge_equivalence':  str(equivalence),
                    'judge_what_differs': what_differs[:200],
                })
                eq_sym = {5: '≡ equiv', 4: '≈ close', 3: '~ diff',
                          2: '≠ gap',   1: '✗ fail'}
                print(f"  [{i+1:3d}/{len(to_judge)}] {pid} "
                      f"raw={raw_s:2d} dist={dist_s:2d} "
                      f"equiv={equivalence} {eq_sym.get(equivalence, '')}")
                break

            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = RETRY_WAIT * (attempt + 1)
                    print(f"  rate limit — waiting {wait}s...")
                    time.sleep(wait)
                    judge_client = Groq(api_key=GROQ_KEY)
                elif attempt == MAX_RETRIES - 1:
                    print(f"  [{pid}] judge error: {e}")
                else:
                    time.sleep(5)

        time.sleep(JUDGE_INTERVAL)

        if (i + 1) % 10 == 0:
            with open(JUDGE_FILE, 'w') as f:
                json.dump(judge_results, f, indent=2)
            _write_csv()
            print(f"  ── checkpoint ({i+1} judged) ──")

    # Final save
    with open(JUDGE_FILE, 'w') as f:
        json.dump(judge_results, f, indent=2)
    _write_csv()

    # Summary
    eq_scores = [v['equivalence'] for v in judge_results.values()
                 if 'equivalence' in v]
    jrs  = [v['raw_score']  for v in judge_results.values()]
    jds  = [v['dist_score'] for v in judge_results.values()]
    from collections import Counter
    eq_dist = Counter(eq_scores)

    print(f"\nJudge complete: {len(judge_results)} pairs (model={JUDGE_MODEL})")
    if eq_scores:
        print(f"Mean equivalence: {np.mean(eq_scores):.2f}/5.0")
        for score in [5, 4, 3, 2, 1]:
            bar = '█' * eq_dist[score]
            print(f"  equiv={score}: {eq_dist[score]:3d} {bar}")
        print(f"Mean raw score:  {np.mean(jrs):.2f}/15")
        print(f"Mean dist score: {np.mean(jds):.2f}/15")

print(f"\nAll scores saved to {INPUT_FILE}")
print(f"Next: %run /content/hypothesis_v3.py")
