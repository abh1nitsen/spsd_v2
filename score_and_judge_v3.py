"""
SPSD v4.3 — Step 4: Quality Scoring + LLM-as-Judge
====================================================
Two independent quality signals for H4 and H5:

Method A — Cosine similarity (all-MiniLM-L6-v2, 22MB)
  Pair-wise embedding similarity between raw and distilled responses.
  Threshold: 0.70 (established from pilot testing).

Method B — LLM-as-judge (llama-3.3-70b-versatile on Groq)
  DIFFERENT model from evaluation model (no contamination).
  Blind A/B: judge does not know which response is raw or distilled.
  Scores each on: accuracy (1-5), completeness (1-5), tone (1-5), quality (1-5)
  Gives a winner verdict and one-sentence reasoning.
  Rate: 8s between judge calls.

Input:  /content/spsd_results_v3.csv  (updated in place)
Output: /content/judge_results_v3.json
Run:    %run score_and_judge_v3.py
"""
import csv, json, os, time, sys, random, re
import numpy as np

# ── Cosine similarity ─────────────────────────────────────────────────────────
print("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Loaded.")

INPUT_FILE  = "/content/spsd_results_v3.csv"
JUDGE_FILE  = "/content/judge_results_v3.json"
SIM_THRESH  = 0.70

with open(INPUT_FILE, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
by_id = {r['id']: r for r in rows}

def sf(v):
    try: return float(v)
    except: return None

# Eligible: distilled, both responses present, no error
paired = [r for r in rows
          if r['passthrough']=='False'
          and r.get('raw_response','').strip()
          and r.get('dist_response','').strip()
          and not r.get('raw_response','').startswith('ERROR')
          and not r.get('dist_response','').startswith('ERROR')]

print(f"\nEligible pairs for quality scoring: {len(paired)}")

# ── Cosine similarity scoring ─────────────────────────────────────────────────
if paired:
    raw_texts  = [r['raw_response'][:600]  for r in paired]
    dist_texts = [r['dist_response'][:600] for r in paired]
    raw_embs   = st_model.encode(raw_texts,  convert_to_numpy=True,
                                  show_progress_bar=True, batch_size=64)
    dist_embs  = st_model.encode(dist_texts, convert_to_numpy=True,
                                  show_progress_bar=True, batch_size=64)
    for row, re_, de in zip(paired, raw_embs, dist_embs):
        sim  = float(np.dot(re_,de) / (np.linalg.norm(re_)*np.linalg.norm(de)))
        flag = 'OK' if sim >= SIM_THRESH else ('BORDERLINE' if sim >= 0.50 else 'LOW')
        by_id[row['id']]['semantic_similarity'] = f"{sim:.4f}"
        by_id[row['id']]['quality_flag']        = flag
    sims = np.array([float(by_id[r['id']]['semantic_similarity']) for r in paired])
    print(f"Cosine similarity — mean={np.mean(sims):.4f} "
          f">=0.70: {sum(sims>=SIM_THRESH)}/{len(sims)}")

# Save after similarity scoring
updated = list(by_id.values())
ak = list(dict.fromkeys(k for r in updated for k in r.keys()))
with open(INPUT_FILE,'w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f,fieldnames=ak,quoting=csv.QUOTE_ALL,
                       extrasaction='ignore',restval='')
    w.writeheader(); w.writerows(updated)
print("Similarity scores saved.")

# ── LLM-as-judge ─────────────────────────────────────────────────────────────
try:
    from google.colab import userdata
    GROQ_KEY = userdata.get("GROQ_API_KEY")
except Exception:
    GROQ_KEY = os.environ.get("GROQ_API_KEY","")

if not GROQ_KEY:
    print("\nNo GROQ_API_KEY — skipping LLM judge. Re-run after adding key.")
else:
    from groq import Groq
    judge_client = Groq(api_key=GROQ_KEY)
    JUDGE_MODEL    = "llama-3.3-70b-versatile"   # distinct from eval model
    JUDGE_INTERVAL = 8.0
    RETRY_WAIT     = 90

    JUDGE_SYSTEM = (
        "You are a precise evaluator assessing the quality of AI assistant responses.\n\n"
        "You receive:\n"
        "  ORIGINAL QUERY: the user's full original prompt\n"
        "  RESPONSE A and RESPONSE B: two AI responses (order is random)\n\n"
        "Score EACH response 1-5 on:\n"
        "  accuracy:     Does it correctly address what the user asked?\n"
        "  completeness: Does it answer every part of the query?\n"
        "  tone:         Is the tone appropriate (helpful, not robotic or sycophantic)?\n"
        "  quality:      Overall quality — clarity, specificity, usefulness?\n\n"
        "Respond ONLY with valid JSON, no other text:\n"
        '{"response_a":{"accuracy":N,"completeness":N,"tone":N,"quality":N,"total":N},'
        '"response_b":{"accuracy":N,"completeness":N,"tone":N,"quality":N,"total":N},'
        '"verdict":"A"|"B"|"TIE","reasoning":"one sentence"}'
    )

    judge_results = {}
    if os.path.exists(JUDGE_FILE):
        with open(JUDGE_FILE) as f: judge_results = json.load(f)
        print(f"\nJudge: resuming ({len(judge_results)} already scored)")

    # Pre-specified exclusion criterion (same as hypothesis tests):
    # sim < 0.40 in general_conversational or multi_intent_linked
    EXCL_CATS = {'general_conversational','multi_intent_linked'}
    excl_ids  = {r['id'] for r in paired
                 if r['category'] in EXCL_CATS
                 and sf(by_id[r['id']].get('semantic_similarity')) is not None
                 and (sf(by_id[r['id']].get('semantic_similarity')) or 1) < 0.40}

    to_judge = [r for r in paired
                if r['id'] not in judge_results and r['id'] not in excl_ids]
    print(f"Judge scoring: {len(to_judge)} pairs (excluded {len(excl_ids)} low-sim misclassified)")

    random.seed(42)
    for i, row in enumerate(to_judge):
        pid     = row['id']
        orig    = row['raw_prompt']
        raw_r   = row['raw_response']
        dist_r  = row['dist_response']

        # Random A/B — record assignment
        if random.random() > 0.5:
            resp_a, resp_b, a_is = raw_r, dist_r, 'raw'
        else:
            resp_a, resp_b, a_is = dist_r, raw_r, 'dist'

        prompt = (f"ORIGINAL QUERY:\n{orig}\n\n---\n\n"
                  f"RESPONSE A:\n{resp_a}\n\n---\n\n"
                  f"RESPONSE B:\n{resp_b}")

        for attempt in range(4):
            try:
                resp = judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role":"system","content":JUDGE_SYSTEM},
                               {"role":"user","content":prompt}],
                    max_tokens=280, temperature=0.1)
                raw_json = resp.choices[0].message.content or ""
                clean_j  = re.sub(r'^```[a-z]*\n?','',raw_json.strip())
                clean_j  = re.sub(r'\n?```$','',clean_j)
                parsed   = json.loads(clean_j)
                sa = parsed['response_a']['total']
                sb = parsed['response_b']['total']
                verdict  = parsed.get('verdict','TIE')
                reasoning= parsed.get('reasoning','')
                # Map back
                if a_is == 'raw':
                    raw_s=sa; dist_s=sb
                    jwin = 'raw' if verdict=='A' else ('dist' if verdict=='B' else 'tie')
                else:
                    raw_s=sb; dist_s=sa
                    jwin = 'dist' if verdict=='A' else ('raw' if verdict=='B' else 'tie')

                judge_results[pid] = {
                    'raw_score':raw_s,'dist_score':dist_s,
                    'winner':jwin,'reasoning':reasoning,'a_is':a_is,
                    'raw_detail': parsed['response_a'] if a_is=='raw' else parsed['response_b'],
                    'dist_detail':parsed['response_b'] if a_is=='raw' else parsed['response_a'],
                }
                by_id[pid].update({
                    'judge_raw_score':str(raw_s),'judge_dist_score':str(dist_s),
                    'judge_winner':jwin,'judge_reasoning':reasoning[:200],
                })
                sym = {'raw':'← raw','dist':'dist →','tie':'= TIE'}
                print(f"  [{i+1:3d}/{len(to_judge)}] {pid} "
                      f"raw={raw_s:2d} dist={dist_s:2d} {sym.get(jwin,'')}")
                break
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    wait = RETRY_WAIT*(attempt+1)
                    print(f"  rate limit — waiting {wait}s..."); time.sleep(wait)
                elif attempt == 3:
                    print(f"  [{pid}] judge error: {e}")
                else:
                    time.sleep(5)

        time.sleep(JUDGE_INTERVAL)

        if (i+1) % 10 == 0:
            with open(JUDGE_FILE,'w') as f: json.dump(judge_results,f,indent=2)
            updated = list(by_id.values())
            ak = list(dict.fromkeys(k for r in updated for k in r.keys()))
            with open(INPUT_FILE,'w',newline='',encoding='utf-8') as f:
                w = csv.DictWriter(f,fieldnames=ak,quoting=csv.QUOTE_ALL,
                                   extrasaction='ignore',restval='')
                w.writeheader(); w.writerows(updated)
            print(f"  ── checkpoint ({i+1} judged) ──")

    with open(JUDGE_FILE,'w') as f: json.dump(judge_results,f,indent=2)
    from collections import Counter
    wins = Counter(v['winner'] for v in judge_results.values())
    jrs  = [v['raw_score']  for v in judge_results.values()]
    jds  = [v['dist_score'] for v in judge_results.values()]
    print(f"\nJudge summary ({len(judge_results)} pairs, model={JUDGE_MODEL}):")
    print(f"  dist wins={wins['dist']}  raw wins={wins['raw']}  ties={wins['tie']}")
    print(f"  mean raw score  = {np.mean(jrs):.2f}")
    print(f"  mean dist score = {np.mean(jds):.2f}")

# Final CSV save
updated = list(by_id.values())
ak = list(dict.fromkeys(k for r in updated for k in r.keys()))
with open(INPUT_FILE,'w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f,fieldnames=ak,quoting=csv.QUOTE_ALL,
                       extrasaction='ignore',restval='')
    w.writeheader(); w.writerows(updated)
print(f"\nAll scores saved to {INPUT_FILE}")
print(f"Next: %run hypothesis_v3.py")
