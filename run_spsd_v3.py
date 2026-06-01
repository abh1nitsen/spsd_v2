"""
SPSD v4.3 — Step 2: SPSD Distillation
=======================================
Runs every prompt through the full SPSD pipeline.
Checkpoint every 25 rows. Resume-safe.

Input:  /content/spsd_corpus_v3.csv
Output: /content/spsd_results_v3.csv
Run:    %run run_spsd_v3.py
"""
import sys, csv, json, os, importlib

sys.path.insert(0, '/content')
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key: del sys.modules[key]
import spsd_v4, ale_prompt
importlib.reload(spsd_v4); importlib.reload(ale_prompt)
print(f"spsd_v4 loaded | SHORT_LIMIT={spsd_v4.SHORT_PROMPT_WORD_LIMIT} | "
      f"MIN_SAVING={spsd_v4.MIN_NET_TOKEN_SAVING}")

# ── Mandatory pre-run verification ───────────────────────────────────────────
MEDICAL_TEST = (
    "A 23-year-old pregnant woman at 22 weeks gestation presents to her physician "
    "with burning upon urination that started one day ago and she is concerned "
    "about treatment safety during pregnancy and potential effects on the fetus"
)
pt, reason = spsd_v4.tier1_check(MEDICAL_TEST)
assert pt and reason == "domain_medical", \
    f"FATAL: medical passthrough broken — pt={pt} reason={reason}\n" \
    f"Re-upload spsd_v4.py and reload modules before running."
print("Medical passthrough gate: OK")

LEGAL_TEST = (
    "My landlord is attempting to evict me without proper notice and I need to "
    "understand what legal rights I have as a tenant and whether I should contact "
    "an attorney or file a complaint with the court directly"
)
pt2, reason2 = spsd_v4.tier1_check(LEGAL_TEST)
assert pt2 and reason2 == "domain_legal", \
    f"FATAL: legal passthrough broken — pt={pt2} reason={reason2}"
print("Legal passthrough gate:   OK")

SUPPORT_TEST = (
    "Hi I'm so sorry to bother you I know you must be incredibly busy but I "
    "placed an order last Tuesday for my daughter's birthday present ORD-847291 "
    "and her birthday is this Saturday I haven't received any shipping notification "
    "and I'm getting a bit worried could you please check on this for me"
)
pt3, _ = spsd_v4.tier1_check(SUPPORT_TEST)
assert not pt3, "FATAL: support prompt being passthroughed at Tier 1 incorrectly"
print("Support pass-to-SLM:      OK")
print()

# ── Load corpus ───────────────────────────────────────────────────────────────
CORPUS  = "/content/spsd_corpus_v3.csv"
OUTPUT  = "/content/spsd_results_v3.csv"
MODEL_PATH = None  # uses default from spsd_v4

with open(CORPUS, newline='', encoding='utf-8') as f:
    corpus = list(csv.DictReader(f))
print(f"Corpus: {len(corpus)} prompts")

# Resume from existing output — ONLY load rows whose IDs exist in current corpus.
# This prevents stale rows from a previous corpus contaminating the output.
current_ids = {row['id'] for row in corpus}
existing = {}
stale_count = 0
if os.path.exists(OUTPUT):
    with open(OUTPUT, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['id'] in current_ids:
                existing[row['id']] = row
            else:
                stale_count += 1
    if stale_count:
        print(f"WARNING: {stale_count} stale rows from a previous corpus — discarded")
    if existing:
        print(f"Resuming: {len(existing)}/{len(corpus)} already processed")
    else:
        print("No valid existing rows — running from scratch")

def token_est(text):
    return max(1, round(len(text.split()) / 0.75))

# Output columns (LLM + scoring columns pre-declared empty)
COLS = [
    'id','category','word_count','source','intent_label',
    'passthrough','passthrough_reason','distill_tier','distill_latency_ms',
    'domain','complexity_profile','social_score','semantic_score',
    'structural_score','repetition_score','rec_ratio','confidence',
    'hfg_aux','compressed_prompt','ale_user_turn',
    'raw_prompt',              # FULL prompt, never truncated
    'raw_input_tokens','dist_input_tokens',
    'token_saving_input','token_saving_pct','compression_ratio',
    # LLM fields (Step 3)
    'raw_response','dist_response','raw_output_tokens','dist_output_tokens',
    'raw_total_tokens','dist_total_tokens','total_token_saving','llm_model',
    # Quality fields (Step 4)
    'semantic_similarity','quality_flag',
    'judge_raw_score','judge_dist_score','judge_winner','judge_reasoning',
]

results = []
for i, row in enumerate(corpus):
    pid = row['id']

    if pid in existing:
        results.append(existing[pid])
        continue

    prompt   = row['prompt']   # FULL text from CSV, no truncation
    category = row['category']
    print(f"[{i+1:3d}/{len(corpus)}] {pid} [{category[:22]:22s}] ", end='', flush=True)

    result = spsd_v4.distill(prompt, model_path=MODEL_PATH)

    if result.passthrough:
        raw_tok  = token_est(prompt) + 2
        dist_tok = token_est(prompt) + 2
        saving   = 0; pct = 0.0; ratio = 1.0
        ale_turn = f"P\n{prompt}"
        print(f"PASSTHRU  reason={result.passthrough_reason}")
    else:
        raw_tok  = token_est(prompt) + 2
        ale_str  = ale_prompt.build_ale_messages(result)["messages"][0]["content"]
        dist_tok = token_est(ale_str)
        saving   = raw_tok - dist_tok
        pct      = saving / raw_tok * 100 if raw_tok > 0 else 0.0
        ratio    = dist_tok / raw_tok
        ale_turn = ale_str
        print(f"DISTILL   save={saving:+d}t ({pct:.0f}%) conf={result.confidence:.2f} "
              f"profile={result.complexity.profile if result.complexity else '?'}")

    out = {col:'' for col in COLS}
    out.update({
        'id':               pid,
        'category':         category,
        'word_count':       row['word_count'],
        'source':           row.get('source',''),
        'intent_label':     row.get('intent_label',''),
        'passthrough':      str(result.passthrough),
        'passthrough_reason': result.passthrough_reason or '',
        'distill_tier':     result.tier or '',
        'distill_latency_ms': f"{result.latency_ms:.1f}",
        'domain':           result.domain or '',
        'complexity_profile': (result.complexity.profile if result.complexity else ''),
        'social_score':     (f"{result.complexity.social_score:.3f}" if result.complexity else ''),
        'semantic_score':   (f"{result.complexity.semantic_score:.3f}" if result.complexity else ''),
        'structural_score': (f"{result.complexity.structural_score:.3f}" if result.complexity else ''),
        'repetition_score': (f"{result.complexity.repetition_score:.3f}" if result.complexity else ''),
        'rec_ratio':        (f"{result.complexity.recommended_ratio:.3f}" if result.complexity else ''),
        'confidence':       f"{result.confidence:.3f}",
        'hfg_aux':          '; '.join(result.hfg_aux) if result.hfg_aux else '',
        'compressed_prompt': result.compressed_prompt or '',
        'ale_user_turn':    ale_turn,
        'raw_prompt':       prompt,   # full text
        'raw_input_tokens': str(raw_tok),
        'dist_input_tokens': str(dist_tok),
        'token_saving_input': str(saving),
        'token_saving_pct':   f"{pct:.1f}",
        'compression_ratio':  f"{ratio:.3f}",
    })
    results.append(out)

    if (i + 1) % 25 == 0:
        ak = list(dict.fromkeys(k for r in results for k in r.keys()))
        with open(OUTPUT,'w',newline='',encoding='utf-8') as f:
            w = csv.DictWriter(f,fieldnames=ak,quoting=csv.QUOTE_ALL,
                               extrasaction='ignore',restval='')
            w.writeheader(); w.writerows(results)
        print(f"  ── checkpoint ({len(results)} rows) ──")

ak = list(dict.fromkeys(k for r in results for k in r.keys()))
with open(OUTPUT,'w',newline='',encoding='utf-8') as f:
    w = csv.DictWriter(f,fieldnames=ak,quoting=csv.QUOTE_ALL,
                       extrasaction='ignore',restval='')
    w.writeheader(); w.writerows(results)

from collections import Counter
dist = [r for r in results if r['passthrough']=='False']
pt   = [r for r in results if r['passthrough']=='True']
print(f"\n{'='*60}")
print(f"SPSD DONE: {len(results)} total | {len(dist)} distilled | {len(pt)} passthrough")
for cat in ['verbose_social','multi_intent_linked','general_conversational',
            'code_technical','high_stakes_medical','short_passthrough']:
    sub = [r for r in results if r['category']==cat]
    nd  = sum(1 for r in sub if r['passthrough']=='False')
    if sub: print(f"  {cat:28s} {nd:3d}/{len(sub):3d} distilled ({nd/len(sub)*100:.0f}%)")
print(f"\nNext: %run run_llm_v3.py")
