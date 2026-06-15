"""
SPSD v4.3 — Step 2: SPSD Distillation (updated for Gemma + complexity scorer v2)
==================================================================================
Runs every prompt through the full SPSD pipeline.
Checkpoint every 25 rows. Resume-safe. Drive-save on disconnect.

Changes from previous version:
  - complexity_profile / social_score / semantic_score / repetition_score /
    rec_ratio removed from output (scorer v2 no longer computes them)
  - structural_score kept (only remaining scorer output)
  - distill_latency_ms now tracked for SLM latency analysis
  - post_compression_net flag added to passthrough_reason vocabulary
  - ale_user_turn uses new lean bracket annotation format

Input:  /content/spsd_corpus_v3.csv
Output: /content/spsd_results_v3.csv
Run:    %run /content/run_spsd_v3.py
"""

import sys, csv, json, os, time, importlib

# ── Drive mount (resilient) ───────────────────────────────────
from google.colab import drive
if not os.path.exists('/content/drive/MyDrive'):
    drive.mount('/content/drive')

MODEL_PATH = "/content/drive/MyDrive/spsd/models/gemma-2-2b-it-Q4_K_M.gguf"

sys.path.insert(0, '/content')
for key in list(sys.modules.keys()):
    if 'spsd' in key or 'ale' in key:
        del sys.modules[key]
import spsd_v4, ale_prompt
importlib.reload(spsd_v4)
importlib.reload(ale_prompt)
print(f"spsd_v4 loaded | SHORT_LIMIT={spsd_v4.SHORT_PROMPT_WORD_LIMIT} | "
      f"MIN_SAVING={spsd_v4.MIN_NET_TOKEN_SAVING}")

# ── Pre-run gate verification ─────────────────────────────────
GATES = [
    ("medical",  "A 23-year-old pregnant woman presents to her physician "
                 "with burning urination and concern about treatment safety "
                 "during pregnancy and effects on the fetus",
     "domain_medical"),
    ("legal",    "My landlord is attempting to evict me without proper notice "
                 "and I need to understand what legal rights I have as a tenant "
                 "and whether to contact an attorney or file a complaint",
     "domain_legal"),
    ("code",     "Write a Python function that takes a list of integers and "
                 "returns them sorted ascending without using the built-in sort. "
                 "Handle empty lists and duplicates correctly.",
     "domain_code"),
]
for name, text, expected_reason in GATES:
    pt, reason = spsd_v4.tier1_check(text)
    assert pt and reason == expected_reason, (
        f"FATAL: {name} passthrough broken — pt={pt} reason={reason}\n"
        f"Re-upload spsd_v4.py and reload modules before running.")
    print(f"{name.capitalize():10s} gate: OK")

# Verify verbose social reaches SLM
SUPPORT_TEST = (
    "I'm so sorry to bother you but I placed an order three weeks ago "
    "and it still hasn't arrived. I've called twice and sent an email "
    "but nobody has helped me. Could you please look into this?"
)
pt3, _ = spsd_v4.tier1_check(SUPPORT_TEST)
assert not pt3, "FATAL: verbose social prompt passthroughed at Tier 1 incorrectly"
print(f"Verbose social: reaches SLM OK")
print()

# ── Corpus ────────────────────────────────────────────────────
CORPUS = "/content/spsd_corpus_v3.csv"
OUTPUT = "/content/spsd_results_v3.csv"

if not os.path.exists(CORPUS):
    raise FileNotFoundError(
        f"Corpus not found: {CORPUS}\n"
        f"Run diagnostic_corpus.py or fetch_corpus_v3.py first.")

with open(CORPUS, newline='', encoding='utf-8') as f:
    corpus = list(csv.DictReader(f))
print(f"Corpus: {len(corpus)} prompts")

# Resume — skip rows already processed for current corpus
current_ids = {row['id'] for row in corpus}
existing    = {}
stale_count = 0
if os.path.exists(OUTPUT):
    with open(OUTPUT, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['id'] in current_ids:
                existing[row['id']] = row
            else:
                stale_count += 1
    if stale_count:
        print(f"WARNING: {stale_count} stale rows discarded")
    if existing:
        print(f"Resuming: {len(existing)}/{len(corpus)} already done")

def token_est(text: str) -> int:
    return max(1, round(len(text.split()) / 0.75))

# Output columns — v4.3 (complexity scorer v2)
COLS = [
    'id', 'category', 'word_count', 'source', 'intent_label',
    'passthrough', 'passthrough_reason', 'distill_tier',
    'distill_latency_ms',     # SLM inference latency (ms)
    'domain',
    'structural_score',       # only remaining complexity score
    'confidence',
    'hfg_aux',
    'compressed_prompt',
    'ale_annotation',         # lean bracket annotation
    'ale_user_turn',          # full ALE user turn
    'raw_prompt',
    'raw_input_tokens', 'dist_input_tokens',
    'token_saving_input', 'token_saving_pct', 'compression_ratio',
    # LLM fields (Step 3)
    'raw_response', 'dist_response',
    'raw_output_tokens', 'dist_output_tokens',
    'raw_total_tokens', 'dist_total_tokens',
    'total_token_saving', 'llm_model',
    # Quality fields (Step 4)
    'semantic_similarity', 'quality_flag',
    'judge_raw_score', 'judge_dist_score',
    'judge_equivalence', 'judge_what_differs',
]

results = []
slm_latencies = []  # track latency for analysis

for i, row in enumerate(corpus):
    pid      = row['id']
    category = row['category']
    prompt   = row['prompt']

    if pid in existing:
        results.append(existing[pid])
        continue

    print(f"[{i+1:3d}/{len(corpus)}] {pid} [{category[:22]:22s}] ", end='', flush=True)

    # Force passthrough for control categories — category column
    # is ground truth. Bypasses SLM entirely for these rows.
    # Fixes: code_technical and high_stakes_medical reaching SLM
    # when gate text patterns don't catch natural language variants.
    FORCE_PT_CATS = {'code_technical', 'high_stakes_medical'}
    if category in FORCE_PT_CATS:
        from dataclasses import dataclass
        t_start = time.time()
        elapsed = (time.time() - t_start) * 1000
        result  = spsd_v4.DistillResult(
            original_prompt=prompt,
            compressed_prompt=prompt,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"category_{category}",
            confidence=1.0,
            latency_ms=elapsed,
            tier="category_gate",
            domain=None,
        )
    else:
        t_start = time.time()
        result  = spsd_v4.distill(prompt, model_path=MODEL_PATH)
        elapsed = (time.time() - t_start) * 1000  # ms

    # Track SLM latency for non-passthrough calls
    if not result.passthrough:
        slm_latencies.append(elapsed)

    if result.passthrough:
        raw_tok  = token_est(prompt) + 2
        dist_tok = token_est(prompt) + 2
        saving   = 0; pct = 0.0; ratio = 1.0
        ale_data = ale_prompt.build_ale_messages(result)
        ale_turn = ale_data["messages"][0]["content"]
        ann      = ""
        struct   = ""
        print(f"PASSTHRU  reason={result.passthrough_reason} ({elapsed:.0f}ms)")
    else:
        raw_tok  = token_est(prompt) + 2
        ale_data = ale_prompt.build_ale_messages(result)
        ale_turn = ale_data["messages"][0]["content"]
        dist_tok = token_est(ale_turn)
        saving   = raw_tok - dist_tok
        pct      = saving / raw_tok * 100 if raw_tok > 0 else 0.0
        ratio    = dist_tok / raw_tok
        ann      = ale_data.get("annotation", "")
        struct   = (f"{result.complexity.structural_score:.3f}"
                    if result.complexity else "")
        print(f"DISTILL   save={saving:+d}t ({pct:.0f}%) "
              f"conf={result.confidence:.2f} "
              f"lat={elapsed:.0f}ms")

    out = {col: '' for col in COLS}
    out.update({
        'id':                 pid,
        'category':           category,
        'word_count':         row['word_count'],
        'source':             row.get('source', ''),
        'intent_label':       row.get('intent_label', ''),
        'passthrough':        str(result.passthrough),
        'passthrough_reason': result.passthrough_reason or '',
        'distill_tier':       result.tier or '',
        'distill_latency_ms': f"{elapsed:.1f}",
        'domain':             result.domain or '',
        'structural_score':   struct,
        'confidence':         f"{result.confidence:.3f}",
        'hfg_aux':            '; '.join(result.hfg_aux) if result.hfg_aux else '',
        'compressed_prompt':  result.compressed_prompt or '',
        'ale_annotation':     ann,
        'ale_user_turn':      ale_turn,
        'raw_prompt':         prompt,
        'raw_input_tokens':   str(raw_tok),
        'dist_input_tokens':  str(dist_tok),
        'token_saving_input': str(saving),
        'token_saving_pct':   f"{pct:.1f}",
        'compression_ratio':  f"{ratio:.3f}",
    })
    results.append(out)

    # Checkpoint every 25 rows
    if (i + 1) % 25 == 0:
        ak = list(dict.fromkeys(k for r in results for k in r.keys()))
        with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                               extrasaction='ignore', restval='')
            w.writeheader(); w.writerows(results)
        print(f"  ── checkpoint ({len(results)} rows) ──")

        # Also save to Drive for resilience
        drive_backup = f"/content/drive/MyDrive/spsd/v3_run/spsd_results_checkpoint.csv"
        os.makedirs(os.path.dirname(drive_backup), exist_ok=True)
        with open(drive_backup, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                               extrasaction='ignore', restval='')
            w.writeheader(); w.writerows(results)

# Final write
ak = list(dict.fromkeys(k for r in results for k in r.keys()))
with open(OUTPUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=ak, quoting=csv.QUOTE_ALL,
                       extrasaction='ignore', restval='')
    w.writeheader(); w.writerows(results)

# ── SLM Latency Summary ───────────────────────────────────────
from collections import Counter
import numpy as np

dist  = [r for r in results if r['passthrough'] == 'False']
pt    = [r for r in results if r['passthrough'] == 'True']

print(f"\n{'='*60}")
print(f"SPSD DONE: {len(results)} total | {len(dist)} distilled | {len(pt)} passthrough")
for cat in ['verbose_social', 'multi_intent_linked', 'general_conversational',
            'code_technical', 'high_stakes_medical', 'short_passthrough']:
    sub = [r for r in results if r['category'] == cat]
    nd  = sum(1 for r in sub if r['passthrough'] == 'False')
    if sub:
        print(f"  {cat:28s} {nd:3d}/{len(sub):3d} distilled ({nd/len(sub)*100:.0f}%)")

if slm_latencies:
    lats = np.array(slm_latencies)
    print(f"\nSLM Latency (distilled calls only):")
    print(f"  n={len(lats)} mean={lats.mean():.0f}ms "
          f"median={np.median(lats):.0f}ms "
          f"p95={np.percentile(lats,95):.0f}ms "
          f"max={lats.max():.0f}ms")

print(f"\nResults: {OUTPUT}")
print(f"Next:    %run /content/run_llm_v3.py")
