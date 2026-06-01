"""
SPSD v4.3 — Step 1: Corpus Fetch (v3.4 — final verbose_social filter)
=======================================================================
Uses HuggingFace `datasets` library with streaming=True.
No REST API. No auth required. Works in Colab free tier.

IMPORTANT — run in a FRESH kernel (Runtime → Restart session first).
The `datasets` library causes ArrowKeyError if imported twice per session.

verbose_social filter (v3.4 — calibrated against 12 test cases, all pass):
  REQUIRES ALL THREE:
    1. 28-400 words (long enough to compress, short enough to be a real prompt)
    2. Service/complaint topic signal (order, refund, landlord, billing, etc.)
    3. Narrative verbs (I've tried, they keep, still haven't) OR apology/politeness
  REJECTS:
    Any prompt with technical signals (ML, code, recipe, essay, workout, etc.)
  This correctly captures:
    - "I've been trying to cancel my gym membership for two months..."
    - "My landlord has been ignoring repair requests for six weeks..."
    - "Hi I'm so sorry to bother you, I ordered ORD-847 and it hasn't arrived..."
  And correctly rejects:
    - YOLOv8 / ML training questions
    - Google/YouTube search tech issues
    - Dating profiles, essays, creative writing
    - Python/programming learning questions

Corpus targets:
  verbose_social:         80+  WildChat service/complaint + narrative   (~92% distilled)
  multi_intent_linked:    50+  WildChat/UltraChat 2+ questions          (~64% distilled)
  general_conversational: 55+  WildChat diverse >=25w                   (~40% distilled)
  code_technical:         20+  CodeFeedback NL queries                  (~80% distilled)
  high_stakes_medical:    15+  MedQA-USMLE                              (  0% — CONTROL)
  short_passthrough:      15+  WildChat <=15w                           (  0% — CONTROL)

Expected distilled: ~120+ out of ~235 fetched

Output: /content/spsd_corpus_v3.csv
Run:    %run /content/fetch_corpus_v3.py
"""

import sys, warnings
warnings.filterwarnings("ignore")

# ── Guard: datasets cannot be imported twice per Colab session ────────────────
if any(k == 'datasets' or k.startswith('datasets.') for k in sys.modules):
    raise RuntimeError(
        "\n\ndatasets already imported this session — causes ArrowKeyError.\n"
        "Fix: Runtime → Restart session, then re-run this script."
    )

_removed = [p for p in ['/content', ''] if p in sys.path]
for p in _removed:
    sys.path.remove(p)

print("Loading HuggingFace datasets library...", flush=True)
from datasets import load_dataset
print("Ready.\n", flush=True)

for p in _removed:
    if p not in sys.path:
        sys.path.insert(0, p)

import csv, re, random
from collections import Counter

# ─── Text utilities ────────────────────────────────────────────────────────────
def clean(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text.strip())

def wc(text): return len(text.split())

def is_english(text):
    common = {'the','a','an','is','are','was','were','i','you','my','me',
              'we','it','to','of','in','for','and','or','can','do','have',
              'has','not','but','be','this','that','with','on','at','your'}
    return len(common & set(text.lower().split())) >= 3

def is_natural_language(text):
    code_lines = sum(1 for line in text.split('\n')
                     if re.match(r'\s*(def |class |import |function |SELECT |FROM '
                                 r'|```|\$[a-zA-Z]|#include|package |<[a-zA-Z])', line))
    total = max(1, len([l for l in text.split('\n') if l.strip()]))
    return (code_lines / total) < 0.35

def looks_complete(text):
    s = text.strip()
    if re.search(r'[.!?:;"\'\)\]]$', s): return True
    last = s.split()[-1].rstrip('.,;:').lower() if s.split() else ''
    return last not in {
        'the','a','an','and','or','but','in','on','at','to','for','of',
        'with','by','from','that','which','who','when','where','how',
        'what','if','as','is','are','was','were','be','been','have',
        'has','had','will','would','should','could','can','may','might',
        'must','not','also','both','either','neither','each','every',
        'any','all','some','other','another','include','including',
        'such','these','those',
    }

def is_clean(text):
    return not any(re.search(p, text.lower(), re.I) for p in [
        r'\b(DAN|do anything now|ignore (previous|all) (instructions|rules))\b',
        r'\b(pretend you (are|have no)|act as if you have no|forget you are an AI)\b',
        r'\b(erotic|explicit sexual|NSFW|hentai|pornograph)\b',
    ])

# ─── Deduplication ─────────────────────────────────────────────────────────────
seen = set()
def is_dup(text):
    t  = re.sub(r'\s+', '', text.lower())
    fp = t[:80] + t[-30:]
    if fp in seen: return True
    seen.add(fp); return False

# ─── Corpus store ──────────────────────────────────────────────────────────────
prompts = []
def add(cat, text, src, intent=""):
    text = clean(text)
    if not text or is_dup(text) or not is_english(text): return False
    if not looks_complete(text) or not is_clean(text): return False
    prompts.append({"id":"","category":cat,"word_count":wc(text),
                    "prompt":text,"source":src,"intent_label":intent})
    return True
def n(cat): return sum(1 for p in prompts if p["category"]==cat)

# ─── verbose_social filter v3.4 ───────────────────────────────────────────────
# Service/complaint topic — real transactional signals
_SVC = re.compile(
    r'\b(order|refund|cancel|subscription|charge|charged|invoice|billing|billed|'
     r'deliver|shipping|tracking|return|exchange|customer service|'
     r'warranty|purchase|bought|landlord|tenant|rent|deposit|evict|'
     r'employer|employee|payslip|salary|dismissed|redundan|'
     r'insurance|claim|policy|bank|transaction|overcharged|'
     r'package|parcel|courier|boiler|heating|repair|maintenance|'
     r'complaint|dispute|appeal|grievance|escalat|'
     r'supplier|provider|utility|broadband|'
     r'account|membership|contract)\b', re.I)

# Narrative verbs — describing an ongoing situation with history
_NARR = re.compile(
    r'\b(i\'ve been|i have been|i\'ve tried|i have tried|i\'ve called|'
     r'i\'ve sent|i\'ve contacted|i\'ve spoken|i\'ve raised|'
     r'i\'ve already|i\'ve submitted|they have|they\'ve|they said|'
     r'they told me|they keep|they haven\'t|they didn\'t|'
     r'nothing has|still waiting|still haven\'t|keeps happening|'
     r'for the past|for weeks|for months|multiple times|several times|'
     r'three times|twice|again and again)\b', re.I)

# Apology/politeness — direct social scaffolding
_APOL = re.compile(
    r'\b(sorry|apologis|apologiz|forgive me|bother you|trouble you|'
     r'thank you so much|really appreciate|hope you|'
     r'i\'m writing to|reaching out because|'
     r'i\'m worried|i\'m anxious|i\'m desperate|'
     r'please forgive|please bear with|if you could please)\b', re.I)

# Technical disqualifiers — one match kills the prompt
_TECH = re.compile(
    r'\b(train|training data|neural network|machine learning|deep learning|'
     r'pytorch|tensorflow|yolo|object detection|dataset|algorithm|'
     r'satellite|pixel|vector|matrix|compile|runtime|'
     r'essay|thesis|dissertation|research paper|academic|citation|'
     r'recipe|ingredient|tablespoon|bake|cuisine|'
     r'poem|haiku|sonnet|fiction|screenplay|translate|grammar|'
     r'workout|exercise routine|protein|calorie|macro)\b', re.I)

def is_verbose_social(text):
    words = len(text.split())
    if words < 28 or words > 400: return False
    if not is_natural_language(text): return False
    if _TECH.search(text): return False                     # any tech signal → reject
    if not _SVC.search(text): return False                  # must have service topic
    return _NARR.search(text) or _APOL.search(text)        # must have narrative OR apology

# ─── Other signals ─────────────────────────────────────────────────────────────
_MULTI = re.compile(
    r'\?.*\?|'
    r'\b(two (questions?|things?|parts?)|secondly|thirdly|'
     r'also (want|need|wonder|ask)|furthermore|additionally|'
     r'another (question|thing)|follow.?up|part (two|2|b|ii))\b',
    re.I | re.DOTALL)

_SVC_GEN = re.compile(   # for general_conv — exclude heavy-service prompts
    r'\b(order|refund|cancel|subscription|charged|billing|delivery|'
     r'shipped|tracking|return|exchange|warranty|purchase)\b', re.I)

_CODE = re.compile(
    r'\b(function|method|class|variable|error|bug|exception|compile|'
     r'runtime|loop|array|list|dict|string|integer|python|javascript|'
     r'java|c\+\+|sql|html|css|api|algorithm|implement|write a|create a|'
     r'return|parameter|argument|recursive|iterate)\b', re.I)

# ═══════════════════════════════════════════════════════════════════════════════
# WildChat pass 1: verbose_social + general_conv + short_passthrough
# Single pass collects all three — exits when all targets met
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_VS = 80; TARGET_GC = 55; TARGET_SP = 15
MAX_SCAN  = 100000   # large scan — strict filter needs more rows to find 80 prompts

print(f"[Pass 1] WildChat: verbose_social({TARGET_VS}) + "
      f"general_conv({TARGET_GC}) + short({TARGET_SP})")
print(f"  Scanning up to {MAX_SCAN:,} rows...", flush=True)

scanned = 0
try:
    for row in load_dataset("allenai/WildChat-4.8M", split="train", streaming=True):
        scanned += 1
        if scanned > MAX_SCAN: break
        if (n("verbose_social")        >= TARGET_VS and
            n("general_conversational") >= TARGET_GC and
            n("short_passthrough")      >= TARGET_SP):
            print(f"  All targets met at row {scanned:,}.")
            break

        if row.get("language","English") != "English": continue
        if row.get("toxic", False): continue
        conv  = row.get("conversation", [])
        first = next((t for t in conv if t.get("role")=="user"), None)
        if not first: continue
        text  = first.get("content","").strip()
        words = wc(text)

        # verbose_social — checked first, strictest filter
        if n("verbose_social") < TARGET_VS and is_verbose_social(text):
            add("verbose_social", text, "allenai/WildChat-4.8M", "service_complaint")

        # general_conversational — diverse, 25-400w, not service-heavy
        elif (n("general_conversational") < TARGET_GC
                and 25 <= words <= 400
                and len(_SVC_GEN.findall(text)) < 2
                and text.count("?") < 2
                and is_natural_language(text)):
            add("general_conversational", text, "allenai/WildChat-4.8M", "general")

        # short_passthrough — <=15w control group
        elif n("short_passthrough") < TARGET_SP and 4 <= words <= 15 and is_english(text):
            add("short_passthrough", text, "allenai/WildChat-4.8M", "short_query")

        if scanned % 10000 == 0:
            print(f"  row={scanned:7,d} | "
                  f"vs={n('verbose_social'):3d}/{TARGET_VS}  "
                  f"gc={n('general_conversational'):3d}/{TARGET_GC}  "
                  f"sp={n('short_passthrough'):2d}/{TARGET_SP}", flush=True)

except Exception as e:
    print(f"  WildChat pass 1 ERROR: {e}")

print(f"  => vs={n('verbose_social')}  "
      f"gc={n('general_conversational')}  "
      f"sp={n('short_passthrough')}")

# ═══════════════════════════════════════════════════════════════════════════════
# WildChat pass 2: multi_intent_linked
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_MI = 50
print(f"\n[Pass 2] WildChat: multi_intent_linked({TARGET_MI}+)...")

try:
    scanned_mi = 0
    for row in load_dataset("allenai/WildChat-4.8M", split="train", streaming=True):
        scanned_mi += 1
        if scanned_mi > 20000 or n("multi_intent_linked") >= TARGET_MI: break
        if row.get("language","English") != "English": continue
        if row.get("toxic", False): continue
        conv  = row.get("conversation", [])
        first = next((t for t in conv if t.get("role")=="user"), None)
        if not first: continue
        text  = first.get("content","").strip()
        words = wc(text)
        if not (25 <= words <= 400): continue
        if text.count("?") < 2 and not _MULTI.search(text): continue
        if not is_natural_language(text): continue
        add("multi_intent_linked", text, "allenai/WildChat-4.8M", "multi_question")
        if scanned_mi % 5000 == 0:
            print(f"  row={scanned_mi:6,d} | mi={n('multi_intent_linked'):3d}/{TARGET_MI}",
                  flush=True)
except Exception as e:
    print(f"  WildChat multi-intent ERROR: {e}")

if n("multi_intent_linked") < TARGET_MI:
    print(f"  UltraChat top-up ({TARGET_MI - n('multi_intent_linked')} needed)...")
    try:
        for row in load_dataset("HuggingFaceH4/ultrachat_200k",
                                 split="train_sft", streaming=True):
            if n("multi_intent_linked") >= TARGET_MI: break
            msgs  = row.get("messages", [])
            first = next((m for m in msgs if m.get("role")=="user"), None)
            if not first: continue
            text = first.get("content","").strip()
            if not (25 <= wc(text) <= 400): continue
            if text.count("?") < 2: continue
            if not is_natural_language(text): continue
            add("multi_intent_linked", text, "HuggingFaceH4/ultrachat_200k", "multi_question")
    except Exception as e:
        print(f"  UltraChat ERROR: {e}")

print(f"  => multi_intent_linked={n('multi_intent_linked')}")

# ═══════════════════════════════════════════════════════════════════════════════
# CodeFeedback: code_technical
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_CT = 20
print(f"\n[CodeFeedback] code_technical({TARGET_CT}+)...")
try:
    for row in load_dataset("m-a-p/CodeFeedback-Filtered-Instruction",
                             split="train", streaming=True):
        if n("code_technical") >= TARGET_CT: break
        text = row.get("query","").strip()
        if not (15 <= wc(text) <= 300): continue
        if text.strip().startswith("```"): continue
        if not is_natural_language(text): continue
        if not _CODE.search(text): continue
        add("code_technical", text, "m-a-p/CodeFeedback-Filtered-Instruction", "coding")
except Exception as e:
    print(f"  CodeFeedback ERROR: {e}")
print(f"  => code_technical={n('code_technical')}")

# ═══════════════════════════════════════════════════════════════════════════════
# MedQA: high_stakes_medical (CONTROL — all passthrough)
# ═══════════════════════════════════════════════════════════════════════════════
TARGET_MED = 15
print(f"\n[MedQA-USMLE] high_stakes_medical({TARGET_MED}+)...")
try:
    for row in load_dataset("GBaker/MedQA-USMLE-4-options",
                             split="train", streaming=True):
        if n("high_stakes_medical") >= TARGET_MED: break
        text = row.get("question","").strip()
        if wc(text) < 20: continue
        add("high_stakes_medical", text, "GBaker/MedQA-USMLE-4-options", "clinical_vignette")
except Exception as e:
    print(f"  MedQA ERROR: {e}")
print(f"  => high_stakes_medical={n('high_stakes_medical')}")

# ═══════════════════════════════════════════════════════════════════════════════
# Write CSV — full text, no truncation
# ═══════════════════════════════════════════════════════════════════════════════
CAT_ORDER = ["verbose_social","multi_intent_linked","general_conversational",
             "code_technical","high_stakes_medical","short_passthrough"]
random.seed(42)
final = []
for cat in CAT_ORDER:
    subset = [p for p in prompts if p["category"]==cat]
    random.shuffle(subset)
    final.extend(subset)
for i,p in enumerate(final,1):
    p["id"] = f"P{i:03d}"

OUTPUT = "/content/spsd_corpus_v3.csv"
with open(OUTPUT,"w",newline="",encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id","category","word_count",
                                       "prompt","source","intent_label"],
                       quoting=csv.QUOTE_ALL)
    w.writeheader(); w.writerows(final)

# ─── Summary ──────────────────────────────────────────────────────────────────
DIST_RATES = {"verbose_social":0.92,"multi_intent_linked":0.64,
              "general_conversational":0.40,"code_technical":0.80,
              "high_stakes_medical":0.0,"short_passthrough":0.0}
wc_by_cat = {}
for p in final: wc_by_cat.setdefault(p["category"],[]).append(p["word_count"])

print(f"\n{'='*72}")
print(f"CORPUS v3  ->  {OUTPUT}")
print(f"{'='*72}")
print(f"\n  {'Category':28s} {'N':>5} {'AvgW':>6} {'MinW':>5} {'MaxW':>5} {'~Dist':>8}")
print(f"  {'-'*66}")
total_exp = 0
for cat in CAT_ORDER:
    nc  = sum(1 for p in final if p["category"]==cat)
    wcs = wc_by_cat.get(cat,[0])
    exp = int(nc * DIST_RATES.get(cat,0)); total_exp += exp
    flag = ("  ← PRIMARY" if cat in ("verbose_social","multi_intent_linked",
                                      "general_conversational") else
            "  ← TEST"    if cat == "code_technical" else
            "  ← CONTROL")
    print(f"  {cat:28s} {nc:5d} {sum(wcs)//max(len(wcs),1):6d} "
          f"{min(wcs):5d} {max(wcs):5d} {exp:>8d}{flag}")

print(f"\n  Total:              {len(final)}")
print(f"  Expected distilled: ~{total_exp}")
print(f"\n  verbose_social:  {n('verbose_social')} "
      f"({'OK ✓' if n('verbose_social') >= 60 else 'LOW — may need more scan rows'})")
if n('verbose_social') < 60:
    print(f"  To increase: raise MAX_SCAN above {MAX_SCAN:,} in this script")
print(f"\nNext: %run /content/run_spsd_v3.py")
