"""
SPSD v4.4 — Step 1: Corpus Fetch (v4.0 — broad real-world corpus)
===================================================================
Fetches real user prompts from multiple sources.
No longer restricted to verbose_social service complaints.
Research paper scope is now broader: SPSD compresses any verbose prompt
where compression preserves response quality.

WHAT IS ACCEPTED:
  - Any English prompt, 20-400 words
  - Natural language (not code blocks)
  - Complete (not mid-sentence)
  - Any topic — service complaints, questions, creative writing,
    analysis requests, personal advice, explanations, tasks

WHAT IS REJECTED (hard filters):
  - CJK characters (Chinese/Japanese/Korean) — any presence
  - Non-ASCII letter ratio > 8% — garbled encoding or non-Latin scripts
  - ASCII-art / symbol-heavy content (>15% non-alphanumeric non-space chars)
  - Jailbreak attempts — DAN, "ignore all instructions", etc.
  - Actual code blocks (```...```) or code-dominant text (>35% code lines)
  - Formal legal boilerplate (FCRA, FDCPA, pursuant, herein)
  - Truncated / incomplete prompts
  - Near-duplicates (fingerprint deduplication)

CATEGORIES (broader than v3):
  verbose_social      — service complaints, personal situations, requests
                        with social scaffolding. High compression opportunity.
  multi_question      — prompts with 2+ distinct questions. Tests multi-Q handling.
  creative_task       — writing, story, content generation requests
  explanation_request — "explain X", "what is Y", "how does Z work"
  analysis_task       — evaluate, compare, analyse, review requests
  general_question    — factual questions, advice, opinions
  code_technical      — CONTROL: should all passthrough domain_code gate
  high_stakes_medical — CONTROL: should all passthrough domain_medical gate

SOURCES:
  1. allenai/WildChat-4.8M      — broad real user prompts
  2. HuggingFaceH4/ultrachat_200k — curated multi-turn conversations
  3. m-a-p/CodeFeedback-Filtered-Instruction — code tasks (control)
  4. GBaker/MedQA-USMLE-4-options — medical questions (control)
  5. lmsys/lmsys-chat-1m         — additional real user prompts

Output: /content/spsd_corpus_v3.csv
Run:    %run /content/fetch_corpus_v3.py

IMPORTANT: Run in a FRESH kernel (Runtime → Restart session).
datasets causes ArrowKeyError if imported twice per session.
"""

import sys, warnings, os
warnings.filterwarnings("ignore")

# ── Guard: datasets cannot be imported twice per Colab session ─
if any(k == 'datasets' or k.startswith('datasets.') for k in sys.modules):
    raise RuntimeError(
        "\ndatasets already imported this session — causes ArrowKeyError.\n"
        "Fix: Runtime → Restart session, then re-run this script.")

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
from collections import Counter, defaultdict

# ── Text utilities ─────────────────────────────────────────────
def clean(text: str) -> str:
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text.strip())

def wc(text: str) -> int:
    return len(text.split())

def is_english(text: str) -> bool:
    """
    Reject if:
    - Any CJK character present (Chinese/Japanese/Korean)
    - Any Arabic/Hebrew/Cyrillic/Devanagari characters
    - Non-ASCII letter ratio > 8%
    """
    # CJK
    if re.search(
            r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]',
            text):
        return False
    # Arabic, Hebrew, Cyrillic, Devanagari, Thai, etc.
    if re.search(
            r'[\u0600-\u06ff\u0400-\u04ff\u0900-\u097f'
             r'\u0e00-\u0e7f\u05d0-\u05ea]',
            text):
        return False
    letters = [c for c in text if c.isalpha()]
    if not letters: return False
    non_ascii = sum(1 for c in letters if ord(c) > 127)
    return non_ascii / len(letters) < 0.08

def is_ascii_clean(text: str) -> bool:
    """
    Reject ASCII-art, symbol-heavy, or garbled encoding.
    Non-alphanumeric non-space chars should be < 15% of all chars.
    """
    total = max(len(text), 1)
    symbol_chars = sum(
        1 for c in text
        if not c.isalnum() and not c.isspace()
    )
    return symbol_chars / total < 0.15

def is_natural_language(text: str) -> bool:
    """Reject code-dominant text."""
    lines = text.split('\n')
    code_lines = sum(
        1 for line in lines
        if re.match(
            r'\s*(def |class |import |function |SELECT |FROM |'
             r'```|\$[a-zA-Z]|#include|package |public |private |'
             r'return |const |let |var |fn |async |await )',
            line)
    )
    non_empty = max(1, sum(1 for l in lines if l.strip()))
    if code_lines / non_empty >= 0.35:
        return False
    # Reject if contains triple backtick code blocks
    if '```' in text:
        return False
    return True

def looks_complete(text: str) -> bool:
    """Reject truncated prompts."""
    s = text.strip()
    if re.search(r'[.!?:;"\')\]]$', s): return True
    last = s.split()[-1].rstrip('.,;:').lower() if s.split() else ''
    return last not in {
        'the','a','an','and','or','but','in','on','at','to','for','of',
        'with','by','from','that','which','is','are','was','were','be',
        'have','has','had','will','would','should','could','can','not',
        'i','my','me','we','they','them','their',
    }

# ── Deduplication ──────────────────────────────────────────────
seen = set()
def is_dup(text: str) -> bool:
    t  = re.sub(r'\s+', '', text.lower())
    fp = t[:80] + t[-30:]
    if fp in seen: return True
    seen.add(fp); return False

# ── Hard-reject patterns ───────────────────────────────────────
_JAILBREAK = re.compile(
    r'\b(DAN|do anything now|jailbreak|'
     r'ignore (your|all|previous|prior) (instructions?|programming|rules)|'
     r'forget everything you (learned|know)|'
     r'you are now|pretend you are|act as if you have no|'
     r'HARDRULES|system ann?ou-?ncement|'
     r'separate universe within a virtual machine|'
     r'hypothetically speaking.*no (rules|restrictions))\b',
    re.I | re.DOTALL)

_REWRITE_TASK = re.compile(
    r'\b(rewrite|rephrase|paraphrase|more human|quillbot|'
     r'reply this email|did you wrote|make this (more|sound)|'
     r'write this (more|in|as)|translate (this|the following)|'
     r'check (the following|this) (statement|grammar|text)|'
     r'correct (the|this|my) (following|grammar|text))\b', re.I)

_LEGAL_BOILERPLATE = re.compile(
    r'\b(pursuant|herein|aforementioned|respondent|complainant|'
     r'15 U\.S\.C|FCRA|FDCPA|TILA|RESPA|ECOA|'
     r'I am not liable|violation of federal|consumer reporting|'
     r'cease and desist|arbitration clause|class action)\b', re.I)

# Hard reject base checks
def hard_reject(text: str) -> bool:
    if not is_english(text):          return True
    if not is_ascii_clean(text):      return True
    if not is_natural_language(text): return True
    if not looks_complete(text):      return True
    if _JAILBREAK.search(text):       return True
    if _LEGAL_BOILERPLATE.search(text): return True
    if _REWRITE_TASK.search(text):    return True
    return False

# ── Category signals ───────────────────────────────────────────
_SVC = re.compile(
    r'\b(order|refund|cancel|subscription|charge|charged|billing|'
     r'deliver|shipping|tracking|return|exchange|customer service|'
     r'warranty|purchase|bought|landlord|tenant|rent|deposit|evict|'
     r'employer|employee|payslip|salary|dismissed|redundan|'
     r'insurance|claim|policy|bank|transaction|overcharged|'
     r'package|parcel|courier|boiler|heating|repair|maintenance|'
     r'complaint|dispute|appeal|grievance|escalat|'
     r'supplier|provider|utility|broadband|'
     r'account|membership|contract|chime|chase|santander|experian)\b',
    re.I)

_SOCIAL_SCAFFOLD = re.compile(
    r'\b(sorry|apologis|bother|thank you|really appreciate|'
     r'hope you|please help|if you could|if possible|'
     r"i've been|i have been|i've tried|i've called|i've sent|"
     r'they said|they told me|they keep|still waiting|'
     r'multiple times|several times|for (the past|weeks|months))\b',
    re.I)

_CREATIVE = re.compile(
    r'\b(write (a|an|the|me|us)|'
     r'create (a|an|the)|'
     r'compose (a|an)|'
     r'draft (a|an)|'
     r'generate (a|an)|'
     r'make (a|an|me|up)|'
     r'story|poem|essay|letter|email|speech|'
     r'blog post|article|script|screenplay|'
     r'short story|fiction|narrative|'
     r'mother.s day|birthday card|cover letter)\b',
    re.I)

_EXPLANATION = re.compile(
    r'\b(explain|what is|what are|what does|what do|'
     r'how does|how do|how is|how are|'
     r'why does|why do|why is|why are|'
     r'define|definition of|meaning of|'
     r'describe|tell me about|'
     r'difference between|compare|contrast|'
     r'what.s the|what.s a)\b',
    re.I)

_ANALYSIS = re.compile(
    r'\b(analyse|analyze|evaluate|assess|review|critique|'
     r'pros and cons|advantages|disadvantages|'
     r'summarise|summarize|summarization|'
     r'provide (a|an) (analysis|review|assessment|evaluation)|'
     r'give me (a|an) (analysis|review|assessment)|'
     r'thoughts on|opinion on|feedback on)\b',
    re.I)

# Code classifier — aligned with tier1_check two-signal logic
# Avoids false positives from common English words
_CODE_LANGUAGE = re.compile(
    r'\b(python|javascript|typescript|kotlin|swift|golang|'
     r'haskell|scala|php|bash|shell|yaml|kubernetes|webpack|'
     r'npm|pip|conda|pytorch|tensorflow|pandas|numpy|graphql|'
     r'react|vue|angular|svelte|django|flask|fastapi|express|'
     r'nodejs|spring|laravel|rails|docker|terraform)\b', re.I)

_CODE_TASK_V = re.compile(
    r'\b(write|build|create|implement|fix|debug|refactor|'
     r'optimise|optimize|generate|convert|parse|deploy|'
     r'execute|compile|configure|migrate|test|lint)\b', re.I)

_CODE_ARTEFACT_V = re.compile(
    r'\b(function|method|class|script|code|program|app|'
     r'algorithm|query|api|endpoint|component|module|loop|'
     r'array|database|schema|interface|library|package|'
     r'repository|pipeline|workflow|container)\b', re.I)

_CODE_ERRORS_V = re.compile(
    r'\b(IndentationError|SyntaxError|TypeError|NameError|'
     r'ValueError|AttributeError|ImportError|KeyError|'
     r'IndexError|RuntimeError|NullPointerException|'
     r'undefined is not|cannot read prop|is not a function|'
     r'unexpected token)\b', re.I)

_CODE_TECHNICAL_QUALS = [
    'output', 'return', 'print', 'input', 'loop', 'iterate',
    'recursive', 'async', 'query', 'api', 'endpoint', 'request',
    'response', 'json', 'xml', 'csv', 'html', 'css', 'database',
    'server', 'client', 'frontend', 'backend', 'deploy', 'test',
    'debug',
]

def _is_code_signal(text: str) -> bool:
    """
    Two-signal coding ask detector — aligned with tier1_check.
    Returns True only for genuine coding tasks.
    """
    if _CODE_LANGUAGE.search(text):       return True
    if _CODE_ERRORS_V.search(text):       return True
    if re.search(
            r'(def [a-z_]+[(]|class [A-Z]\w+[:(]|'
             r'import [a-z]|from [a-z]+ import|'
             r'function\s*\w*\s*[(]|#include\s*<)',
            text):
        return True
    if _CODE_TASK_V.search(text) and _CODE_ARTEFACT_V.search(text):
        return any(q in text.lower() for q in _CODE_TECHNICAL_QUALS)
    return False

_MULTI_Q = re.compile(
    r'\?.*\?|'
    r'\b(two (questions?|things?|parts?|points?)|'
     r'secondly|thirdly|'
     r'also (want|need|wonder|curious)|'
     r'additionally|furthermore|'
     r'another (question|thing|part)|'
     r'follow.?up|part (two|2|b|ii)|'
     r'first.*second.*third)\b',
    re.I | re.DOTALL)

def classify(text: str) -> str:
    """
    Classify a prompt into a category.
    Priority order matters — more specific first.
    Returns category name or None if no category fits.
    """
    words = wc(text)
    if words < 20 or words > 400:
        return None

    # Service complaint with social scaffolding → verbose_social
    if (_SVC.search(text)
            and _SOCIAL_SCAFFOLD.search(text)
            and 40 <= words <= 300):
        return "verbose_social"

    # Two or more questions → multi_question
    if (_MULTI_Q.search(text) and words >= 30
            and text.count('?') >= 2):
        return "multi_question"

    # Creative writing task → creative_task
    if _CREATIVE.search(text) and words >= 25:
        return "creative_task"

    # Analysis/evaluation → analysis_task
    if _ANALYSIS.search(text) and words >= 25:
        return "analysis_task"

    # Explanation/definition → explanation_request
    if _EXPLANATION.search(text) and words >= 20:
        return "explanation_request"

    # Anything else that passes hard filters → general_question
    if words >= 20:
        return "general_question"

    return None

# ── Corpus store ───────────────────────────────────────────────
prompts      = []
cat_counts   = Counter()
TARGETS = {
    "verbose_social":      80,
    "multi_question":      50,
    "creative_task":       50,
    "analysis_task":       40,
    "explanation_request": 40,
    "general_question":    40,
    "code_technical":      20,  # control
    "high_stakes_medical": 15,  # control
}
TOTAL_TARGET = sum(TARGETS.values())

def add(cat: str, text: str, src: str, intent: str = "") -> bool:
    text = clean(text)
    if not text: return False
    if hard_reject(text): return False
    if is_dup(text): return False
    if cat_counts[cat] >= TARGETS.get(cat, 999): return False
    prompts.append({
        "id":           "",
        "category":     cat,
        "word_count":   wc(text),
        "prompt":       text,
        "source":       src,
        "intent_label": intent,
    })
    cat_counts[cat] += 1
    return True

def needs_more() -> bool:
    return any(cat_counts[c] < TARGETS[c]
               for c in TARGETS
               if c not in ("code_technical", "high_stakes_medical"))

def status() -> str:
    parts = [f"{c[:8]}={cat_counts[c]}/{TARGETS[c]}"
             for c in TARGETS]
    return " | ".join(parts)

# ═══════════════════════════════════════════════════════════════
# SOURCE 1: WildChat — primary source for all non-control categories
# ═══════════════════════════════════════════════════════════════
print(f"[1/4] WildChat-4.8M — scanning for all categories...")
print(f"      Targets: {TARGETS}")
print(f"      Hard filters: CJK, non-ASCII>8%, ASCII-art, jailbreak, legal boilerplate")
print(flush=True)

MAX_SCAN_WC = 200000
scanned     = 0

try:
    for row in load_dataset(
            "allenai/WildChat-4.8M", split="train", streaming=True):
        scanned += 1
        if scanned > MAX_SCAN_WC: break
        if not needs_more(): break

        # Pre-filters
        if row.get("language", "English") != "English": continue
        if row.get("toxic", False): continue

        conv  = row.get("conversation", [])
        first = next((t for t in conv if t.get("role") == "user"), None)
        if not first: continue
        text  = first.get("content", "").strip()

        cat = classify(text)
        if cat and cat not in ("code_technical", "high_stakes_medical"):
            add(cat, text, "allenai/WildChat-4.8M", cat)

        if scanned % 20000 == 0:
            print(f"  scanned={scanned:,} | {status()}", flush=True)

except Exception as e:
    print(f"  WildChat ERROR: {e}")

print(f"  WildChat done: scanned={scanned:,}")
print(f"  {status()}\n")

# ═══════════════════════════════════════════════════════════════
# SOURCE 2: LMSYS Chat 1M — top-up for underrepresented categories
# ═══════════════════════════════════════════════════════════════
if needs_more():
    print(f"[2/4] LMSYS Chat 1M — top-up...")
    MAX_SCAN_LMSYS = 30000
    scanned_lmsys  = 0
    try:
        for row in load_dataset(
                "lmsys/lmsys-chat-1m", split="train", streaming=True):
            scanned_lmsys += 1
            if scanned_lmsys > MAX_SCAN_LMSYS: break
            if not needs_more(): break

            if row.get("language", "English") != "English": continue
            conv  = row.get("conversation", [])
            first = next((t for t in conv if t.get("role") == "user"), None)
            if not first: continue
            text  = first.get("content", "").strip()

            cat = classify(text)
            if cat and cat not in ("code_technical", "high_stakes_medical"):
                add(cat, text, "lmsys/lmsys-chat-1m", cat)

    except Exception as e:
        print(f"  LMSYS ERROR: {e}")
    print(f"  LMSYS done: scanned={scanned_lmsys:,}")
    print(f"  {status()}\n")

# ═══════════════════════════════════════════════════════════════
# SOURCE 3: UltraChat — top-up for explanation + analysis + multi-Q
# ═══════════════════════════════════════════════════════════════
NEED_TOPUP = ["multi_question", "explanation_request", "analysis_task"]
if any(cat_counts[c] < TARGETS[c] for c in NEED_TOPUP):
    print(f"[3/4] UltraChat 200k — top-up for {NEED_TOPUP}...")
    try:
        for row in load_dataset(
                "HuggingFaceH4/ultrachat_200k",
                split="train_sft", streaming=True):
            if all(cat_counts[c] >= TARGETS[c] for c in NEED_TOPUP):
                break
            msgs  = row.get("messages", [])
            first = next((m for m in msgs if m.get("role") == "user"), None)
            if not first: continue
            text  = first.get("content", "").strip()
            cat   = classify(text)
            if cat in NEED_TOPUP:
                add(cat, text, "HuggingFaceH4/ultrachat_200k", cat)
    except Exception as e:
        print(f"  UltraChat ERROR: {e}")
    print(f"  {status()}\n")

# ═══════════════════════════════════════════════════════════════
# SOURCE 4: Control groups (code + medical)
# ═══════════════════════════════════════════════════════════════
print(f"[4/4] Control groups — code_technical + high_stakes_medical...")

# Code — CodeFeedback
try:
    for row in load_dataset(
            "m-a-p/CodeFeedback-Filtered-Instruction",
            split="train", streaming=True):
        if cat_counts["code_technical"] >= TARGETS["code_technical"]: break
        text = row.get("query", "").strip()
        if not (15 <= wc(text) <= 300): continue
        if text.strip().startswith("```"): continue
        if not is_natural_language(text): continue
        if not is_english(text): continue
        if _CODE_SIGNAL.search(text):
            add("code_technical", text,
                "m-a-p/CodeFeedback-Filtered-Instruction", "coding")
except Exception as e:
    print(f"  CodeFeedback ERROR: {e}")

# Medical — MedQA
try:
    for row in load_dataset(
            "GBaker/MedQA-USMLE-4-options",
            split="train", streaming=True):
        if cat_counts["high_stakes_medical"] >= TARGETS["high_stakes_medical"]: break
        text = row.get("question", "").strip()
        if wc(text) < 20: continue
        if not is_english(text): continue
        add("high_stakes_medical", text,
            "GBaker/MedQA-USMLE-4-options", "clinical_vignette")
except Exception as e:
    print(f"  MedQA ERROR: {e}")

print(f"  {status()}\n")

# ═══════════════════════════════════════════════════════════════
# Write corpus
# ═══════════════════════════════════════════════════════════════
CAT_ORDER = [
    "verbose_social", "multi_question", "creative_task",
    "analysis_task", "explanation_request", "general_question",
    "code_technical", "high_stakes_medical",
]

random.seed(42)
final = []
for cat in CAT_ORDER:
    subset = [p for p in prompts if p["category"] == cat]
    random.shuffle(subset)
    final.extend(subset)

for i, p in enumerate(final, 1):
    p["id"] = f"P{i:03d}"

OUTPUT = "/content/spsd_corpus_v3.csv"
with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["id", "category", "word_count",
                    "prompt", "source", "intent_label"],
        quoting=csv.QUOTE_ALL)
    w.writeheader()
    w.writerows(final)

# ── Summary ────────────────────────────────────────────────────
wc_by_cat = defaultdict(list)
for p in final:
    wc_by_cat[p["category"]].append(p["word_count"])

# Expected distillation rates based on Gemma testing
EXPECTED_DIST = {
    "verbose_social":      0.90,
    "multi_question":      0.70,
    "creative_task":       0.70,
    "analysis_task":       0.65,
    "explanation_request": 0.60,
    "general_question":    0.50,
    "code_technical":      0.00,  # all passthrough
    "high_stakes_medical": 0.00,  # all passthrough
}

print(f"\n{'='*72}")
print(f"CORPUS v4.0  →  {OUTPUT}")
print(f"{'='*72}")
print(f"\n  {'Category':24s} {'N':>5} {'AvgW':>6} {'Min':>5} {'Max':>5} "
      f"{'~Dist':>7}  Note")
print(f"  {'-'*68}")

total_expected = 0
for cat in CAT_ORDER:
    wcs = wc_by_cat.get(cat, [0])
    nc  = len(wcs)
    avg = sum(wcs) // max(nc, 1)
    exp = int(nc * EXPECTED_DIST.get(cat, 0.5))
    total_expected += exp
    note = "CONTROL (passthrough)" if EXPECTED_DIST.get(cat, 1) == 0 else ""
    print(f"  {cat:24s} {nc:5d} {avg:6d} {min(wcs):5d} {max(wcs):5d} "
          f"{exp:7d}  {note}")

print(f"\n  Total prompts:      {len(final)}")
print(f"  Expected distilled: ~{total_expected}")

# Source breakdown
src_counts = Counter(p["source"] for p in final)
print(f"\n  Sources:")
for src, count in src_counts.most_common():
    print(f"    {count:4d}  {src}")

print(f"\n  Hard filters applied:")
print(f"    CJK characters rejected")
print(f"    Non-ASCII letter ratio > 8% rejected")
print(f"    ASCII-art / symbol-heavy rejected")
print(f"    Jailbreak patterns rejected")
print(f"    Legal boilerplate rejected")
print(f"    Code-dominant text rejected")
print(f"    Incomplete prompts rejected")
print(f"    Near-duplicate fingerprints rejected")

print(f"\n  NEXT STEP:")
print(f"    Update run_spsd_v3.py CORPUS path to:")
print(f"    CORPUS = '/content/spsd_corpus_v3.csv'")
print(f"    Then: %run /content/run_spsd_v3.py")
