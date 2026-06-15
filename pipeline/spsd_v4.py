"""
SPSD v4.1 — Semantic Prompt Structural Distillation
SLM-based single-pass distillation pipeline with complexity scorer
and High Fidelity Guard (HFG).

Architecture:
    User prompt
        ↓
    Tier 1      — rule-based guards (~0ms, zero model cost)
        ↓ if not passthrough
    Complexity  — rule-based scorer (~0ms): social/semantic/structural/
    Scorer        repetition scores → compression_ceiling, recommended_ratio
                  hard passthrough if structural density too high
        ↓ if not passthrough
    HFG         — High Fidelity Guard (~0ms): extracts high-entropy
                  emotional/life-event/dependency phrases into aux context
                  before SLM runs, ensuring they are never dropped
        ↓
    SLM         — Qwen2.5-1.5B-Instruct Q4_K_M via llama-cpp-python
                  single call → compressed_prompt + metadata envelope
                  receives: domain tag + complexity-guided compression target
                  + HFG-seeded aux context
        ↓
    Tier 1b     — fallback safety re-check on compressed_prompt (~0ms)
        ↓
    ALE         — builds frontier LLM packet

Design constraints:
    - SLM must run on CPU, target <200ms on free-tier Colab CPU
    - No frontier model dependency in distillation layer
    - Passthrough is not failure — it is the system working correctly
    - Urgency/tone scored on original text, never distilled payload
    - HFG phrases are guaranteed to appear in aux regardless of SLM output

v4.1 additions over v4.0:
    - ComplexityScorer: rule-based 5-dimension scorer, feeds compression
      target to SLM and triggers structural passthrough
    - HighFidelityGuard: extracts life events, emotional anchors,
      dependency markers, crisis indicators into guaranteed aux context
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

MODEL_REPO   = "bartowski/gemma-2-2b-it-GGUF"
MODEL_FILE   = "gemma-2-2b-it-Q4_K_M.gguf"
MODEL_PATH   = Path("/content/drive/MyDrive/spsd/models") / MODEL_FILE

# Inference parameters — tuned for latency on free-tier Colab CPU
SLM_CONTEXT_LENGTH   = 4096  # raised from 2048 — eliminates n_ctx warning, supports prompts up to ~3000 tokens
SLM_MAX_TOKENS       = 150    # agreed cap — no sentence cap needed
SLM_TEMPERATURE      = 0.0    # deterministic — agreed after Gemma testing
SLM_N_THREADS        = 4      # used throughout all testing

# ─────────────────────────────────────────────
# TOKEN ECONOMICS CONFIGURATION
# ─────────────────────────────────────────────

# Short prompt guard — prompts at or under this word count passthrough.
# Raised from 5 to 15: anything shorter than ~15 words cannot produce
# net-positive token saving once ALE header overhead (~35 tokens) is added.
SHORT_PROMPT_WORD_LIMIT = 15

# ALE header overhead in tokens — compact format:
# "D|tone|urgency|" = ~6 tokens fixed (W:<n> removed in v4.1).
# Passthrough format "P\n" = ~2 tokens (negligible).
# Aux items encoded inline as semicolon-separated — ~2 tokens per item.
ALE_HEADER_OVERHEAD_TOKENS = 6

# Minimum net token saving to justify distillation.
# Set to 10 — a saving of fewer than 10 tokens is noise, not a win.
MIN_NET_TOKEN_SAVING = 10

# Dynamic confidence threshold — varies by prompt length.
# Short prompts: higher bar (less room for error, smaller saving upside).
# Long prompts: standard bar (more room to compress faithfully).
def dynamic_confidence_threshold(word_count: int) -> float:
    """
    word_count 16-30 : 0.80 — short prompts, need high confidence
    word_count 31-60 : 0.72 — medium prompts, moderate bar
    word_count > 60  : 0.65 — long prompts, standard bar
    """
    if word_count <= 30:
        return 0.80
    elif word_count <= 60:
        return 0.68
    else:
        return 0.65

# ─────────────────────────────────────────────
# TIER 1 — DOMAIN KEYWORD SETS
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# TIER 1 — DOMAIN DETECTION (tag, not block)
# ─────────────────────────────────────────────
# Tier 1 no longer passthroughs on domain.
# It detects domain and attaches a marker so the SLM
# can compress with domain awareness.
# Only TRUE passthrough cases: prompts so short distillation
# cannot be net-positive regardless of compression quality.
# ─────────────────────────────────────────────

# Hard passthrough — prompts containing these should never be
# compressed because exact wording is safety-critical and
# any loss of fidelity could cause real harm.
# Keep this list SHORT and unambiguous.
HARD_PASSTHROUGH_PHRASES = {
    "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
    "heart attack", "overdose", "suicidal", "suicide", "self harm",
    "bleeding heavily", "unconscious", "not breathing", "anaphylaxis",
    "911", "999", "112",   # emergency numbers
}

# Domain detection — used for tagging only, not blocking
MEDICAL_KEYWORDS = {
    # Conditions and diagnoses
    "diagnosis", "symptom", "symptoms", "condition", "disorder", "disease",
    "cancer", "tumor", "tumour", "diabetes", "insulin", "chemotherapy",
    "vaccine", "allergy", "allergies", "seizure", "stroke", "cardiac",
    "hypertension", "depression", "anxiety", "schizophrenia", "bipolar",
    "adhd", "autism", "infection", "fracture", "inflammation", "lesion",
    "haemorrhage", "hemorrhage", "anemia", "anaemia", "chronic", "acute",
    # Treatments and medications
    "prescription", "medication", "medications", "dosage", "dose",
    "surgery", "surgical", "ibuprofen", "paracetamol", "antibiotic",
    "antibiotics", "antidepressant", "opioid", "narcotic", "painkiller",
    "chemotherapy", "radiotherapy", "biopsy", "mri", "ct scan",
    "pathology", "prognosis", "palliative", "therapy", "treatment",
    "decongestant", "decongestants",
    # Clinical roles and settings
    "physician", "doctor", "nurse", "pediatrician", "paediatrician",
    "psychiatrist", "psychologist", "surgeon", "cardiologist",
    "emergency department", "hospital", "clinic", "primary care",
    # Clinical presentation language — core USMLE patterns
    "presents", "presentation", "presenting",
    "patient", "gestation", "pregnant", "pregnancy",
    "primigravida", "gravida", "trimester", "prenatal", "postnatal",
    "neonatal", "infant", "newborn", "perinatal",
    "complains", "complaining", "complaints",
    "examination", "checkup", "check-up",
    "menorrhagia", "diplopia", "urination", "dysuria",
    "autopsy", "died suddenly", "cause of death", "sudden death",
    # Body systems
    "cardiac", "pulmonary", "hepatic", "renal", "neurological",
    "gastrointestinal", "musculoskeletal", "dermatological",
    "blood pressure", "heart rate", "respiratory",
}

LEGAL_KEYWORDS = {
    # Tier 1 — unambiguous legal procedure terms only.
    # Each of these essentially never appears in a service complaint.
    # Removed: verdict, rights, legal rights, tenant rights, divorce,
    #          custody, bankruptcy, foreclosure, wrongful, negligence,
    #          liability, attorney, lawyer, litigation, court —
    #          all appear in CFPB-style service complaints as context.
    "plaintiff", "defendant", "subpoena",
    "statute", "jurisdiction", "injunction",
    "employment law", "unfair dismissal", "small claims",
    "restraining order", "breach of contract",
    "patent", "trademark", "copyright infringement",
    "gdpr", "arbitration", "malpractice",
}

# Tier 2 legal keywords — need legal-seeking framing to fire.
# These appear in service complaints as context (e.g. "they
# threatened a lawsuit", "I contacted an attorney") but are
# also genuinely legal when the person is seeking legal action.
_LEGAL_TIER2 = {
    "lawsuit", "attorney", "lawyer", "litigation", "court", "legal advice",
}

# Legal-seeking framing — required alongside Tier 2 keywords.
# The person must be asking about taking legal action,
# not merely describing a situation involving legal elements.
_LEGAL_SEEKING = re.compile(
    r"\b("
    r"need (a |an )?(lawyer|attorney|solicitor|legal advice)|"
    r"(file|filing|filed) (a |the )?(lawsuit|claim|complaint|case|suit)|"
    r"(take|taking|took) (them|him|her|the company|it) to court|"
    r"(sue|suing|sued) (them|him|her|the company)|"
    r"(press|pursue|considering) (charges|legal action)|"
    r"(my |the )?(legal )?(rights|options|recourse)|"
    r"is (this|it|that) (legal|illegal|lawful|unlawful)|"
    r"what (are|were) my (legal )?(rights|options)|"
    r"how do i (sue|file|press charges|take legal)"
    r")\b",
    re.I)

_FINANCIAL_STRONG = {
    "tax", "mortgage", "401k", "ira", "roth", "dividend",
    "capital gains", "depreciation", "portfolio", "hedge fund",
    "mutual fund", "etf", "bond", "bonds", "equity", "derivative",
    "futures", "forex", "cryptocurrency", "bitcoin", "ethereum",
    "annuity", "pension", "fiduciary", "brokerage", "securities",
    "index fund", "index funds", "stock market", "shares", "interest rate",
    "compound interest", "net worth", "asset allocation", "retirement fund",
}

_FINANCIAL_WEAK = {
    "account", "bank", "pay", "payment", "bill", "charge", "refund",
    "credit", "fee", "money", "cash", "cost", "price", "budget",
    "expense", "transfer", "deposit", "withdraw", "balance", "invoice",
}

_FINANCIAL_ADVICE_VERBS = {
    "invest", "advise", "recommend", "allocate", "hedge",
    "speculate", "trade", "rebalance", "diversify", "claim", "deduct",
}

# Code — strong unambiguous terms only for tagging
# Language names — unambiguous alone: nobody says "my python" meaning a snake
CODE_STRONG = {
    "python", "javascript", "typescript", "kotlin", "swift", "golang",
    "haskell", "scala", "php", "bash", "shell", "yaml",
    "kubernetes", "webpack", "npm", "pip", "conda",
    "pytorch", "tensorflow", "pandas", "numpy",
    "graphql", "oauth", "jwt", "webhook",
    "virtualenv", "dockerfile", "terraform",
    "lua", "sql", "ruby", "rust", "golang",
}

# Framework and library names — unambiguous in coding context
CODE_FRAMEWORKS = {
    "react", "vue", "angular", "svelte", "nextjs", "nuxtjs",
    "django", "flask", "fastapi", "nodejs", "node.js",
    "spring", "laravel", "rails", "aspnet", "dotnet",
    "redux", "mobx", "tailwind", "bootstrap",
    "pytest", "jest", "mocha", "cypress",
    "mongodb", "postgresql", "sqlite", "redis", "elasticsearch",
    "docker", "github actions", "ci/cd", "gitlab",
}

# Task verbs — signal A (need at least one more signal)
_CODE_TASK_VERBS = re.compile(
    r"\b(write|build|create|implement|fix|debug|refactor|optimise|optimize|"
     r"generate|convert|parse|deploy|run|execute|compile|install|configure|"
     r"set up|set up|migrate|update|upgrade|test|lint|format|render)\b",
    re.I)

# Code artefacts — signal B (need at least one more signal)
_CODE_ARTEFACTS = re.compile(
    r"\b(function|method|class|script|code|program|app|application|"
     r"algorithm|query|api|endpoint|component|module|loop|array|list|"
     r"dict|dictionary|database|schema|model|interface|library|package|"
     r"repository|repo|branch|commit|pipeline|workflow|container|"
     r"variable|parameter|argument|object|instance|thread|process)\b",
    re.I)

# Specific error types — unambiguous alone
_CODE_ERRORS = re.compile(
    r"\b(IndentationError|SyntaxError|TypeError|NameError|ValueError|"
     r"AttributeError|ImportError|KeyError|IndexError|RuntimeError|"
     r"NullPointerException|NullReferenceException|SegmentationFault|"
     r"OutOfMemoryError|StackOverflowError|undefined is not|"
     r"cannot read prop|is not a function|unexpected token)\b",
    re.I)

CODE_PHRASES = {
    "syntax error", "null pointer", "stack overflow", "time complexity",
    "pull request", "code review", "type error", "index out of bounds",
    "memory leak", "race condition", "deadlock", "object oriented",
    "functional programming", "version control", "dependency injection",
}

# ─────────────────────────────────────────────
# POLITENESS MARKERS
# Used by SLM prompt builder to preserve tone signal.
# ─────────────────────────────────────────────

POLITENESS_MARKERS = {
    "sorry", "apologise", "apologize", "apologies", "forgive",
    "embarrassed", "ashamed", "bother you", "bothering you",
    "hate to ask", "feel bad", "incredibly sorry", "so sorry",
    "really sorry", "truly sorry", "hate to trouble", "please forgive",
}


def _detect_domain(text: str) -> Optional[str]:
    """
    Detect domain of prompt for tagging.
    Returns domain string or None.
    Priority: medical > legal > financial > code > None
    """
    text_lower = text.lower()

    if _word_boundary_match(text, MEDICAL_KEYWORDS):
        return "MEDICAL"

    if _phrase_match(text, LEGAL_KEYWORDS) or _word_boundary_match(text, LEGAL_KEYWORDS):
        return "LEGAL"

    if _word_boundary_match(text, _FINANCIAL_STRONG):
        return "FINANCIAL"
    has_weak_fin = _word_boundary_match(text, _FINANCIAL_WEAK)
    has_advice   = _word_boundary_match(text, _FINANCIAL_ADVICE_VERBS)
    if has_weak_fin and has_advice:
        return "FINANCIAL"

    if _word_boundary_match(text, CODE_STRONG) or _phrase_match(text, CODE_PHRASES):
        return "CODE"

    return None


# ─────────────────────────────────────────────
# MEDICAL TIER 1 DETECTION — TWO LAYERS
# Used only by tier1_check for passthrough decisions.
# Deliberately separate from MEDICAL_KEYWORDS which is used
# by _detect_domain() for SLM tagging — do not merge these.
# ─────────────────────────────────────────────

# Layer A — unambiguous clinical terms
# These NEVER appear in support, e-commerce, or conversational prompts.
# Expanding this set is safe. Shrinking it risks missing medical prompts.
_MEDICAL_TIER1_HARD = {
    # Clinical roles — unambiguous
    "physician", "pediatrician", "paediatrician", "psychiatrist",
    "psychologist", "surgeon", "cardiologist", "oncologist",
    "radiologist", "anaesthetist", "anesthesiologist", "dermatologist",
    "neurologist", "gynaecologist", "gynecologist", "obstetrician",
    # Obstetric / perinatal — never in support
    "primigravida", "multigravida", "gravida", "gestation",
    "prenatal", "antenatal", "postnatal", "postnatal", "perinatal",
    "neonatal", "trimester",
    # Specific drugs — unambiguous in medical context
    "ibuprofen", "paracetamol", "acetaminophen", "metformin",
    "levothyroxine", "warfarin", "sertraline", "omeprazole",
    "amoxicillin", "amlodipine", "atorvastatin", "lisinopril",
    "metoprolol", "furosemide", "prednisone", "prednisolone",
    # Procedures and diagnostics
    "biopsy", "chemotherapy", "radiotherapy", "mri",
    "pathology", "prognosis", "palliative", "autopsy",
    "laparoscopy", "endoscopy", "colonoscopy", "mammography",
    # Conditions — unambiguous (never colloquial)
    "hypertension", "diabetes", "insulin", "seizure",
    "menorrhagia", "diplopia", "anaphylaxis",
    "haemorrhage", "hemorrhage", "thrombosis", "embolism",
    "malignant", "metastasis", "metastatic",
    "tumour", "tumor", "carcinoma", "lymphoma", "leukaemia",
    "leukemia", "melanoma",
    # Death / forensic — clinical context
    "died suddenly", "cause of death", "post-mortem",
    "sudden cardiac death",
    # Lab values and clinical markers
    "mmhg", "mg/dl", "mmol/l", "bpm", "g/dl",
    "haemoglobin", "hemoglobin", "platelet", "creatinine",
    "cholesterol", "triglycerides", "glucose",
}

# Layer B — clinical structure regex
# Catches USMLE vignettes and clinical presentation language.
# Pattern: age descriptor, presentation verbs, MCQ question format.
# Zero false positive risk on support/e-commerce prompts.
_CLINICAL_STRUCTURE_RE = re.compile(
    r'\b\d{1,3}[\s-]?year[\s-]?old\b'           # age: "23-year-old"
    r'|\bpresents?\s+(?:with|to)\b'              # "presents with"
    r'|\bcomes?\s+to\s+(?:the\s+)?'
    r'(?:physician|doctor|hospital|emergency|clinic)\b'
    r'|\bis\s+brought\s+to\b'                    # "is brought to"
    r'|\bweeks?\s+(?:of\s+)?gestation\b'         # "22 weeks gestation"
    r'|\bmost\s+likely\s+(?:diagnosis|cause)\b'  # USMLE MCQ
    r'|\bbest\s+next\s+step\b'                   # USMLE MCQ
    r'|\bwhich\s+of\s+the\s+following\b'         # USMLE MCQ
    r'|\bchief\s+complaint\b'                    # clinical note
    r'|\bpresenting\s+complaint\b'               # clinical note
    r'|\bhistory\s+of\s+present(?:ing)?\s+illness\b',  # SOAP note
    re.IGNORECASE
)


def tier1_check(text: str) -> tuple[bool, Optional[str]]:
    """
    Minimal passthrough gate — fires on:
    1. Very short prompts (net saving impossible regardless)
    2. Hard safety phrases (exact wording is safety-critical)
    3. MEDICAL — two layers:
       Layer A: unambiguous clinical keywords (never appear in support)
       Layer B: clinical structure regex (USMLE/vignette patterns)
       Justified: hypothesis test mean similarity = 0.615 on medical
       Prompts that slip through (general health questions) go to SLM
       which tags [MEDICAL] and compresses carefully — acceptable.
    4. LEGAL — keyword/phrase match, legal precision non-negotiable

    FINANCIAL and CODE: tag-only, not blocked.
      CODE: mean saving 48.9t, quality unaffected
      FINANCIAL: dangerous edge cases covered by HARD_PASSTHROUGH_PHRASES

    Returns (passthrough: bool, reason: str | None)
    """
    tokens = text.split()

    # ── Guard 1: short prompt ────────────────────────────────────────
    if len(tokens) <= SHORT_PROMPT_WORD_LIMIT:
        return True, "short_prompt"

    # ── Guard 2: hard safety phrases ─────────────────────────────────
    if _phrase_match(text, HARD_PASSTHROUGH_PHRASES):
        return True, "safety_critical"

    # ── Guard 3: medical — Layer A (unambiguous clinical keywords) ──
    # These terms never appear in support/conversational prompts.
    if _word_boundary_match(text, _MEDICAL_TIER1_HARD):
        return True, "domain_medical"

    # ── Guard 3: medical — Layer B (clinical structure regex) ─────────
    if _CLINICAL_STRUCTURE_RE.search(text):
        return True, "domain_medical"

    # ── Guard 3: medical — Layer C (conversational medical phrases) ───
    # Real corpus medical prompts use natural language not clinical terms.
    if re.search(
            r'\b(my doctor|my physician|my specialist|my consultant|'
             r'my medication|my prescription|my treatment|my diagnosis|'
             r'my symptoms|my condition|my surgery|my operation|'
             r'prescribed (me|by)|taking medication|side effects? of|'
             r'drug interaction|drug dosage|safe to take|'
             r'is it safe to take|medical advice|health condition|'
             r'\d+ years? old (patient|woman|man|male|female|child|boy|girl)|'
             r'presents (to|with)|chief complaint|past medical history|'
             r'vital signs|blood pressure|heart rate|'
             r'\d+ mg (of|per|daily|twice)|dose of|dosage of)\b',
            text, re.I):
        return True, "domain_medical"

    # ── Guard 4: legal ─────────────────────────────────────────────────────────────
    # Tier 1 — unambiguous legal procedure terms fire alone.
    # Tier 2 — ambiguous legal words need legal-seeking framing.
    # sue/suing checked via word boundary to avoid "issue" substring.
    if (_phrase_match(text, LEGAL_KEYWORDS) or
            _word_boundary_match(text, LEGAL_KEYWORDS)):
        return True, "domain_legal"
    if re.search(r'\bsue\b|\bsuing\b', text, re.I):
        return True, "domain_legal"
    if (_phrase_match(text, _LEGAL_TIER2) or
            _word_boundary_match(text, _LEGAL_TIER2)):
        if _LEGAL_SEEKING.search(text):
            return True, "domain_legal"

    # ── Guard 5: code — two-signal scorer ───────────────────────
    # Signal A alone (language/framework name) → always code
    # Signal B alone (specific error type) → always code
    # Signal C (task verb + artefact + technical qualifier) → coding ask
    # Avoids false positives from "function"/"method"/"error" in plain English

    # A: unambiguous language or framework name
    if _word_boundary_match(text, CODE_STRONG):
        return True, "domain_code"
    if any(fw in text.lower() for fw in CODE_FRAMEWORKS):
        return True, "domain_code"

    # B: specific error types — unambiguous
    if _CODE_ERRORS.search(text):
        return True, "domain_code"

    # C: technical code phrases — unambiguous combinations
    if _phrase_match(text, CODE_PHRASES):
        return True, "domain_code"

    # D: actual code syntax present
    if re.search(
            r'(def [a-z_]+[(]|class [A-Z]\w+[:(]|'
             r'import [a-z]|from [a-z]+ import|'
             r'function\s*\w*\s*[(]|'
             r'#include\s*<)',
            text):
        return True, "domain_code"

    # E check removed — industry standard:
    # Natural language coding task detection via artefact words
    # is unreliable. Words like "process", "application", "program",
    # "model", "interface", "pipeline" appear equally in financial
    # complaints, legal documents, and general English.
    # Signals A-D (language name, error type, code phrases, syntax)
    # are sufficient and have zero false positives.

    return False, None


def tier1b_check(compressed: str) -> tuple[bool, Optional[str]]:
    """
    Fallback safety re-check on SLM compressed output.
    Only checks hard safety phrases — NOT word count.
    A short compressed output is a success, not a failure.
    The original prompt already passed the word count gate at Tier 1.
    """
    if _phrase_match(compressed, HARD_PASSTHROUGH_PHRASES):
        return True, "safety_critical"
    return False, None

# ─────────────────────────────────────────────
# AUX URGENCY ESCALATORS
# ─────────────────────────────────────────────

AUX_URGENCY_ESCALATORS = {
    "crying", "screaming", "emergency", "accident", "deadline",
    "dying", "urgent", "asap", "right now", "immediately",
    "meeting in", "presentation in", "flight in", "interview in",
    "surgery", "hospital", "ambulance", "bleeding", "burning", "fire",
}


# ─────────────────────────────────────────────
# FEW-SHOT EXAMPLES FOR SLM PROMPT
# Cover: support/tracking, domain-tagged medical, domain-tagged code,
#        explanatory/idiom, multi-question, casual
# ─────────────────────────────────────────────

FEW_SHOT_EXAMPLES = """
EXAMPLE 1 — Support / tracking (short, anxious, high urgency)
INPUT: "Hi I'm so sorry to bother you, I know you must be really busy, but I ordered something yesterday and I can't find the tracking number anywhere in my emails and my kids are crying in the background and I really need to know where this package is. The order number is TRK-00492. I'm incredibly stressed."
OUTPUT:
{
  "compressed_prompt": "Can't find tracking number for order TRK-00492 placed yesterday. Checked email, not found.",
  "intent": "retrieval",
  "tone": "anxious/apologetic",
  "urgency": "high",
  "aux": ["kids crying", "order placed yesterday", "checked email already"],
  "confidence": 0.91
}

EXAMPLE 2 — Medical domain (tagged)
INPUT: "[MEDICAL] I've been taking ibuprofen 400mg twice a day for about two weeks for back pain and I'm wondering if that's too long and whether I should be worried about side effects given I also have mild hypertension."
OUTPUT:
{
  "compressed_prompt": "Is 400mg ibuprofen twice daily for 2 weeks too long, and are side effects a concern with mild hypertension?",
  "intent": "information",
  "tone": "neutral",
  "urgency": "medium",
  "aux": ["back pain context", "concurrent hypertension", "2-week duration"],
  "confidence": 0.87
}

EXAMPLE 3 — Code domain (tagged)
INPUT: "[CODE] I'm getting a null pointer exception in my Java checkout service on line 42 every time a user tries to apply a discount code. The stack trace shows it's in the CartService.applyDiscount method. I've checked that the discount object isn't null but it keeps throwing."
OUTPUT:
{
  "compressed_prompt": "NullPointerException in Java CartService.applyDiscount (line 42) when applying discount codes. Discount object confirmed non-null.",
  "intent": "action",
  "tone": "neutral",
  "urgency": "medium",
  "aux": ["checkout service", "triggered by discount code application", "discount object already checked"],
  "confidence": 0.92
}

EXAMPLE 4 — Multi-question
INPUT: "Two things: first, what's the difference between supervised and unsupervised learning? And based on that, which one would be better for clustering customer purchase data where I don't have predefined labels?"
OUTPUT:
{
  "compressed_prompt": "What is the difference between supervised and unsupervised learning? Given that, which suits clustering unlabelled customer purchase data?",
  "intent": "information",
  "tone": "neutral",
  "urgency": "low",
  "aux": ["questions are linked", "use case: customer purchase clustering", "no predefined labels"],
  "confidence": 0.90
}

EXAMPLE 5 — Casual / conversational
INPUT: "Hey, random question — do you think remote work is actually better for productivity or is it just something people say? I've been going back and forth on this."
OUTPUT:
{
  "compressed_prompt": "Is remote work genuinely better for productivity, or is it overstated?",
  "intent": "information",
  "tone": "casual",
  "urgency": "low",
  "aux": ["user undecided", "looking for evidence-based view"],
  "confidence": 0.88
}

EXAMPLE 6 — Verbose service complaint (long, 80+ words) — FAITHFUL EXTRACTION REQUIRED
INPUT: "Hi there, I'm really sorry to bother you with this and I know you must deal with so many queries every day, but I placed an order about ten days ago and it still hasn't arrived. I've checked the tracking and it's been stuck on the same status for a week now. My daughter was really looking forward to it and I'm getting quite worried. Could you please look into this for me and let me know what's happening? I would really appreciate your help with this."
OUTPUT:
{
  "compressed_prompt": "Order placed ten days ago hasn't arrived. Tracking stuck for one week. Please look into it.",
  "intent": "retrieval",
  "tone": "anxious/apologetic",
  "urgency": "high",
  "aux": ["daughter looking forward to it", "tracking stuck one week", "order placed ten days ago"],
  "confidence": 0.90
}

EXAMPLE 7 — Verbose insurance complaint — COPY FACTS EXACTLY, DO NOT SUBSTITUTE
INPUT: "Hi there. I'm reaching out because I'm getting really frustrated with an insurance claim that's been ongoing for two months. I had a home contents claim after a burglary and it's been assessed and approved but I still haven't received the payment. I've been told multiple times it's been processed but nothing has appeared in my bank account. I'm really stressed as some of the stolen items were things I need for work. Could you please investigate where the payment is?"
OUTPUT:
{
  "compressed_prompt": "Approved home contents insurance claim after burglary hasn't paid out after two months. Told multiple times it was processed but nothing in bank account. Where is the payment?",
  "intent": "retrieval",
  "tone": "anxious",
  "urgency": "high",
  "aux": ["home contents claim", "post-burglary", "claim approved but not paid out", "stolen items needed for work"],
  "confidence": 0.93
}

EXAMPLE 8 — Verbose landlord complaint — PRESERVE SPECIFIC DETAILS
INPUT: "I'm really worried about my housing situation and I need some guidance. My landlord has been ignoring my repair requests for two months. There's a broken window in my living room that I reported at the beginning of winter. I've called, texted, and emailed and I just keep getting told someone will come but nobody ever does. It's freezing and I'm also concerned about security. Can you help me write a formal letter to my landlord demanding the repair be carried out?"
OUTPUT:
{
  "compressed_prompt": "Landlord ignoring broken window repair for two months despite calls, texts, and emails. Affecting warmth and security. Help writing a formal letter demanding the repair.",
  "intent": "action",
  "tone": "anxious",
  "urgency": "high",
  "aux": ["broken window living room", "reported start of winter", "called texted emailed all tried", "security and warmth affected"],
  "confidence": 0.91
}

===STRICT FIDELITY RULES FOR LONG PROMPTS===
When the input is longer than 40 words, apply these rules without exception:
RULE A — COPY TIMEFRAMES EXACTLY: "ten days" stays "ten days". Never round up or down.
RULE B — DO NOT INVENT CONTACT HISTORY: Only include "contacted support" if the user explicitly said they did. If not stated, do not add it.
RULE C — PRESERVE THE SCENARIO: A burglary claim stays a burglary claim. A boiler repair stays a boiler repair. Never substitute a different scenario.
RULE D — DO NOT ELABORATE: If the user mentioned one issue, compress to one issue. Do not add related issues they did not mention.
RULE E — UNCERTAINTY → OMIT: If you are not certain a fact appears in the original, leave it out of compressed_prompt and aux entirely.
""".strip()

# ─────────────────────────────────────────────
# SLM SYSTEM PROMPT
# ─────────────────────────────────────────────

SLM_SYSTEM_PROMPT = """Read this message. Rewrite it as the same person speaking, but more briefly. Keep their voice and first-person perspective.

Focus on: what is the main problem and what do they need. Start with the most important fact.

Rules:
1. Write in first person — I, my, me
2. Start with the core problem, not the backstory
3. Keep ALL named entities exactly: company names, person names, amounts, dates — never generalise
4. Keep direct quotes and specific facts verbatim
5. Remove apologies, emotional filler, repetition
6. Maximum 150 tokens — write as much as needed but no more
7. Do not add anything not in the original
8. Always include the final question or request

Message: {INPUT}

Essential request:"""

# ─────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────

@dataclass
class DistillResult:
    original_prompt:    str
    compressed_prompt:  str
    intent:             str
    tone:               str
    urgency:            str
    aux:                list[str]
    passthrough:        bool
    passthrough_reason: Optional[str]
    confidence:         float
    latency_ms:         float
    tier:               str             # "tier1", "tier1b", "slm"
    schema_valid:       bool = True
    token_saving:       int  = 0
    domain:             Optional[str] = None
    complexity:         Optional[object] = None  # ComplexityResult
    hfg_aux:            list = None  # High Fidelity Guard extracted phrases

    def __post_init__(self):
        if self.hfg_aux is None:
            self.hfg_aux = []

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if self.passthrough:
            return (
                f"[PASSTHROUGH — {self.passthrough_reason}]\n"
                f"Original forwarded to frontier LLM unchanged.\n"
                f"Tier: {self.tier} | Latency: {self.latency_ms:.1f}ms"
            )
        cx = self.complexity
        cx_str = f" | structural={cx.structural_score:.2f}" if cx and not self.passthrough else ""
        domain_str = f" | domain={self.domain}" if self.domain else ""
        hfg_str = f"\nHFG Aux    : {self.hfg_aux}" if self.hfg_aux else ""
        return (
            f"[DISTILLED — {self.tier} | {self.latency_ms:.1f}ms | "
            f"confidence={self.confidence:.2f} | "
            f"token_saving={self.token_saving:+d}{domain_str}{cx_str}]\n"
            f"Compressed : {self.compressed_prompt}\n"
            f"Intent     : {self.intent}\n"
            f"Tone       : {self.tone}\n"
            f"Urgency    : {self.urgency}\n"
            f"Aux        : {self.aux}"
            f"{hfg_str}"
        )


# ─────────────────────────────────────────────
# TIER 1 — RULE-BASED GUARDS
# ─────────────────────────────────────────────

def _word_boundary_match(text: str, word_set: set[str]) -> bool:
    """Match whole words only — prevents 'import' matching 'important'."""
    text_lower = text.lower()
    for word in word_set:
        pattern = r'\b' + re.escape(word) + r'\b'
        if re.search(pattern, text_lower):
            return True
    return False


def _phrase_match(text: str, phrase_set: set[str]) -> bool:
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in phrase_set)


def _extract_identifiers(text: str) -> list[str]:
    """Extract reference numbers, URLs, order IDs."""
    patterns = [
        r'\b[A-Z]{2,6}-\d{3,10}\b',          # TRK-00492, ORD-123456
        r'\b\d{6,15}\b',                        # plain numeric order IDs
        r'https?://[^\s]+',                     # URLs
        r'\b[A-Z0-9]{8,20}\b',                  # alphanumeric reference codes
    ]
    identifiers = []
    for pattern in patterns:
        identifiers.extend(re.findall(pattern, text))
    return list(set(identifiers))



# ─────────────────────────────────────────────
# COMPLEXITY SCORER
# Rule-based, ~0ms, no model dependency.
# Runs after Tier 1 passes. Measures how much of
# the prompt is load-bearing vs compressible.
# Outputs a ComplexityResult that:
#   - triggers passthrough if structural density
#     is too high to compress safely
#   - sets a compression_ceiling for the SLM prompt
#   - adds a complexity profile to DistillResult
# ─────────────────────────────────────────────

# Social scaffolding — normalise apostrophes before matching
# Semantic anchors — load-bearing, must survive compression


# ─────────────────────────────────────────────
# HIGH FIDELITY GUARD (HFG)
# ~0ms, rule-based, no model dependency.
#
# Purpose: before the SLM runs, extract high-entropy phrases
# that must NEVER be dropped from aux context regardless of
# what the SLM decides to compress away.
#
# These are phrases where:
#   - The specific wording carries disproportionate emotional weight
#   - Loss of the phrase changes how the frontier LLM should respond
#   - The SLM might treat them as social noise and remove them
#
# HFG phrases are injected into aux BEFORE the SLM call.
# The SLM system prompt instructs it to preserve aux context —
# so even if the SLM would drop "kids crying" from the compressed
# prompt, it will still appear in aux from HFG injection.
#
# Three guard layers:
#   1. Life event markers    — events that reframe everything
#   2. Emotional anchors     — high-entropy specific phrases
#   3. Dependency markers    — who is affected (person specificity)
#   4. Crisis indicators     — slow-burn high-stakes situations
# ─────────────────────────────────────────────

# ── Layer 1: Life event markers ──────────────────────────────────────
# Specific life events that change the urgency and nature of the response
_HFG_LIFE_EVENTS = [
    # Celebrations / milestones
    (r'\b(wedding|our wedding|my wedding)\b',              'life_event: wedding'),
    (r'\b(birthday|her birthday|his birthday|my daughter.s birthday|'
     r'my son.s birthday|my wife.s birthday|my husband.s birthday)\b',
                                                           'life_event: birthday'),
    (r'\b(anniversary|our anniversary|wedding anniversary)\b',
                                                           'life_event: anniversary'),
    (r'\b(graduation|graduating|graduation day|graduation ceremony)\b',
                                                           'life_event: graduation'),
    (r'\b(baby shower|gender reveal|baby is due|due date|'
     r'expecting a baby|pregnant|newborn|just had a baby)\b',
                                                           'life_event: new_baby'),
    (r'\b(christmas|christmas day|christmas morning|christmas eve|'
     r'christmas present|christmas gift)\b',               'life_event: christmas'),
    (r'\b(holiday|holiday gift|holiday present|going on holiday|'
     r'holiday tomorrow|holiday this week)\b',             'life_event: holiday'),
    # Loss / grief
    (r'\b(funeral|someone passed|someone died|death in the family|'
     r'bereavement|my (mother|father|parent|husband|wife|child|son|daughter) '
     r'(died|passed|is dying|has died))\b',               'life_event: bereavement'),
    (r'\b(diagnosed with|terminal|hospice|end of life|palliative)\b',
                                                           'life_event: serious_diagnosis'),
    # Major transitions
    (r'\b(moving house|moving home|just moved|new home|new house|'
     r'starting (a new job|new job|university|college|school))\b',
                                                           'life_event: major_transition'),
    (r'\b(just got (married|engaged|divorced)|getting married|'
     r'getting divorced|separation)\b',                   'life_event: relationship_change'),
]

# ── Layer 2: Emotional anchors ────────────────────────────────────────
# High-entropy emotional phrases — specific enough to be meaningful
_HFG_EMOTIONAL_ANCHORS = [
    # Children distress
    (r'\b(kids? (is|are|has been|have been|was|were) (crying|upset|'
     r'screaming|sick|ill|in hospital))\b',               'emotional: child_distress'),
    (r'\b(my (daughter|son|child|baby|toddler|kid) (is|has been|'
     r'was) (crying|upset|sick|ill|scared|frightened))\b',
                                                           'emotional: child_distress'),
    (r'\b(kids? crying|children crying|baby crying|baby won.t stop)\b',
                                                           'emotional: child_distress'),
    # Personal distress
    (r'\b(incredibly stressed|extremely stressed|very stressed|'
     r'so stressed|beyond stressed)\b',                   'emotional: high_stress'),
    (r'\b(in tears|crying (myself|over this)|been crying|'
     r'sobbing|breaking down)\b',                         'emotional: crying'),
    (r'\b(panicking|having a panic|panic attack|anxiety attack|'
     r'can.t breathe|hyperventilating)\b',               'emotional: panic'),
    (r'\b(terrified|absolutely terrified|frightened|petrified|'
     r'scared (to death|out of my mind))\b',              'emotional: fear'),
    (r'\b(devastated|absolutely devastated|heartbroken|'
     r'completely heartbroken)\b',                        'emotional: devastated'),
    # Embarrassment / vulnerability
    (r'\b(so embarrassed|really embarrassed|mortified|'
     r'feel (terrible|awful|horrible) about)\b',          'emotional: embarrassed'),
    (r'\b(vulnerable|feeling vulnerable|at my most vulnerable)\b',
                                                           'emotional: vulnerable'),
]

# ── Layer 3: Dependency markers ───────────────────────────────────────
# Who is affected — person specificity changes response register
_HFG_DEPENDENCY_MARKERS = [
    # Children and ages
    (r'\b(my (daughter|son|child|baby|toddler|infant|newborn))\b',
                                                           'dependency: child'),
    (r'\b(\d+[\s-]?(year|yr)[\s-]?old (daughter|son|child|girl|boy))\b',
                                                           'dependency: child_with_age'),
    (r'\b(my kids?|my children|my little (one|ones|girl|boy))\b',
                                                           'dependency: children'),
    # Elderly / vulnerable adults
    (r'\b(my (elderly|aging|aged|old) (mother|father|parent|'
     r'mum|mam|dad|grandma|grandpa|grandmother|grandfather))\b',
                                                           'dependency: elderly_relative'),
    (r'\b(my (mother|father|mum|dad|parent) (is|has been|was) '
     r'(ill|sick|in hospital|unwell|frail|vulnerable))\b',
                                                           'dependency: ill_parent'),
    # Disability / care
    (r'\b(my (disabled|special needs|autistic|deaf|blind|'
     r'wheelchair) (child|son|daughter|brother|sister|partner|husband|wife))\b',
                                                           'dependency: disabled_dependent'),
    (r'\b(carer|full.time carer|caring for|I care for)\b',
                                                           'dependency: carer'),
    # Pregnancy
    (r'\b(I am pregnant|I.m pregnant|I.m expecting|'
     r'\d+ weeks (pregnant|gestation))\b',               'dependency: pregnant'),
]

# ── Layer 4: Crisis indicators ────────────────────────────────────────
# Slow-burn but high-stakes — not immediate safety emergencies
# but situations where the person's life is significantly affected
_HFG_CRISIS_INDICATORS = [
    # Financial crisis
    (r'\b(facing eviction|eviction notice|being evicted|'
     r'losing my home|about to lose my home)\b',          'crisis: eviction'),
    (r'\b(repossession|car is being repossessed|house repossessed)\b',
                                                           'crisis: repossession'),
    (r'\b(bailiff|debt collector|final demand|county court '
     r'judgment|CCJ)\b',                                  'crisis: debt_enforcement'),
    (r'\b(can.t afford|cannot afford|no money left|'
     r'run out of money|financially desperate)\b',        'crisis: financial_hardship'),
    # Employment crisis
    (r'\b(losing my job|lost my job|been made redundant|'
     r'facing redundancy|laid off)\b',                    'crisis: job_loss'),
    (r'\b(last paycheck|final paycheck|unpaid wages|'
     r'not been paid|employer hasn.t paid)\b',            'crisis: unpaid'),
    # Time-critical situations
    (r'\b(flight (is|in) (tomorrow|tonight|this morning|'
     r'in \d+ hours?))\b',                               'crisis: imminent_travel'),
    (r'\b(surgery (is|in) (tomorrow|tonight|this morning|'
     r'in \d+ (hours?|days?)))\b',                       'crisis: imminent_surgery'),
    (r'\b(interview (is|in) (tomorrow|tonight|this morning|'
     r'in \d+ (hours?|minutes?)))\b',                    'crisis: imminent_interview'),
    (r'\b(court (date|hearing|appearance) (is|in) '
     r'(tomorrow|tonight|this morning))\b',               'crisis: imminent_court'),
    (r'\b(deadline (is|in) (tomorrow|tonight|today|'
     r'in \d+ (hours?|minutes?)))\b',                    'crisis: imminent_deadline'),
    # Health crisis (non-emergency but serious)
    (r'\b(just been diagnosed|recently diagnosed|'
     r'diagnosed (with|yesterday|today|last week))\b',   'crisis: new_diagnosis'),
    (r'\b(medication (ran out|running out|I.ve run out|'
     r'I have run out))\b',                               'crisis: medication_shortage'),
]


def extract_hfg_phrases(text: str) -> list[str]:
    """
    High Fidelity Guard — extract guaranteed aux context phrases.
    Returns a list of HFG labels found in the text.
    These are merged into aux BEFORE the SLM call.

    ~0ms, no model, pure regex.

    The labels follow the format "category: specific_value" so the
    frontier LLM receives structured context even if the SLM
    compresses the emotional content away from the main prompt.
    """
    text_lower = text.lower()
    found = []
    seen_labels = set()

    all_guards = (
        _HFG_LIFE_EVENTS +
        _HFG_EMOTIONAL_ANCHORS +
        _HFG_DEPENDENCY_MARKERS +
        _HFG_CRISIS_INDICATORS
    )

    for pattern, label in all_guards:
        if label in seen_labels:
            continue
        if re.search(pattern, text_lower, re.IGNORECASE):
            found.append(label)
            seen_labels.add(label)

    return found


# ─────────────────────────────────────────────
# SLM INFERENCE
# ─────────────────────────────────────────────

_llm = None  # module-level singleton — load once


def load_model(model_path: Optional[str] = None) -> None:
    """
    Load Gemma-2-2B-IT Q4_K_M.
    Research runtime: llama-cpp-python on CPU.
    Production runtime: replace with NPU SDK call.
    Call once at startup. Safe to call multiple times (no-op if loaded).
    
    model_path: override default path (useful for Colab where model
                is downloaded to a specific location).
    """
    global _llm
    if _llm is not None:
        return

    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError(
            "llama-cpp-python not installed. "
            "Run: pip install llama-cpp-python"
        )

    path = model_path or str(MODEL_PATH)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run download_model() first or pass the correct path."
        )

    print(f"Loading SLM from {path} ...")
    t0 = time.time()
    _llm = Llama(
        model_path=path,
        n_ctx=SLM_CONTEXT_LENGTH,
        n_threads=SLM_N_THREADS,
        n_gpu_layers=0,         # CPU only
        verbose=False,
    )
    print(f"SLM loaded in {(time.time()-t0)*1000:.0f}ms")


def download_model(dest_dir: str = "./models") -> str:
    """
    Download Qwen2.5-1.5B-Instruct Q4_K_M from HuggingFace Hub.
    Returns path to downloaded file.
    Requires: pip install huggingface-hub
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub not installed. "
            "Run: pip install huggingface-hub"
        )

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO} ...")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        local_dir=dest_dir,
    )
    print(f"Downloaded to {path}")
    return path


def _build_slm_prompt(text: str) -> str:
    """
    Build Gemma compression prompt — v0.7 instruction.
    Summary-first, first-person, 8 rules.
    Gemma-2-2B chat template: <start_of_turn>user / model.

    Production runtime note:
    Research runtime: llama-cpp-python on CPU.
    Production runtime: replace _run_slm() body with NPU SDK call.
    Model weights and this prompt format remain identical.
    """
    return (
        f"<start_of_turn>user\n"
        f"Read this message. Rewrite it as the same person speaking, "
        f"but more briefly. Keep their voice and first-person perspective.\n\n"
        f"Focus on: what is the main problem and what do they need. "
        f"Start with the most important fact.\n\n"
        f"Rules:\n"
        f"1. Write in first person — I, my, me\n"
        f"2. Start with the core problem, not the backstory\n"
        f"3. Keep ALL named entities exactly: company names, person names, "
        f"amounts, dates — never generalise\n"
        f"4. Keep direct quotes and specific facts verbatim\n"
        f"5. Remove apologies, emotional filler, repetition\n"
        f"6. Maximum 150 tokens — write as much as needed but no more\n"
        f"7. Do not add anything not in the original\n"
        f"8. Always include the final question or request\n\n"
        f'Message: "{text}"\n\n'
        f"Brief first-person version:<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )



def _validate_envelope(env: dict) -> tuple[dict, bool]:
    """
    Validate required fields. Fill missing fields with safe defaults.
    Returns (envelope, is_valid).
    """
    required = ["compressed_prompt", "intent", "tone", "urgency",
                "aux", "confidence"]
    valid = True
    for field in required:
        if field not in env:
            valid = False
            if field == "compressed_prompt": env[field] = ""
            elif field == "aux":             env[field] = []
            elif field == "confidence":      env[field] = 0.0
            else:                            env[field] = "unknown"
    if not isinstance(env.get("aux"), list):
        env["aux"] = []
    env["confidence"] = float(env.get("confidence", 0.0))
    return env, valid


def _fidelity_check(env: dict, original: str) -> dict:
    """
    Post-generation fidelity check.
    Detects if compressed_prompt contains words not in original.
    Lowers confidence if hallucinated words detected.
    """
    compressed = env.get("compressed_prompt", "")
    if not compressed:
        return env
    orig_words = set(re.sub(r"[^a-z0-9\s]", " ", original.lower()).split())
    comp_words = set(re.sub(r"[^a-z0-9\s]", " ", compressed.lower()).split())
    STOP = {"the","a","an","and","or","but","in","on","at","to","for",
            "of","with","by","from","that","which","is","are","was","were",
            "be","have","has","had","will","would","should","could","can",
            "not","i","my","me","we","they","them","their","this","it",
            "its","as","so","do","did","get","got","been","also","just"}
    novel = {w for w in comp_words if len(w) >= 4 and w not in STOP
             and w not in orig_words}
    if len(novel) >= 3:
        env["confidence"] = max(0.0, env.get("confidence", 0.85) - 0.15)
    return env


def _classify_rule_based(compressed: str, original: str) -> dict:
    """
    Rule-based classification of tone, urgency, intent, aux.
    Reads from the ORIGINAL prompt. SLM provides compressed_prompt only.
    All other fields always rebuilt rule-based — never from SLM output.
    """
    o = original.lower()
    if re.search(
            r"sorry|apologis|bother you|forgive|hope you can|please help|"
            r"appreciate|thank you|i know you|don.t want to be|"
            r"not sure what to do|at a loss|genuinely", o):
        tone = "anxious/apologetic"
    elif re.search(
            r"frustrated|angry|furious|unacceptable|appalling|"
            r"disgraceful|outrageous|let down", o):
        tone = "negative"
    elif re.search(r"\bhey\b|quick question|just wondering", o):
        tone = "casual"
    else:
        tone = "neutral"

    if re.search(
            r"urgent|asap|immediately|today|tonight|tomorrow|"
            r"birthday|deadline|emergency|baby|young child|"
            r"hospital|freezing|no heating|cannot work|locked out|"
            r"affecting.{0,20}job|for work|work from home|"
            r"worried|concerned|stressed|anxious|desperate|"
            r"really need|need urgently|time sensitive", o):
        urgency = "high"
    elif re.search(
            r"when you get a chance|no rush|curious|whenever|not urgent", o):
        urgency = "low"
    else:
        urgency = "medium"

    if re.search(
            r"write|draft|help.{0,10}write|formal letter|"
            r"complaint letter|letter to|grievance", o):
        intent = "action"
    elif re.search(
            r"what are my rights|can i |am i entitled|"
            r"is it legal|what should i do|what can i do|"
            r"how do i|advise me|advice on", o):
        intent = "information"
    elif re.search(
            r"where is|track|status|find|locate|"
            r"check on|look into|investigate", o):
        intent = "retrieval"
    else:
        intent = "action"

    aux = []
    if re.search(
            r"birthday.{0,30}(saturday|sunday|tomorrow|next week|this week)",
            o):
        aux.append("birthday time pressure")
    m = re.search(
        r"almost a month|\w+ months? ago|\w+ weeks? ago|\w+ days? ago", o)
    if m: aux.append(m.group(0)[:35])
    m = re.search(r"for (the past )?(\w+ \w+|\w+) (months?|weeks?|days?)", o)
    if m:
        c = m.group(0)[:35]
        if c not in aux: aux.append(c)
    if re.search(r"\bbaby\b|young child|infant|toddler", o):
        aux.append("young child")
    if re.search(r"cannot work|affecting.{0,20}\bjob\b|work from home", o):
        aux.append("work impact")
    if re.search(r"no heating|freezing|boiler broken|no hot water", o):
        aux.append("no heating/hot water")
    if re.search(
            r"called.{0,15}(twice|three times|multiple|several|\d+ times)",
            o):
        aux.append("called multiple times")
    elif re.search(r"\bcalled\b|\bphoned\b|\brang\b", o):
        aux.append("called already")
    if re.search(
            r"emailed.{0,15}(twice|three|multiple|several|\d+ times)", o):
        aux.append("emailed multiple times")
    elif re.search(r"\bemailed\b", o):
        aux.append("emailed already")
    if re.search(r"(sent|submitted).{0,20}(form|ticket|report|claim)", o):
        aux.append("submitted already")
    if re.search(r"\bno\b.{0,10}(reply|response|answer|resolution|update)",
                 o):
        aux.append("no response received")
    if re.search(r"been told.{0,40}(will be|would be|should be)", o):
        aux.append("promised resolution not delivered")

    return {
        "compressed_prompt": compressed.strip(),
        "intent":     intent,
        "tone":       tone,
        "urgency":    urgency,
        "aux":        aux[:6],
        "confidence": 0.85,
    }


def _fallback_envelope(original: str, reason: str) -> dict:
    """
    Parse failure — return low-confidence envelope, never silent passthrough.
    """
    return {
        "compressed_prompt": original,
        "intent":    "unknown",
        "tone":      "neutral",
        "urgency":   "medium",
        "aux":       [],
        "passthrough":    False,
        "confidence":     0.0,
        "_parse_failure": reason,
    }


def _run_slm(text: str, domain: str = None) -> tuple[dict, float]:
    """
    Run Gemma SLM inference.
    Returns (envelope dict, latency_ms).

    Research runtime: llama-cpp-python on CPU.
    Production runtime: replace this function body with NPU SDK call.
    Model weights and _build_slm_prompt() format remain identical.
    """
    if _llm is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    prompt = _build_slm_prompt(text)

    t0 = time.time()
    response = _llm(
        prompt,
        max_tokens=SLM_MAX_TOKENS,
        temperature=SLM_TEMPERATURE,
        stop=["<end_of_turn>", "<start_of_turn>"],
        echo=False,
    )
    latency_ms = (time.time() - t0) * 1000

    raw = response["choices"][0]["text"].strip()

    # Clean output — remove markdown fences and leading labels
    raw = re.sub(r"```[a-z]*", "", raw).strip()
    raw = re.sub(
        r"^(Brief first-person version:|Summary:|Output:|Result:)\s*",
        "", raw, flags=re.I).strip()

    # Remove quoted wrapper if model echoed format
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1].strip()

    # Validate non-empty
    if not raw:
        return _fallback_envelope(text, "empty_output"), latency_ms

    # Build envelope using rule-based classification on original
    # SLM provides compressed_prompt only — all other fields rule-based
    envelope = _classify_rule_based(raw, text)
    envelope, _ = _validate_envelope(envelope)
    envelope = _fidelity_check(envelope, text)

    return envelope, latency_ms



# ─────────────────────────────────────────────
# TOKEN ECONOMICS HELPERS
# ─────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """
    Lightweight token estimator — no tokenizer dependency.
    Approximation: 1 token ≈ 0.75 words (standard rule of thumb).
    Good enough for the net saving gate; real tokenizers cost latency.
    """
    return max(1, round(len(text.split()) / 0.75))


def _compute_net_saving(original: str, compressed: str, aux: list) -> int:
    """
    Net token saving = tokens(original) - tokens(distilled packet)

    Distilled packet cost:
        tokens(compressed_prompt)
      + ALE_HEADER_OVERHEAD_TOKENS (fixed HEADER block)
      + aux overhead (~4 tokens per aux item for the "  - item" lines)

    A positive value means distillation saves tokens.
    A negative value means distillation costs more than passthrough.
    """
    original_tokens    = _estimate_tokens(original)
    compressed_tokens  = _estimate_tokens(compressed)
    aux_overhead       = len(aux) * 2   # inline semicolon-separated: ~2 tokens per item
    packet_tokens      = compressed_tokens + ALE_HEADER_OVERHEAD_TOKENS + aux_overhead
    return original_tokens - packet_tokens


# ── Structural patterns (complexity scorer v2) ────────────────
_STRUCTURAL_HARD = [
    r'''```[\s\S]{10,}?```''',
    r'def \w+\s*\([^)]*\)\s*(?:->|:)',
    r'function\s+\w+\s*\([^)]*\)\s*\{',
    r'class\s+\w+\s*(?:\([^)]*\))?\s*:',
    r'SELECT\s+.{5,}\s+FROM\s+\w+',
    r'<\?php|<!DOCTYPE',
    r'(?:const|let|var)\s+\w+\s*=\s*.{10,}[;,]',
    r'(?:public|private|protected)\s+\w+\s+\w+\s*\(',
    r'import\s+\w+(?:\.\w+)+',
    r'#include\s*<\w+>',
    r'from\s+\w+\s+import\s+\w+',
    r'Traceback\s+\(most\s+recent\s+call\s+last\)',
]
_STRUCTURAL_SOFT = [
    r'https?://\S{20,}',
    r'\b[A-Z]{2,5}-\d{4,}\b',
]


@dataclass
class ComplexityResult:
    """
    Complexity scorer v2 — minimal binary safety gate output.
    Removed: social_score, semantic_score, repetition_score,
             compression_ceiling, recommended_ratio, profile.
    Gemma does not need ratio guidance — gate only decides
    whether to attempt compression at all.
    """
    word_count:         int
    structural_score:   float
    passthrough:        bool
    passthrough_reason: Optional[str]

    def summary(self) -> str:
        return (
            f"[COMPLEXITY] structural={self.structural_score:.2f} "
            f"passthrough={self.passthrough}"
            + (f" reason={self.passthrough_reason}"
               if self.passthrough else "")
        )


def score_complexity(text: str) -> ComplexityResult:
    """
    Complexity scorer v2 — minimal binary safety gate.

    Removed: social_score, semantic_score, repetition_score,
             compression_ceiling, recommended_ratio, profile labels.

    Kept:
      1. Short prompt gate (<10 words) — Gemma expands short prompts
      2. Structural density gate (>=0.22) — code/math blocks

    Post-compression safety net lives in distill():
      If compressed word count >= original → passthrough no_saving.
    """
    words      = text.split()
    word_count = len(words)
    char_count = max(len(text), 1)

    if word_count < 10:
        return ComplexityResult(
            word_count=word_count,
            structural_score=0.0,
            passthrough=True,
            passthrough_reason="too_short",
        )

    structural_chars = 0.0
    for pattern in _STRUCTURAL_HARD:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            structural_chars += len(match.group())
    for pattern in _STRUCTURAL_SOFT:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            structural_chars += len(match.group()) * 0.8

    structural_score = min(1.0, structural_chars / char_count)

    if structural_score >= 0.22:
        return ComplexityResult(
            word_count=word_count,
            structural_score=structural_score,
            passthrough=True,
            passthrough_reason=f"structural_density_{structural_score:.2f}",
        )

    return ComplexityResult(
        word_count=word_count,
        structural_score=structural_score,
        passthrough=False,
        passthrough_reason=None,
    )


# ─────────────────────────────────────────────
# MAIN DISTILL ENTRY POINT
# ─────────────────────────────────────────────

def _escalate_urgency(urgency: str, aux: list) -> str:
    """
    If aux contains active stressors, promote urgency to high
    regardless of SLM-scored urgency value.
    """
    AUX_ESCALATORS = [
        'crying', 'emergency', 'deadline', 'meeting', 'hospital',
        'surgery', 'evict', 'homeless', 'no power', 'no heat',
        'shut off', 'disconnected', 'baby', 'child', 'infant',
        'pregnant', 'funeral', 'death', 'dying', 'urgent',
        'birthday time pressure', 'work impact', 'no heating',
    ]
    combined = " ".join(aux).lower()
    for escalator in AUX_ESCALATORS:
        if escalator in combined:
            return "high"
    return urgency


def distill(text: str, model_path: Optional[str] = None) -> DistillResult:
    """
    Main entry point. Returns DistillResult.

    text:       raw user prompt
    model_path: override model path (for Colab or custom setups)
    """
    t_start = time.time()
    text = text.strip()

    # ── Tier 1 ──────────────────────────────
    passthrough, reason = tier1_check(text)
    if passthrough:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=reason,
            confidence=1.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="tier1",
            domain=None,
        )

    # ── Domain detection ─────────────────────
    # Tag the domain for the SLM — does not block, only informs compression
    domain = _detect_domain(text)

    # ── Complexity scorer ────────────────────
    # Rule-based, ~0ms. Determines how much of the prompt is
    # compressible vs load-bearing. May trigger passthrough before
    # the SLM runs if structural density is too high.
    complexity = score_complexity(text)
    if complexity.passthrough:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"complexity_{complexity.passthrough_reason}",
            confidence=1.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="complexity",
            domain=domain,
        )

    # ── High Fidelity Guard (HFG) ────────────
    # Extract guaranteed aux context before SLM runs.
    # Life events, emotional anchors, dependency markers, crisis
    # indicators are captured here so they cannot be compressed away.
    # HFG phrases seed the aux list; SLM adds to it, never replaces it.
    hfg_aux = extract_hfg_phrases(text)

    # ── SLM ─────────────────────────────────
    # Lazy load if not already loaded
    if _llm is None:
        load_model(model_path)

    try:
        envelope, slm_latency_ms = _run_slm(
            text,
            domain=domain,
        )
    except Exception as e:
        # SLM failure — safe fallback to passthrough with reason
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"slm_error: {str(e)}",
            confidence=0.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="slm_error",
            domain=domain,
        )

    # ── Post-compression safety net ──────────────
    # If Gemma expanded the prompt instead of compressing it,
    # discard the compression and passthrough.
    # This catches: short dense prompts, no-scaffolding prompts,
    # and any unexpected model behaviour regardless of word count.
    compressed_wc = len(envelope.get("compressed_prompt", "").split())
    original_wc   = len(text.split())
    if compressed_wc >= original_wc and compressed_wc > 0:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"no_saving_{original_wc}w_to_{compressed_wc}w",
            confidence=1.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="post_compression",
            domain=domain,
        )

    # ── Tier 1b — fallback safety re-check ──
    compressed = envelope.get("compressed_prompt", text)
    tier1b_flag, tier1b_reason = tier1b_check(compressed)
    if tier1b_flag:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"tier1b_{tier1b_reason}",
            confidence=1.0,
            latency_ms=(time.time() - t_start) * 1000,
            tier="tier1b",
            domain=domain,
        )

    # ── Dynamic confidence gate ──────────────
    # Threshold scales with prompt length — short prompts need higher
    # confidence because the margin for error is smaller and the
    # compression upside is lower.
    word_count  = len(text.split())
    threshold   = dynamic_confidence_threshold(word_count)
    confidence  = envelope.get("confidence", 0.0)
    if confidence < threshold:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"low_confidence_{confidence:.2f}_threshold_{threshold:.2f}",
            confidence=confidence,
            latency_ms=(time.time() - t_start) * 1000,
            tier="slm",
            token_saving=0,
            domain=domain,
        )

    # ── Net token saving gate ────────────────
    # Single economic gate — if the full packet (compressed + header +
    # aux overhead) does not save at least MIN_NET_TOKEN_SAVING tokens
    # vs sending the original, passthrough. This catches poor compressions,
    # marginal compressions, and prompts too short to benefit — without
    # a separate ratio gate that penalises the SLM for keeping useful context.
    aux       = envelope.get("aux", [])
    net_saving = _compute_net_saving(text, compressed, aux)
    if net_saving < MIN_NET_TOKEN_SAVING:
        return DistillResult(
            original_prompt=text,
            compressed_prompt=text,
            intent="passthrough",
            tone="unknown",
            urgency="unknown",
            aux=[],
            passthrough=True,
            passthrough_reason=f"no_token_saving_{net_saving:+d}_tokens",
            confidence=confidence,
            latency_ms=(time.time() - t_start) * 1000,
            tier="slm",
            token_saving=net_saving,
            domain=domain,
        )

    # ── HFG merge ────────────────────────────
    # Merge HFG-extracted phrases into SLM aux.
    # HFG items are prepended (highest priority) then SLM items added
    # de-duplicated. HFG items are never dropped even if the SLM
    # didn't extract them — that's the guarantee.
    slm_aux_lower = {a.lower() for a in aux}
    hfg_new = [h for h in hfg_aux
               if not any(h.lower() in s or s in h.lower()
                          for s in slm_aux_lower)]
    aux = hfg_new + aux  # HFG first — highest priority

    # ── Aux urgency escalation ───────────────
    urgency  = _escalate_urgency(envelope.get("urgency", "medium"), aux)

    # ── Identifier preservation check ───────
    # Any identifiers in the original must appear in compressed_prompt
    identifiers = _extract_identifiers(text)
    for ident in identifiers:
        if ident not in compressed:
            compressed += f" [{ident}]"

    total_latency_ms = (time.time() - t_start) * 1000

    return DistillResult(
        original_prompt=text,
        compressed_prompt=compressed,
        intent=envelope.get("intent", "unknown"),
        tone=envelope.get("tone", "neutral"),
        urgency=urgency,
        aux=aux,
        passthrough=False,
        passthrough_reason=None,
        confidence=confidence,
        latency_ms=total_latency_ms,
        tier="slm",
        schema_valid=True,
        token_saving=net_saving,
        domain=domain,
        complexity=complexity,
        hfg_aux=hfg_new,
    )


# ─────────────────────────────────────────────
# BATCH EVALUATION
# ─────────────────────────────────────────────

def run_batch(prompts: list[str], model_path: Optional[str] = None) -> dict:
    """
    Run distillation on a list of prompts.
    Returns summary stats + per-prompt results.
    Useful for Colab evaluation cells.
    """
    results = []
    passthrough_count = 0
    tier1_count  = 0
    tier1b_count = 0
    slm_count    = 0
    latencies    = []
    total_tokens_saved = 0

    for prompt in prompts:
        r = distill(prompt, model_path=model_path)
        results.append(r)
        latencies.append(r.latency_ms)
        total_tokens_saved += r.token_saving

        if r.passthrough:
            passthrough_count += 1
            if r.tier == "tier1":
                tier1_count += 1
            elif r.tier == "tier1b":
                tier1b_count += 1
        else:
            slm_count += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    summary = {
        "total": len(prompts),
        "passthrough": passthrough_count,
        "distilled": slm_count,
        "tier1_exits": tier1_count,
        "tier1b_exits": tier1b_count,
        "avg_latency_ms": round(avg_latency, 1),
        "p95_latency_ms": round(p95_latency, 1),
        "passthrough_rate": f"{passthrough_count/len(prompts)*100:.1f}%",
        "total_tokens_saved": total_tokens_saved,
        "results": results,
    }
    return summary


def print_batch_summary(summary: dict) -> None:
    print("\n" + "="*50)
    print("BATCH EVALUATION SUMMARY")
    print("="*50)
    print(f"Total prompts    : {summary['total']}")
    print(f"Distilled        : {summary['distilled']}")
    print(f"Passthrough      : {summary['passthrough']} ({summary['passthrough_rate']})")
    print(f"  ↳ Tier 1 exits : {summary['tier1_exits']}")
    print(f"  ↳ Tier 1b exits: {summary['tier1b_exits']}")
    print(f"Total tokens saved: {summary['total_tokens_saved']:+d}")
    print(f"Avg latency      : {summary['avg_latency_ms']}ms")
    print(f"P95 latency      : {summary['p95_latency_ms']}ms")
    print("="*50)
    for i, r in enumerate(summary["results"]):
        print(f"\n[{i+1}] {r.summary()}")


# ─────────────────────────────────────────────
# INTERACTIVE REPL
# ─────────────────────────────────────────────

def run_interactive(model_path: Optional[str] = None) -> None:
    """
    Interactive REPL for manual testing.
    Type 'quit' or 'exit' to stop.
    Model loads lazily on first non-passthrough prompt.
    """
    print("SPSD v4 — Interactive REPL")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            text = input("Prompt > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue

        result = distill(text, model_path=model_path)
        print("\n" + result.summary() + "\n")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_interactive()
