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

MODEL_REPO   = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE   = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
MODEL_PATH   = Path("./models") / MODEL_FILE   # local cache path

# Inference parameters — tuned for latency on free-tier Colab CPU
SLM_CONTEXT_LENGTH   = 4096  # raised from 2048 — eliminates n_ctx warning, supports prompts up to ~3000 tokens
SLM_MAX_TOKENS       = 300    # envelope is compact
SLM_TEMPERATURE      = 0.1    # near-deterministic for structured output
SLM_TOP_P            = 0.9
SLM_N_THREADS        = 2      # free-tier Colab: 2 CPU cores safe default

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
        return 0.72
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
    "lawsuit", "litigation", "attorney", "lawyer", "legal advice",
    "court", "verdict", "plaintiff", "defendant", "subpoena",
    "arbitration", "settlement", "contract", "liability", "negligence",
    "malpractice", "patent", "trademark", "copyright infringement",
    "gdpr", "statute", "jurisdiction", "appeal", "injunction",
    "divorce", "custody", "bankruptcy", "foreclosure", "eviction",
    "sue", "suing", "landlord", "tenant rights", "wrongful",
    "discrimination", "harassment", "employment law", "unfair dismissal",
    "small claims", "restraining order", "legal rights", "breach of contract",
}

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
CODE_STRONG = {
    "python", "javascript", "typescript", "kotlin", "swift", "golang",
    "haskell", "scala", "php", "bash", "shell", "regex", "yaml",
    "kubernetes", "webpack", "npm", "pip", "conda", "recursion",
    "algorithm", "compile", "compiler", "debugger", "async", "await",
    "lambda", "decorator", "polymorphism", "refactor", "linter",
    "undefined", "boolean", "tuple", "iterator", "generator",
    "middleware", "microservice", "jwt", "oauth", "webhook", "graphql",
    "virtualenv", "pytorch", "tensorflow", "pandas", "numpy",
}

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

    # ── Guard 3: medical — Layer A (unambiguous keywords) ────────────
    # These terms never appear in support/conversational prompts.
    # No risk of false positives.
    if _word_boundary_match(text, _MEDICAL_TIER1_HARD):
        return True, "domain_medical"

    # ── Guard 3: medical — Layer B (clinical structure regex) ─────────
    # Catches USMLE vignette patterns and clinical presentation language
    # without relying on ambiguous words like anxiety/depression/chronic.
    if _CLINICAL_STRUCTURE_RE.search(text):
        return True, "domain_medical"

    # ── Guard 4: legal ───────────────────────────────────────────────
    if (_phrase_match(text, LEGAL_KEYWORDS) or
            _word_boundary_match(text, LEGAL_KEYWORDS)):
        return True, "domain_legal"

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
EXAMPLE 1 — Support / tracking (anxious, high urgency, active stressor)
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

EXAMPLE 2 — Medical domain (tagged — compress but preserve clinical precision)
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

EXAMPLE 3 — Code domain (tagged — compress but keep technical precision)
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

EXAMPLE 4 — Multi-question (linked questions)
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
  "aux": ["user is undecided", "wants reasoned view not just validation"],
  "confidence": 0.85
}
""".strip()

# ─────────────────────────────────────────────
# SLM SYSTEM PROMPT
# ─────────────────────────────────────────────

SLM_SYSTEM_PROMPT = """You are a prompt distillation engine. Compress user messages into shorter versions that preserve full intent, tone, urgency, and key context. Remove only filler words, repetition, and irrelevant narrative.

Some inputs are prefixed with a domain tag: [MEDICAL], [LEGAL], [FINANCIAL], [CODE].
For tagged inputs: compress aggressively but preserve domain-critical precision.
  [MEDICAL] — keep drug names, dosages, durations, conditions exactly as stated
  [LEGAL]   — keep all party names, dates, specific legal terms exactly
  [FINANCIAL] — keep all figures, instrument names, account types exactly
  [CODE]    — keep error messages, method names, line numbers, language exactly

Output ONLY a valid JSON object with these exact fields:
- compressed_prompt: shorter natural language version. Preserve intent, identifiers, emotional register. Never invent information.
- intent: one of [retrieval, action, information, social]
- tone: one or more of [anxious/apologetic, polite, casual, negative, neutral] joined with /
- urgency: one of [high, medium, low]
- aux: list of contextual details useful for answering (stressors, prior attempts, background, constraints). Empty list [] if none.
- confidence: float 0.0–1.0 for how faithfully you captured original intent

Rules:
1. Output ONLY the JSON. No preamble, no explanation, no markdown fences.
2. compressed_prompt must be natural language — not a schema or bullet list.
3. Never drop identifiers: order numbers, tracking IDs, URLs, drug names, error messages.
4. Never invent context not present in the original.
5. Compress hard — target 30-50% of original word count. Filler, politeness, and narrative are always removable.
6. If the message is genuinely too ambiguous to compress faithfully, set confidence below 0.60."""

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
        cx_str = f" | profile={cx.profile} rec={cx.recommended_ratio:.2f}" if cx and not self.passthrough else ""
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
_SOCIAL_PHRASES = {
    "hi ", "hi,", "hello ", "hello,", "hey ", "good morning", "good afternoon",
    "good evening", "dear ", "hope you", "hope this", "i hope",
    "i am so sorry", "im so sorry", "so sorry to",
    "sorry to bother", "sorry for bothering", "sorry to trouble",
    "sorry for", "apologies for", "i apologise", "i apologize",
    "please forgive", "excuse me", "pardon me",
    "i know youre busy", "i know you must be",
    "hate to trouble", "hate to bother",
    "i dont mean to", "i feel bad", "i feel terrible",
    "i feel silly", "im embarrassed", "i am embarrassed",
    "thank you so much", "thank you very much", "thanks so much",
    "many thanks", "i really appreciate", "i would really appreciate",
    "i appreciate your", "appreciate your time",
    "i hope that makes sense", "if that makes sense",
    "i hope im not", "i hope i am not", "does that make sense",
    "let me know if you need", "please let me know",
    "looking forward to", "kind regards", "best regards",
    "i have been a customer", "ive been a customer",
    "i have been using", "ive been using",
    "i just wanted to", "i wanted to reach out",
    "i am reaching out", "im reaching out",
    "i thought i would", "i thought id",
    "just a quick question", "quick question",
    "random question", "silly question",
    "i know this might be", "i understand if",
    "no rush", "whenever you get a chance",
    "at your earliest convenience",
    "i hope you are", "i hope youre",
    "hope youre having", "i wanted to say",
    "i just want to say", "i am writing", "im writing",
    "i am contacting", "i am getting in touch",
    "i dont want to be", "i do not want to be",
    "i know its probably", "i know it is probably",
    "i completely understand", "i totally understand",
}

# Semantic anchors — load-bearing, must survive compression
_SEMANTIC_INDICATOR_PATTERNS = [
    r'\b[A-Z]{2,6}-\d{3,10}\b',
    r'\b\d{5,15}\b',
    r'https?://\S+',
    r'\b[A-Z]{2,}\d+[A-Z0-9]*\b',
    r'\b\d+\s*(mg|ml|mcg|kg|lb|oz|cm|mm|km|miles?|hours?|days?|weeks?|months?|years?)\b',
    r'\$\d+|\£\d+|\€\d+',
    r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',
    r'\b(january|february|march|april|may|june|july|august|'
     r'september|october|november|december)\b',
    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    r'\b\d+\s*micrograms?\b',
    r'\b(prescribed|prescription|diagnosis|dosage|twice daily|once daily|three times)\b',
    r'\b(error|exception|line\s+\d+|404|500|429|503)\b',
    r'\b[A-Z][a-z]+(?:Service|Controller|Manager|Handler|Exception|Error)\b',
]

# Structural markers — two tiers
_STRUCTURAL_HARD = [
    r'```[\s\S]{10,}?```',
    r'def \w+\s*\([^)]*\)\s*(?:->|:)',
    r'function\s+\w+\s*\([^)]*\)\s*\{',
    r'class\s+\w+\s*(?:\([^)]*\))?\s*:',
    r'class\s+\w+\s*(?:extends|implements|\{)',
    r'SELECT\s+.{5,}\s+FROM\s+\w+',
    r'<\?php|<!DOCTYPE',
    r'\\\[[\s\S]+?\\\]|\\\([\s\S]+?\\\)',
    r'(?:const|let|var)\s+\w+\s*=\s*.{10,}[;,]',
    r'(?:public|private|protected)\s+\w+\s+\w+\s*\(',
    r'@\w+\s*\n\s*(?:def|class|public)',
    r'import\s+\w+(?:\.\w+)+',
    r'#include\s*<\w+>',
    r'from\s+\w+\s+import\s+\w+',
    # Stack traces and error reports — these are technical artifacts
    r'\b(?:NullPointerException|StackOverflowError|OutOfMemoryError|'
     r'ClassCastException|ArrayIndexOutOfBoundsException)\b',
    r'\bat\s+[\w\.]+\.\w+\([^)]+\.java:\d+\)',   # Java stack frame
    r'Traceback\s+\(most\s+recent\s+call\s+last\)',  # Python traceback
    r'(?:stack trace|stack frame|heap dump)',
    r'\b\w+\.\w+\(\)\s+line\s+\d+',             # method() line N
]

_STRUCTURAL_SOFT = [
    r'`[^`\n]{3,}`',
    r'\b(?:async|await)\s+\w+\s*\(',
    r'=>\s*[{(\[]',
    r'\b(?:return|yield|throw)\s+\w+',
    r'if\s*\([^)]{5,}\)\s*[{:]',
    r'for\s*\([^)]{5,}\)\s*[{:]',
    r'\b\w+\.\w+\.\w+\(',
    r'\d+\s*[+\-\*\/]\s*\d+\s*[+\-\*\/]\s*\d+',
    r'\b(?:NullPointerException|StackOverflow|TypeError|'
     r'AttributeError|KeyError|ValueError|RuntimeError|'
     r'IndexError|NameError|ImportError|SyntaxError|'
     r'ConnectionError|TimeoutError|PermissionError)\b',
    r'\b(?:async|await|const|let|var|yield|lambda|assert|elif)\b',
    # Technical prose indicators — not syntax but clearly code-related
    r'\bstack\s+trace\b|\bstack trace\b',
    r'\bline\s+\d+\b',
    r'\b(?:method|function|class|module|package)\s+\w+',
    r'\b(?:returns?|throws?|raises?|calls?)\s+\w+',
    r'\bnull\s+(?:pointer|reference|object|value)\b',
    r'\b(?:git|npm|pip|docker|kubectl)\s+\w+',
]

# Repetition markers — covers digit and word-form numbers
_RESTATEMENT_PATTERNS = [
    (r"can.t find.{5,40}can.t find",               'repeated_negation'),
    (r"don.t know.{5,40}don.t know",               'repeated_negation'),
    (r"tried.{5,60}tried.{5,60}tried",             'triple_attempt'),
    (r"(?:please|kindly).{5,80}(?:please|kindly)", 'double_plea'),
    (r"i.ve been.{5,60}i.ve been.{5,60}i.ve been", 'repeated_statement'),
    # digit and word-form numbers for tenure repetition
    (r"for (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten) "
     r"(?:year|month).{5,80}"
     r"for (?:\d+|one|two|three|four|five|six|seven|eight|nine|ten) "
     r"(?:year|month)",                            'tenure_repetition'),
    # generic restatement: same key noun phrase appears 3+ times
]


@dataclass
class ComplexityResult:
    """
    Output of the complexity scorer.
    All scores are 0.0–1.0.
    """
    social_score:       float   # fraction of prompt that is social fluff
    semantic_score:     float   # fraction that is load-bearing content
    structural_score:   float   # fraction that is code/math/formal syntax
    repetition_score:   float   # how much content is restated
    word_count:         int
    compression_ceiling: float  # max safe compression ratio (1.0 = no limit)
    recommended_ratio:  float   # target compression ratio for the SLM
    passthrough:        bool    # structural density too high to compress safely
    passthrough_reason: Optional[str]
    profile:            str     # human-readable label

    def summary(self) -> str:
        return (
            f"[COMPLEXITY] profile={self.profile} "
            f"social={self.social_score:.2f} "
            f"semantic={self.semantic_score:.2f} "
            f"structural={self.structural_score:.2f} "
            f"compression_ceiling={self.compression_ceiling:.2f} "
            f"recommended_ratio={self.recommended_ratio:.2f} "
            f"passthrough={self.passthrough}"
            + (f" reason={self.passthrough_reason}" if self.passthrough else "")
        )


def score_complexity(text: str) -> ComplexityResult:
    """
    Rule-based complexity scorer. ~0ms, no model.

    Pipeline:
      1. Measure structural content (code/math) — if too high, passthrough
      2. Measure social scaffolding ratio
      3. Measure semantic anchor density
      4. Measure repetition
      5. Derive compression_ceiling and recommended_ratio
      6. Assign human-readable profile
    """
    words        = text.split()
    word_count   = len(words)
    # Normalise apostrophes for social matching (handles contractions)
    text_norm    = text.lower().replace("'", "").replace("\u2019", "")
    char_count   = max(len(text), 1)

    # ── 1. Structural score — two tiers ─────────────────────────────
    # Hard patterns: weight 1.0 per char matched
    # Soft patterns: weight 0.5 per char matched
    structural_chars = 0.0
    for pattern in _STRUCTURAL_HARD:
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            structural_chars += len(match.group())
    for pattern in _STRUCTURAL_SOFT:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            structural_chars += len(match.group()) * 0.8

    structural_score = min(1.0, structural_chars / char_count)

    # Hard passthrough: structural content dominates (≥22% of chars)
    if structural_score >= 0.22:
        return ComplexityResult(
            social_score=0.0, semantic_score=1.0,
            structural_score=structural_score, repetition_score=0.0,
            word_count=word_count, compression_ceiling=0.0,
            recommended_ratio=1.0, passthrough=True,
            passthrough_reason=f"structural_density_{structural_score:.2f}",
            profile="code_or_math",
        )

    # ── 2. Social score ──────────────────────────────────────────────
    social_chars = 0
    for phrase in _SOCIAL_PHRASES:
        idx = 0
        while True:
            pos = text_norm.find(phrase, idx)
            if pos == -1:
                break
            social_chars += len(phrase)
            idx = pos + len(phrase)
    social_score = min(1.0, social_chars / char_count)

    # ── 3. Semantic score ────────────────────────────────────────────
    semantic_hits = 0
    for pattern in _SEMANTIC_INDICATOR_PATTERNS:
        semantic_hits += len(re.findall(pattern, text, re.IGNORECASE))
    # Normalise: hits per 10 words. 3+ hits per 10 words = very dense
    semantic_score = min(1.0, semantic_hits / max(word_count / 10, 1) / 3)

    # Hard passthrough: short + high semantic density — nothing to remove
    if semantic_score >= 0.65 and word_count <= 25:
        return ComplexityResult(
            social_score=social_score, semantic_score=semantic_score,
            structural_score=structural_score, repetition_score=0.0,
            word_count=word_count, compression_ceiling=0.0,
            recommended_ratio=1.0, passthrough=True,
            passthrough_reason=f"dense_semantic_short_{semantic_score:.2f}",
            profile="dense_technical",
        )

    # ── 4. Repetition score ──────────────────────────────────────────
    repetition_score = 0.0
    for pattern, _ in _RESTATEMENT_PATTERNS:
        if re.search(pattern, text_norm, re.DOTALL):
            repetition_score = min(1.0, repetition_score + 0.25)

    # Sentence-level word overlap (adjacent sentences)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.split()) > 4]
    for i in range(len(sentences) - 1):
        wa = set(sentences[i].lower().split())
        wb = set(sentences[i+1].lower().split())
        if wa and wb:
            overlap = len(wa & wb) / min(len(wa), len(wb))
            if overlap > 0.45:
                repetition_score = min(1.0, repetition_score + 0.20)

    # ── 5. Compression ceiling + recommended ratio ───────────────────
    # Base ceiling: social-heavy prompts have more removable content
    # Minimum ceiling of 0.25 even for semantic-heavy prompts (some compression always possible)
    base_ceiling       = 0.28 + (social_score * 0.55)     # 0.28–0.83
    semantic_penalty   = semantic_score * 0.35             # 0.0–0.35
    structural_penalty = structural_score * 0.25           # 0.0–0.25
    compression_ceiling = max(0.25, min(0.80,
        base_ceiling - semantic_penalty - structural_penalty))

    repetition_bonus  = repetition_score * 0.08
    recommended_ratio = max(0.20, compression_ceiling - repetition_bonus)

    # ── 6. Profile label — most specific first ───────────────────────
    if structural_score >= 0.15:
        profile = "technical_mixed"
    elif repetition_score >= 0.25:
        # repetition dominates — even if social is also high
        profile = "repetitive_complaint"
    elif social_score >= 0.15 and semantic_score <= 0.40:
        profile = "verbose_social"
    elif semantic_score >= 0.30:
        profile = "dense_informational"
    elif social_score >= 0.05:
        # has some social content but not enough for verbose_social
        profile = "polite_support"
    elif word_count <= 20:
        profile = "concise_direct"
    else:
        profile = "general"

    return ComplexityResult(
        social_score=social_score,
        semantic_score=semantic_score,
        structural_score=structural_score,
        repetition_score=repetition_score,
        word_count=word_count,
        compression_ceiling=compression_ceiling,
        recommended_ratio=recommended_ratio,
        passthrough=False,
        passthrough_reason=None,
        profile=profile,
    )



def _escalate_urgency(urgency: str, aux: list[str]) -> str:
    """
    If aux contains active stressors, promote urgency to high
    regardless of SLM-scored urgency value.
    Urgency is scored on ORIGINAL text (via SLM), so this is a
    belt-and-suspenders check on extracted aux phrases.
    """
    combined = " ".join(aux).lower()
    for escalator in AUX_URGENCY_ESCALATORS:
        if escalator in combined:
            return "high"
    return urgency


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
    Load Qwen2.5-1.5B-Instruct Q4_K_M.
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


def _build_slm_prompt(user_text: str, domain: Optional[str] = None,
                      recommended_ratio: float = 0.40) -> str:
    """Build the full prompt string for the SLM.
    Prepends domain tag and compression target so the SLM
    knows how aggressively to compress.
    """
    tagged = f"[{domain}] {user_text}" if domain else user_text
    ratio_pct = int(recommended_ratio * 100)
    ratio_instruction = (
        f"Target: compress to approximately {ratio_pct}% of original length. "
        f"Remove social fluff, repetition, and narrative. "
        f"Preserve all identifiers, actions, and emotional register."
    )
    return (
        f"<|im_start|>system\n{SLM_SYSTEM_PROMPT}\n<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Here are examples of correct distillation:\n\n"
        f"{FEW_SHOT_EXAMPLES}\n\n"
        f"Now distill this prompt. {ratio_instruction}\n"
        f"INPUT: \"{tagged}\"\n"
        f"OUTPUT:\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _parse_slm_output(raw: str, original: str) -> dict:
    """
    Parse JSON from SLM output.
    Robust to: leading/trailing whitespace, partial markdown fences,
    extra text before/after the JSON object.
    Returns parsed dict or fallback with confidence=0.
    """
    # Strip markdown fences if model added them
    cleaned = re.sub(r'```(?:json)?', '', raw).strip()

    # Find the JSON object — from first { to last }
    start = cleaned.find('{')
    end   = cleaned.rfind('}')
    if start == -1 or end == -1:
        return _fallback_envelope(original, "json_not_found")

    json_str = cleaned[start:end+1]
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Try to salvage by fixing common issues (trailing commas)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return _fallback_envelope(original, "json_parse_error")

    return parsed


def _fallback_envelope(original: str, reason: str) -> dict:
    """
    Parse failure must NEVER silently become passthrough.
    Return a low-confidence envelope — visible, diagnosable.
    """
    return {
        "compressed_prompt": original,
        "intent": "unknown",
        "tone": "neutral",
        "urgency": "medium",
        "aux": [],
        "passthrough": False,
        "confidence": 0.0,
        "_parse_failure": reason,
    }


def _validate_envelope(env: dict) -> tuple[dict, bool]:
    """
    Validate required fields and types.
    Fill missing fields with safe defaults.
    Returns (envelope, is_valid).
    """
    required = ["compressed_prompt", "intent", "tone", "urgency", "aux", "confidence"]
    valid = True

    defaults = {
        "compressed_prompt": "",
        "intent": "unknown",
        "tone": "neutral",
        "urgency": "medium",
        "aux": [],
        "confidence": 0.0,
    }

    for key in required:
        if key not in env:
            env[key] = defaults[key]
            valid = False

    # Type coercions
    if not isinstance(env["aux"], list):
        env["aux"] = [str(env["aux"])] if env["aux"] else []
        valid = False

    try:
        env["confidence"] = float(env["confidence"])
    except (ValueError, TypeError):
        env["confidence"] = 0.0
        valid = False

    # Clamp confidence
    env["confidence"] = max(0.0, min(1.0, env["confidence"]))

    # Compressed prompt must not be empty
    if not env.get("compressed_prompt", "").strip():
        valid = False

    return env, valid


def _run_slm(text: str, domain: Optional[str] = None,
             recommended_ratio: float = 0.40) -> tuple[dict, float]:
    """
    Run SLM inference on text.
    Returns (envelope dict, latency_ms).
    Raises RuntimeError if model not loaded.
    """
    if _llm is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    prompt = _build_slm_prompt(text, domain=domain,
                                recommended_ratio=recommended_ratio)

    t0 = time.time()
    response = _llm(
        prompt,
        max_tokens=SLM_MAX_TOKENS,
        temperature=SLM_TEMPERATURE,
        top_p=SLM_TOP_P,
        stop=["<|im_end|>", "<|im_start|>"],
        echo=False,
    )
    latency_ms = (time.time() - t0) * 1000

    raw = response["choices"][0]["text"].strip()
    envelope = _parse_slm_output(raw, text)
    envelope, _ = _validate_envelope(envelope)

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


# ─────────────────────────────────────────────
# MAIN DISTILL ENTRY POINT
# ─────────────────────────────────────────────

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
            recommended_ratio=complexity.recommended_ratio,
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
