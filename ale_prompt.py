"""
ALE — Adaptive Logic Engine (v4.1)
Builds the message packet for the frontier LLM using the SPSD v4.1 DistillResult.

Cache architecture:
    CACHED (identical every call):
        ALE_SYSTEM_PROMPT — all rules, voice maps, quality checks (~420 tokens)
        cache_control: {"type": "ephemeral"} — 5-minute TTL on Anthropic API
    NOT CACHED (rebuilt per request):
        User turn — compressed_prompt (or original if passthrough),
                    metadata header (tone, urgency, aux, optional domain)

Critical rule: NOTHING dynamic enters the system prompt.
Urgency, tone, aux — all live in the user turn only.

v4.1 change: W:<n> word budget removed from header.
Output length control is outside scope of input-side distillation.
Enforce output length via max_tokens at API call level instead.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spsd_v4 import DistillResult

# ─────────────────────────────────────────────
# STATIC SYSTEM PROMPT — CACHED, NEVER MODIFIED AT RUNTIME
# ─────────────────────────────────────────────

ALE_SYSTEM_PROMPT = """You are a support and information assistant.

Each message is either a PASSTHROUGH or a DISTILLED packet.

PASSTHROUGH — two lines:
P
<original user message>
Respond naturally. Ignore all rules below. Never mention passthrough.

DISTILLED — two lines:
D[|DOMAIN]|<tone>|<urgency>|<aux1>;<aux2>;...
<compressed message>
Fields: D=distilled, optional DOMAIN tag (MEDICAL/LEGAL/FINANCIAL/CODE), tone, urgency, aux=semicolon-separated context.
Domain guidance:
  MEDICAL   → be precise, recommend consulting a professional, do not diagnose
  LEGAL     → be precise, recommend consulting a lawyer, do not give specific legal advice
  FINANCIAL → be precise, recommend consulting a financial advisor for investment decisions
  CODE      → be technical, match the language/framework mentioned, give working solutions

RULE 1 — ANSWER
a) Address the request directly and completely.
b) Reproduce identifiers exactly — order numbers, IDs, URLs. Never paraphrase.
c) Missing info needed? Say so. Never invent specifics.
d) Never ask for something the user already said they tried or don't have.
e) All self-service paths before any clarifying question.
f) No false capability claims — you cannot retrieve orders, access accounts, or look up live data.

RULE 2 — VOICE
tone=anxious/apologetic + urgency=high:
  Opener: "No need to apologise — let me get this sorted right now."
  Agent language: "let me" not "we can". No bullet lists. Answer by sentence two.
urgency=high → answer by sentence two, one opener max.
urgency=medium → warm, clear, moderate detail.
urgency=low → full explanation, structure welcome.
tone=casual → conversational, no support-ticket framing.
tone=negative → one-sentence acknowledgement, then resolution.
AUX OVERRIDE: aux contains crying/emergency/deadline/meeting/hospital → treat as urgency=high.

RULE 3 — AUX
Weave aux through framing — never name it.
  RIGHT: "Since the order was placed yesterday..."
  WRONG: "I can see your order was placed yesterday" / "I understand your kids are crying"
Prior attempt in aux → respect it: "if that's buried or missing" not "check your email."

RULE 4 — CONTINUITY
Never mention: distillation, compression, packet, metadata, header, preprocessing.

RULE 5 — EXPLANATORY
Answer all aspects in order. Pitch depth to any user level in aux.
If aux has a skip/shortcut question → address it directly.

RULE 6 — MULTI-QUESTION
Answer every question in order. If aux says questions are linked, connect them explicitly."""

# Build the cached system block for Anthropic API
CACHED_SYSTEM_BLOCK = [
    {
        "type": "text",
        "text": ALE_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]

# ─────────────────────────────────────────────
# WORD BUDGET FORMULA

# ─────────────────────────────────────────────
# USER TURN BUILDER
# ─────────────────────────────────────────────

def _build_user_turn(result: "DistillResult") -> str:
    """
    Build compact user turn from DistillResult.

    Passthrough format (~2 tokens overhead):
        P
        <original prompt>

    Distilled format (single header line + message):
        D[|DOMAIN]|<tone>|<urgency>|<aux1>;<aux2>;...
        <compressed prompt>

    W:<n> word budget removed in v4.1 — output length control is
    outside the scope of input-side distillation. T3 hypothesis test
    confirmed Llama 3.1 8B does not honour soft budget hints.
    Output token control should be enforced via max_tokens at API level.

    Fixed overhead of compact header: ~8-10 tokens vs ~40 for verbose.
    Aux items encoded as semicolon-separated inline — no per-item line breaks.
    """
    if result.passthrough:
        return f"P\n{result.original_prompt}"

    aux_str    = ";".join(result.aux) if result.aux else ""
    domain_str = f"|{result.domain}" if getattr(result, 'domain', None) else ""
    header     = f"D{domain_str}|{result.tone}|{result.urgency}|{aux_str}"

    return f"{header}\n{result.compressed_prompt}"


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def build_ale_messages(result: "DistillResult") -> dict:
    """
    Main ALE entry point.
    Returns dict with system= and messages= ready to pass to any
    frontier LLM API that accepts a system prompt and messages list.

    The returned dict is model-agnostic — it contains no reference to
    any specific frontier LLM provider. Pass it to whichever API you use:

    Groq (used in evaluation pipeline):
        from groq import Groq
        client = Groq(api_key=GROQ_KEY)
        packet = build_ale_messages(distill_result)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": packet["system"][0]["text"]},
                packet["messages"][0],
            ],
            max_tokens=150,
            temperature=0.4,
        )

    Anthropic (optional, for live demos):
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.messages.create(
            model="claude-sonnet-4-6",   # or any other Anthropic model
            max_tokens=300,
            temperature=0.4,
            **packet,
        )

    OpenAI-compatible APIs:
        messages = [
            {"role": "system", "content": packet["system"][0]["text"]},
            packet["messages"][0],
        ]
        response = openai_client.chat.completions.create(
            model="gpt-4o", messages=messages, max_tokens=300
        )
    """
    user_turn = _build_user_turn(result)

    return {
        "system": CACHED_SYSTEM_BLOCK,
        "messages": [
            {"role": "user", "content": user_turn}
        ],
    }


# ─────────────────────────────────────────────
# ROUND TRIP
# ─────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class RoundTripResult:
    """
    Complete record of one full pipeline pass.
    Contains everything needed to evaluate distillation quality:
      - what the user sent
      - what the SLM compressed it to + metadata
      - what the frontier LLM responded
      - token counts and latency at each stage
    """
    # Distillation
    original_prompt:    str
    compressed_prompt:  str
    passthrough:        bool
    passthrough_reason: str
    intent:             str
    tone:               str
    urgency:            str
    aux:                list
    confidence:         float
    distill_latency_ms: float
    distill_tier:       str

    # Frontier LLM
    frontier_response:      str
    input_tokens:           int
    output_tokens:          int
    cache_read_tokens:      int
    cache_write_tokens:     int
    frontier_latency_ms:    float

    # Totals
    total_latency_ms:       float

    def display(self) -> None:
        """Print a clear side-by-side view of the full round trip."""
        sep = "=" * 60

        print(f"\n{sep}")
        print("ROUND TRIP RESULT")
        print(sep)

        # ── Original ──
        print("\n[ ORIGINAL PROMPT ]")
        print(self.original_prompt)

        # ── Distillation ──
        print(f"\n[ DISTILLATION — {self.distill_tier} | {self.distill_latency_ms:.0f}ms ]")
        if self.passthrough:
            print(f"  PASSTHROUGH — {self.passthrough_reason}")
            print("  Original forwarded to frontier LLM unchanged.")
        else:
            print(f"  Compressed : {self.compressed_prompt}")
            print(f"  Intent     : {self.intent}")
            print(f"  Tone       : {self.tone}")
            print(f"  Urgency    : {self.urgency}")
            print(f"  Aux        : {self.aux}")
            print(f"  Confidence : {self.confidence:.2f}")
            orig_words = len(self.original_prompt.split())
            comp_words = len(self.compressed_prompt.split())
            reduction  = (1 - comp_words / orig_words) * 100 if orig_words else 0
            print(f"  Word reduction: {orig_words} → {comp_words} words (~{reduction:.0f}%)")

        # ── Frontier LLM response ──
        print(f"\n[ FRONTIER LLM RESPONSE ]")
        print(self.frontier_response)

        # ── Token economics ──
        print(f"\n[ TOKEN ECONOMICS ]")
        print(f"  Input tokens      : {self.input_tokens}")
        print(f"  Output tokens     : {self.output_tokens}")
        if self.cache_write_tokens:
            print(f"  Cache write       : {self.cache_write_tokens} (1.25x, one-time)")
        if self.cache_read_tokens:
            print(f"  Cache read        : {self.cache_read_tokens} (0.10x, 90% saving)")

        # ── Latency ──
        print(f"\n[ LATENCY ]")
        print(f"  Distillation      : {self.distill_latency_ms:.0f}ms")
        print(f"  Frontier LLM      : {self.frontier_latency_ms:.0f}ms")
        print(f"  Total             : {self.total_latency_ms:.0f}ms")
        print(sep)


def round_trip(
    prompt: str,
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 300,
    temperature: float = 0.4,
    spsd_model_path: str = None,
    provider: str = "groq",
) -> RoundTripResult:
    """
    Full pipeline: distill → ALE packet → frontier LLM → RoundTripResult.
    Convenience function for live demos. The evaluation pipeline uses
    run_spsd_v2.py + run_llm_v2.py + score_and_test.py instead.

    Args:
        prompt:           raw user prompt
        api_key:          API key for the chosen provider
        model:            frontier model identifier
                          Groq:     "llama-3.1-8b-instant" (default)
                          Anthropic:"claude-sonnet-4-6"
                          OpenAI:   "gpt-4o"
        max_tokens:       max tokens for frontier response
        temperature:      frontier LLM temperature
        spsd_model_path:  path to Qwen GGUF model (optional override)
        provider:         "groq" (default) | "anthropic" | "openai"

    Returns:
        RoundTripResult — everything in one object, call .display() to print.
    """
    import time
    import spsd_v4

    # ── Step 1: Distill ──────────────────────────────────────
    distill_result = spsd_v4.distill(prompt, model_path=spsd_model_path)

    # ── Step 2: Build ALE packet ─────────────────────────────
    packet = build_ale_messages(distill_result)
    system_text = packet["system"][0]["text"]
    user_msg    = packet["messages"][0]

    # ── Step 3: Call frontier LLM ────────────────────────────
    t0 = time.time()

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                user_msg,
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        frontier_text   = response.choices[0].message.content or ""
        usage           = response.usage
        cache_read      = 0
        cache_write     = 0
        input_tokens    = usage.prompt_tokens
        output_tokens   = usage.completion_tokens

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **packet,
        )
        frontier_text = response.content[0].text
        usage         = response.usage
        cache_read    = getattr(usage, "cache_read_input_tokens",  0) or 0
        cache_write   = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tokens  = usage.input_tokens
        output_tokens = usage.output_tokens

    elif provider == "openai":
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_text},
                user_msg,
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        frontier_text = response.choices[0].message.content or ""
        usage         = response.usage
        cache_read    = 0
        cache_write   = 0
        input_tokens  = usage.prompt_tokens
        output_tokens = usage.completion_tokens

    else:
        raise ValueError(f"Unknown provider: {provider!r}. Use 'groq', 'anthropic', or 'openai'.")

    frontier_latency_ms = (time.time() - t0) * 1000
    total_latency_ms    = distill_result.latency_ms + frontier_latency_ms

    return RoundTripResult(
        # Distillation fields
        original_prompt    = distill_result.original_prompt,
        compressed_prompt  = distill_result.compressed_prompt,
        passthrough        = distill_result.passthrough,
        passthrough_reason = distill_result.passthrough_reason or "",
        intent             = distill_result.intent,
        tone               = distill_result.tone,
        urgency            = distill_result.urgency,
        aux                = distill_result.aux,
        confidence         = distill_result.confidence,
        distill_latency_ms = distill_result.latency_ms,
        distill_tier       = distill_result.tier,

        # Frontier LLM fields
        frontier_response   = frontier_text,
        input_tokens        = input_tokens,
        output_tokens       = output_tokens,
        cache_read_tokens   = cache_read,
        cache_write_tokens  = cache_write,
        frontier_latency_ms = frontier_latency_ms,

        total_latency_ms    = total_latency_ms,
    )


# ─────────────────────────────────────────────
# DEBUG PREVIEW
# ─────────────────────────────────────────────

def preview_packet(result: "DistillResult") -> None:
    """Pretty-print the full ALE packet for debugging."""
    user_turn = _build_user_turn(result)
    lines     = user_turn.split('\n')
    tokens    = round(len(user_turn.split()) / 0.75)

    print("\n" + "="*60)
    print("ALE PACKET PREVIEW")
    print("="*60)
    print(f"[USER TURN — ~{tokens} tokens]")
    for line in lines:
        print(f"  {line}")
    print("="*60)
