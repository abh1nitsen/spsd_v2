"""
ALE — Adaptive Logic Engine (v4.2)
====================================
Builds the message packet for the frontier LLM using the SPSD v4.3 DistillResult.

v4.2 changes (aligned with Gemma compression + complexity scorer v2):
  - ALE system prompt updated: AI assistant framing (not support agent)
  - Tone tags updated: anxious / distressed / frustrated / neutral
  - Removed: D|DOMAIN pipe format (replaced by lean bracket annotation)
  - Format: [tone|urgency — context] compressed_message
  - No fabrication of system access
  - JSON extraction fix in judge (re.search for JSON object)

Cache architecture:
    CACHED (identical every call):
        ALE_SYSTEM_PROMPT — role + tone tag definitions (~150 tokens)
        cache_control: {"type": "ephemeral"} — 5-minute TTL
    NOT CACHED (rebuilt per request):
        User turn — [annotation] compressed_prompt

Critical rule: NOTHING dynamic enters the system prompt.
Tone, urgency, context — all live in the user turn annotation only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from spsd_v4 import DistillResult

# ── Static system prompt — cached, never modified at runtime ──
ALE_SYSTEM_PROMPT = """You are an AI assistant helping a user with their situation.

The user message starts with a tone tag in brackets:
  [anxious]    — person is worried and uncertain. Reassure briefly, then help.
  [distressed] — person is upset and confused. Acknowledge briefly, then give clear practical steps.
  [frustrated] — person knows they have been wronged. Skip comfort. Validate and go straight to action.
  [neutral]    — respond helpfully and directly.

Do not fabricate access to systems or accounts you do not have.
Respond to what the person actually asked. Be specific and practical."""

# ── Raw passthrough system prompt ─────────────────────────────
RAW_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Respond helpfully and specifically."
)


# ── Annotation builder ────────────────────────────────────────
def build_annotation(original: str, compressed: str) -> str:
    """
    Build lean bracket annotation from original prompt.
    Format: [tone] or [tone|urgency] or [tone|urgency — context]
    Only adds fields when they add information not visible in compressed message.
    """
    import re
    o = original.lower()
    c = compressed.lower()

    # Tone — always include, never visible in compressed message
    if re.search(
            r"sad|unsafe|scared|confused|helpless|"
            r"electricity.*shut|shut off|single parent", o):
        tone = "distressed"
    elif re.search(
            r"sorry|apologis|bother|appreciate|thank you|"
            r"genuinely|hoping|please help|a bit of a|"
            r"i hope you|don.t want to|not sure what to do|"
            r"i know you must|embarrassed|awkward", o):
        tone = "anxious"
    elif re.search(
            r"rude|unacceptable|unlawful|illegal|violation|"
            r"breach|outrageous|appalling|frustrated|"
            r"getting nowhere|running out|made it very clear|"
            r"they can not|they refuse|let down", o):
        tone = "frustrated"
    else:
        tone = "neutral"

    # Urgency — only add if NOT already visible in compressed message
    urgency_in_msg = re.search(
        r"birthday|this (saturday|sunday|weekend|week)|tomorrow|"
        r"for work|urgently|emergency|asap|"
        r"young child|baby|no heating|freezing|locked out|"
        r"shut off|single parent", c)
    urgency = None
    if not urgency_in_msg:
        if re.search(
                r"birthday|this (saturday|sunday|weekend|week)|tomorrow|"
                r"for work|work from home|urgently|deadline|emergency|"
                r"young child|baby|no heating|freezing|locked out|"
                r"shut off|really need|desperate|cannot wait|single parent|"
                r"denied|reversed|stolen|running out", o):
            urgency = "high"

    # Context — prior attempts only if NOT in compressed message
    context = None
    prior_orig = re.search(
        r"called.{0,20}(multiple|several|many|\d+ times)|"
        r"emailed.{0,20}(multiple|several|\d+ times)|"
        r"contacted.{0,20}(multiple|several|\d+ times)|"
        r"sent.{0,20}(multiple|several|letters?|emails?)|"
        r"for (over )?(two|three|\d+) months", o)
    prior_comp = re.search(
        r"called|emailed|contacted|sent|tried|months|multiple", c)
    if prior_orig and not prior_comp:
        context = prior_orig.group(0)[:35].strip()

    tag = tone
    if urgency: tag += f"|{urgency}"
    if context: tag += f" — {context}"
    return f"[{tag}]"


@dataclass
class ALEPacket:
    """Built ALE packet ready to send to frontier LLM."""
    system_prompt: str   # ALE_SYSTEM_PROMPT (for distilled) or RAW_SYSTEM_PROMPT
    user_turn:     str   # "[annotation] compressed" or "original"
    passthrough:   bool  # True = raw call, False = distilled call
    annotation:    str   # "[tone|urgency — context]" or ""
    token_est:     int   # estimated tokens for user turn


def _est_tokens(text: str) -> int:
    return max(1, round(len(text.split()) / 0.75))


def build_ale_packet(result: "DistillResult") -> ALEPacket:
    """
    Build ALE packet from a DistillResult.
    Returns ALEPacket with system_prompt and user_turn ready for API call.
    """
    if result.passthrough:
        user_turn = result.original_prompt
        return ALEPacket(
            system_prompt=RAW_SYSTEM_PROMPT,
            user_turn=user_turn,
            passthrough=True,
            annotation="",
            token_est=_est_tokens(user_turn),
        )

    compressed = result.compressed_prompt or result.original_prompt
    annotation = build_annotation(result.original_prompt, compressed)
    user_turn  = f"{annotation} {compressed}"

    return ALEPacket(
        system_prompt=ALE_SYSTEM_PROMPT,
        user_turn=user_turn,
        passthrough=False,
        annotation=annotation,
        token_est=_est_tokens(user_turn),
    )


def build_ale_messages(result: "DistillResult") -> dict:
    """
    Backward-compatible wrapper used by run_spsd_v3.py.
    Returns dict with 'messages' key containing the user message.
    """
    packet = build_ale_packet(result)
    return {
        "system":   packet.system_prompt,
        "messages": [{"role": "user", "content": packet.user_turn}],
        "passthrough": packet.passthrough,
        "annotation":  packet.annotation,
    }
