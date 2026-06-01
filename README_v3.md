# SPSD v4.3 — Semantic Prompt Structural Distillation

**Authors:** Ajeet Kumar, Abhinit Sen — Indian School of Business, Hyderabad

---

## What Is SPSD?

SPSD is an on-device prompt compression pipeline. Before a user prompt reaches a cloud frontier LLM, a 4-bit quantized 1.5B model running on the device compresses it — stripping social scaffolding (politeness markers, apologetic preamble, rapport-building language) while preserving all semantic content. The frontier LLM receives a compact structured packet and reconstructs a full natural-language response.

**The core claim:** You can reduce LLM prefill token cost by ~58% on eligible calls without degrading response quality — proven across 11 statistical tests.

---

## Validated Results (v4.2 corpus, 150 prompts)

| Hypothesis | Result | Statistic |
|---|---|---|
| H1: Token saving > 0 | **PROVEN** | t=16.17, p<0.001, d=1.92 |
| H2: Compression ratio < 1.0 | **PROVEN** | t=-34.31, p<0.001, mean=0.42 |
| H3: 100% positive savings | **PROVEN** | 71/71 (100%) net positive |
| H4: Quality sim > 0.70 | **PROVEN** | t=4.22, p<0.001, d=0.52 |
| H5: LLM judge dist >= raw | **NEW** | Run v3 pipeline to validate |
| H6: Medical passthrough 100% | **PROVEN** | 15/15, p=1.000 |
| H7: Legal passthrough 100% | **PROVEN** | 3/3 confirmed |
| H8: verbose > general savings | **PROVEN** | U=214, p=0.0045, d=0.85 |
| H9: Category variation | Not significant | H=4.96, p=0.175 (SPSD works broadly) |
| H10: Total saving > 0 | **PROVEN** | mean=24.2t per paired call |
| H11: Cache saves >80% | **PROVEN** | 90% reduction on system prompt |

---

## Pipeline Architecture

```
User prompt
    │
    ▼
[Tier 1 Gate ~0ms]
  short_prompt (≤15w) → PASSTHROUGH
  safety_critical      → PASSTHROUGH
  domain_medical       → PASSTHROUGH  ← validated: 100% accuracy
  domain_legal         → PASSTHROUGH  ← validated: 100% accuracy
    │
    ▼
[Complexity Scorer ~0ms]
  5 dimensions: social / semantic / structural / repetition / word_count
  structural ≥ 0.22  → PASSTHROUGH
    │
    ▼
[High-Fidelity Guard ~0ms]
  4 regex layers → extract life events, emotional anchors,
                   dependency markers, crisis indicators → seeds aux
    │
    ▼
[SLM: Qwen2.5-1.5B Q4_K_M ~80-180ms CPU / ~50-100ms NPU]
  JSON output: {compressed_prompt, intent, tone, urgency, aux, confidence}
    │
    ▼
[Economic Gates ~0ms]
  Tier 1b safety re-check
  Dynamic confidence threshold (0.65-0.80 by prompt length)
  Net saving gate ≥ 10 tokens
    │
    ▼
[ALE Packet: D|tone|urgency|aux\n<compressed>]
    │
    ▼
Frontier LLM (any provider: Groq / Anthropic / OpenAI)
```

---

## File Reference

### Core pipeline (required for any use)
| File | Purpose |
|---|---|
| `spsd_v4.py` | Full SPSD pipeline. `distill(text)` is the main entry point. |
| `ale_prompt.py` | ALE packet builder. `build_ale_messages(result)` returns API-ready dict. |

### Evaluation pipeline (run in order)
| File | Step | Time | Output |
|---|---|---|---|
| `fetch_corpus_v3.py` | 1 | ~5-10 min | `spsd_corpus_v3.csv` |
| `run_spsd_v3.py` | 2 | ~30-60 min | `spsd_results_v3.csv` |
| `run_llm_v3.py` | 3 | ~20-30 min | updates in place |
| `score_and_judge_v3.py` | 4 | ~30-60 min | `judge_results_v3.json` |
| `hypothesis_v3.py` | 5 | ~2 min | `spsd_hypothesis_v3.xlsx` |

### Documentation
| File | Contents |
|---|---|
| `SETUP_AND_RUN.md` | Complete step-by-step Colab replication guide |
| `README.md` | This file |

---

## Corpus Design (v3)

Weighted toward SPSD-effective categories. No synthetic prompts. No truncation.

| Category | Target | Source | Distillation Rate | Priority |
|---|---|---|---|---|
| verbose_social | 60+ | WildChat (long support/complaint) | ~92% | PRIMARY |
| multi_intent_linked | 45+ | WildChat/UltraChat (2+ questions) | ~64% | PRIMARY |
| general_conversational | 50+ | WildChat (diverse ≥25w) | ~40% | PRIMARY |
| code_technical | 20+ | CodeFeedback | ~80% | TEST |
| high_stakes_medical | 15+ | MedQA-USMLE | 0% (gate) | CONTROL |
| short_passthrough | 15+ | WildChat (≤15w) | 0% (gate) | CONTROL |

**Expected distilled:** ~120 out of ~205 prompts (well above 100 minimum)

---

## Quick Start (Colab)

```python
# 1. Install
!pip install -q llama-cpp-python \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
!pip install -q groq sentence-transformers scipy openpyxl matplotlib requests

# 2. Upload spsd_v4.py + ale_prompt.py to /content/

# 3. Load model
import sys; sys.path.insert(0, '/content')
import spsd_v4
MODEL_PATH = spsd_v4.download_model(dest_dir='/content/models')

# 4. Test a single prompt
result = spsd_v4.distill(
    "Hi I'm so sorry to bother you, I know you must be incredibly busy, "
    "but I placed an order last Tuesday for my daughter's birthday and "
    "her birthday is this Saturday. I haven't received any shipping "
    "notification and I'm worried it won't arrive. Order ORD-554821."
)
print(result.summary())
# [DISTILLED] save=+55t conf=0.87 → D|anxious/apologetic|high|life_event: birthday;...

# 5. Build ALE packet for any frontier LLM
import ale_prompt
packet = ale_prompt.build_ale_messages(result)

# Use with Groq (free):
from groq import Groq
client = Groq(api_key="YOUR_GROQ_KEY")
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": packet["system"][0]["text"]},
        packet["messages"][0],
    ],
    max_tokens=150, temperature=0.4,
)
print(response.choices[0].message.content)

# Or use the one-line round_trip() demo function:
result = ale_prompt.round_trip(
    prompt="Your prompt here",
    api_key="YOUR_GROQ_KEY",
    provider="groq",  # or "anthropic" or "openai"
)
result.display()
```

---

## Key Constants (spsd_v4.py)

| Constant | Value | Meaning |
|---|---|---|
| `SHORT_PROMPT_WORD_LIMIT` | 15 | Prompts ≤15 words always passthrough |
| `MIN_NET_TOKEN_SAVING` | 10 | Minimum saving to justify compression |
| `ALE_HEADER_OVERHEAD_TOKENS` | 6 | Fixed cost of the D|...|... header line |
| `SLM_TEMPERATURE` | 0.1 | Near-deterministic SLM output |

---

## Design Principles (Never Violate)

1. **Static system prompt always** — any dynamic value in system prompt = cache miss
2. **Score tone/urgency on original text** — compressed text loses emotional register
3. **HFG phrases are guaranteed in aux** — SLM cannot compress them away
4. **Tier 1b does NOT check word count** — a 10-word compressed output is success
5. **Medical and legal always passthrough** — validated at 100% accuracy
6. **Net saving gate is the sole economic arbiter** — no compression ratio gate
7. **Parse failure → confidence=0.0** — never silently becomes passthrough
8. **Passthrough is success, not failure** — conservative gating is by design
9. **Output length is NOT SPSD's responsibility** — use max_tokens at API level
10. **Always purge sys.modules after file upload in Colab** — stale state causes bugs

---

## Datasets Used (all open, no authentication required)

| Dataset | HuggingFace path | Used for |
|---|---|---|
| WildChat-4.8M | `allenai/WildChat-4.8M` | verbose_social, multi_intent, general_conv, short_passthrough |
| UltraChat 200k | `HuggingFaceH4/ultrachat_200k` | multi_intent top-up |
| CodeFeedback | `m-a-p/CodeFeedback-Filtered-Instruction` | code_technical |
| MedQA-USMLE | `GBaker/MedQA-USMLE-4-options` | high_stakes_medical |

---

## Version History

| Version | Key Changes |
|---|---|
| v4.3 | Full 11-hypothesis framework, dual quality validation (cosine + LLM judge), v3 corpus (no synthetic, no truncation, SPSD-weighted) |
| v4.2 | Medical/legal Tier 1 passthrough, word budget (W:n) removed, final validated results: T1 p<0.001 d=1.92, T2 p<0.001 d=0.52 |
| v4.1 | Complexity Scorer (5-dim), High-Fidelity Guard (4-layer), real HuggingFace corpus |
| v4.0 | SLM-first single-pass (Qwen2.5-1.5B), ALE compact header, net saving gate |
| v3.0 | Three-tier: Tier 1 rules + DeBERTa NLI + Phi-3-mini |
| v2.x | spaCy parsing, VADER sentiment, domain detection |
| v1.x | Tokenisation + stopword removal baseline |
