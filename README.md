# TechJam 2026: Conversational Shopping Assistant

An AI-powered conversational search agent for e-commerce, built for the TechJam 2026 Conversational Search Challenge. The agent helps shoppers find products through natural, multi-turn conversation — detecting whether someone is "browsing" or "buying," asking smart clarifying questions when needed, and combining keyword search, semantic search, and LLM-based ranking to surface the most relevant products.

## Overview

Traditional e-commerce search relies on rigid keyword matching, which fails to capture the difference between a shopper who knows exactly what they want and one who is casually exploring. This project addresses that gap with a four-part architecture:

1. **Core Architecture** — Rule-based intent detection (Buying vs. Browsing) routes each turn through a hybrid retrieval pipeline combining BM25 keyword search and dense embedding search, weighted according to detected intent.
2. **Dialog Strategy** — A per-session slot-tracking state machine accumulates shopper preferences across turns, detects when a shopper changes their mind (intent override), and asks targeted clarifying questions when the search space is too broad — capped at a budget to avoid stalling the conversation.
3. **Self-Evolution** — After each non-clarifying turn, an LLM call both ranks the candidate products and distills a short summary of the shopper's inferred preferences, which is fed back into the next ranking call as contextual grounding.
4. **Evaluation** — Performance is measured against the organizer's local evaluator using Hit Rate@10, Mean Reciprocal Rank (MRR), and Mean Turns to Conversion (MTTC), combined into a single `TechnicalScore`.

## Results

Measured on the 200-sample public development set, compared against the unmodified starter agent (pure BM25, no state, no clarification):


| Metric            | Baseline | This Solution | Improvement |
| ----------------- | -------- | ------------- | ----------- |
| Hit Rate@10       | 0.125    | 0.340         | 2.7x        |
| MRR               | 0.068    | 0.242         | 3.5x        |
| MTTC (avg. turns) | 9.81     | 8.20          | ↓ 1.6 turns |
| TechnicalScore    | 0.107    | 0.299         | 2.8x        |


Every metric improved across every one of the four evaluation scenario types (buying, browsing, intent_override, boundary), with no regressions anywhere. The largest gains came in the "browsing" scenario category (Hit Rate@10 improved roughly 16x), which is expected: pure keyword matching structurally cannot recognize semantic similarity (e.g., "sneakers" vs. "running shoes"), while the hybrid embedding-based retrieval can.

Notably, building conversation state and clarification logic (Pillar II) on top of the existing BM25 baseline alone raised the technical score from 0.107 to roughly 0.295 — the large majority of our total improvement. Adding hybrid retrieval and LLM-based ranking (Pillars I and III) contributed a further, smaller gain to 0.299. This matches the organizer's own recommended build order (state and clarification before retrieval sophistication), and is worth noting honestly: the highest-leverage work was fixing the conversation loop, not the retrieval pipeline.

## Architecture Details

### Pillar I — Core Architecture

- **Intent detection** is rule-based: messages containing explicit, extractable constraints (material, color, size, budget, etc.) or substantive non-generic content are classified as "buying"; vague or exploratory language, or the evaluator's own generic filler replies (e.g., "I don't have an additional preference for X"), default to "browsing."
- **Retrieval** always runs both BM25 (SQLite FTS5, weighted by field: title > categories > features > details/store > description) and dense embedding search (`all-MiniLM-L6-v2` via `sentence-transformers`, cosine similarity) on every turn.
- **Blending** is intent-weighted rather than a hard either/or switch, with one adjustment made based on direct testing: after confirming that embeddings measurably hurt precision on buying-intent queries (MRR dropped from 0.214 to 0.206 when blending vs. pure BM25), buying-intent queries use BM25 exclusively (weight 1.0/0.0), while browsing-intent queries use a 40/60 BM25/embedding blend. The browsing-side split was set based on initial reasoning and has not yet been further tuned — see Limitations.
- **LLM ranking** (Claude Haiku, `claude-haiku-4-5-20251001`) re-ranks the top candidates from the hybrid blend and produces a one-sentence preference summary, used by Pillar III. This step is skipped entirely on turns where the agent is asking a clarifying question, since ranking is not needed until a final recommendation is returned — confirmed via testing that this roughly halves LLM API calls per session with no change to Hit Rate@10 or MTTC.

### Pillar II — Dialog Strategy

- **Slots** are tracked per session, aligned exactly to the evaluator's own `ALLOWED_ATTRIBUTES` set (`category, material, color, size, style, brand, budget, feature, use_case`) — extracted directly from the evaluator's source code to ensure our `ask_attribute` choices are ones the simulated shopper can actually answer.
- **Accumulation vs. override** is detected via explicit trigger phrases (e.g., "actually," "never mind," "ignore my earlier preference" — matching the evaluator's own override message format) combined with a category-clash check; on override, all slots are wiped and rebuilt from the new message.
- **Over-generality** triggers a clarifying question when either too many slots remain empty (≥6 of 9) or the candidate pool is too large (>500 matches) — whichever fires first — capped at 5 clarifying questions per session to guarantee turns remain for an actual recommendation. This budget was set deliberately generous rather than minimal, since the scoring formula (see Pillar IV) penalizes a total miss more heavily than a slow success.
- The agent tracks which attributes it has already asked about per session, so it never repeats a question the simulated shopper has already declined to answer — an earlier version of this logic had a bug that caused exactly this repetition, which we found and fixed by debugging against real evaluator transcripts.

### Pillar III — Self-Evolution

- After each ranking call, Claude Haiku returns both a ranked candidate list and a short natural-language summary of the shopper's inferred preferences.
- That summary is stored per session and passed back into the next ranking call's prompt as light contextual grounding ("What we know about this shopper so far: ..."). This was tested against the version without it and found to be measurably neutral (no significant change to Hit Rate@10, MRR, or MTTC) — we kept it regardless, since it satisfies the architectural requirement and carries no measured downside.
- A long-term, cross-session user profile was designed but not implemented, since the evaluation harness treats every session as an isolated, single-user interaction with no continuity between sessions — see Limitations.

### Pillar IV — Evaluation

Scoring formula (extracted directly from `evaluator/local_evaluator.py`):

```
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 − MTTC) / 10, 0, 1)
```

Because Hit Rate@10 is weighted most heavily and a total miss is scored worse than the slowest possible success (MTTC = 11 vs. a max of 10), the system is deliberately tuned to favor accuracy over speed — e.g., the clarification budget was set generously (5, not 1–2) once this weighting became clear.

## Setup and Installation

### Prerequisites

- Python 3.13+
- An [Anthropic API key](https://console.anthropic.com) (used for LLM ranking — a small amount of billing credit is required; see Cost, Latency & Token Usage Disclosure below for actual measured cost)

### 1. Clone the repository

bash

```bash
git clone <your-fork-url>
cd techjam-conversational-search
```

### 2. Install dependencies

bash

```bash
pip3 install sentence-transformers anthropic --break-system-packages
```

### 3. Download the participant kit

Download `catalog.jsonl.gz` from the [Releases page](https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit), decompress it, and place it at `data/catalog.jsonl`.

### 4. Set your Anthropic API key as an environment variable

bash

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

To persist this across terminal sessions:

bash

```bash
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.zshrc
source ~/.zshrc
```

**Do not** hardcode the key in any file — it is read exclusively from this environment variable and is never written to disk or committed to this repository.

## Reproducing Results

### Full evaluation (200 sessions)

bash

```bash
python3 -m evaluator.local_evaluator
```

The evaluator prints live progress (`Loading agent for 200 session(s)...`, then `Evaluating session 12/200 (sample_id)...`) so a run's status is never ambiguous.

Note: the first run will download the embedding model (~80MB) and compute embeddings for all 50,000 catalog products, caching them to `data/embeddings_cache.pkl` (excluded from version control — it is regenerated automatically). Subsequent runs load from this cache and start much faster. If you ever change the catalog or the text fields used for embeddings, delete `data/embeddings_cache.pkl` to force a rebuild.

A full run takes approximately 20 minutes, primarily bound by sequential Claude Haiku API calls (see disclosure below).

### Fast smoke-testing on a subset

bash

```bash
python3 -m evaluator.local_evaluator --limit 20
```

Runs the identical evaluation logic on only the first N sessions — useful for quickly checking that a change hasn't broken anything, without waiting for the full 200-session run. **Note:** results from a small subset are not directly comparable to the full run's numbers due to sample size and scenario-distribution differences; use this only to catch major regressions, not to fine-tune small changes.

### Quick single-message test

bash

```bash
python3 demo_single_turn.py
```

Sends one message through the agent and prints the response directly — useful for fast manual sanity checks outside the evaluation harness.

## Cost, Latency & Token Usage Disclosure

Measured on a full run of the 200-sample public development set (Claude Haiku, `claude-haiku-4-5-20251001`):


| Metric                           | Value       |
| -------------------------------- | ----------- |
| Total wall-clock time            | ~20 minutes |
| Average time per session         | ~6 seconds  |
| Total prompt (input) tokens      | 394,731     |
| Total completion (output) tokens | 107,332     |
| Total tokens                     | 502,063     |
| Average tokens per session       | ~2,510      |


**Note on LLM call frequency:** the LLM ranking step is skipped entirely on turns where the agent returns a clarifying question rather than a final recommendation, so actual LLM calls per session are fewer than the maximum 10 turns — this is why total token usage is lower than a naive "one call per turn" estimate would suggest.

**Estimated cost:** based on Claude Haiku's published per-token pricing and this run's measured token usage (502,063 total tokens), a full 200-session run costs approximately $0.50–$1.00 (exact rate depends on current Haiku pricing — see [Anthropic's pricing page](https://www.anthropic.com/pricing) for authoritative figures). Note: our total account spend shown in the Anthropic Console reflects cumulative usage across all development, testing, and debugging sessions this evening, not this single run in isolation, so it is not reported here as a per-run figure.

**Extrapolated to the 800-session private evaluation set:** assuming similar session characteristics to the public set, we estimate approximately 80 minutes of runtime and roughly 2,000,000 total tokens — actual figures will vary based on the specific distribution of scenario types and conversation lengths in the private set.

## Limitations and Future Improvements

- **Slot extraction vocabulary is fixed and incomplete.** Attributes like "buckle closure" or "water resistant" fall outside our fixed word lists (material, color, size, etc.) and are not captured as structured slots, even though they represent real constraints. A more general extraction approach (e.g., using the LLM itself to extract constraints) would likely improve slot coverage.
- **Buying-intent queries underperform browsing-intent queries in absolute terms** (Hit Rate@10: 0.325 vs. 0.4125; MRR: 0.206 vs. 0.297), despite buying-intent shoppers providing more explicit information. We believe this is connected to the slot-extraction gap above: BM25's simple keyword-OR matching does not fully exploit rich, specific detail the way a more complete extraction and filtering approach could. This is a concrete target for further improvement.
- **The browsing-intent blend weight (40% BM25 / 60% embeddings) was set by initial reasoning, not systematically tuned.** Unlike the buying-intent decision (which was validated by directly testing pure BM25 against a blend), we did not test alternative browsing-side splits. A grid search over this weight is a natural next step.
- **LLM ranking currently provides a marginal, roughly neutral effect on TechnicalScore** compared to the pure retrieval blend without any LLM step. We attempted a more detailed prompt with explicit ranking criteria and additional shopper context, which measurably regressed performance (Hit Rate@10 dropped from 0.34 to 0.31), likely because the smaller/faster model was distracted by additional instructions rather than helped by them; we reverted to a simpler prompt. Given more time, a larger model or a more carefully validated prompt could likely extract more value from this step.
- **The long-term, cross-session user profile (Pillar III)** is architecturally unimplemented, since the evaluation harness treats each of the 200/800 sessions as an independent, isolated shopper with no continuity — there is currently no scenario in which this feature would be exercised by the evaluator. The short-term, within-session preference summary is implemented and feeds into ranking, though measured to have a neutral effect (see above).
- **Over-generality thresholds** (6 empty slots, 500 candidates) were set based on reasoning and single-pass testing, not systematic hyperparameter search. There is likely additional score available from more rigorous tuning.

## Development Tools and Resources

- **Language/Runtime:** Python 3.13
- **Retrieval:** SQLite FTS5 (BM25), `sentence-transformers` (`all-MiniLM-L6-v2`) for dense embeddings
- **LLM:** Anthropic Claude Haiku (`claude-haiku-4-5-20251001`), via the official `anthropic` Python SDK
- **Dataset:** TechJam 2026 frozen catalog (50,000 products, Amazon Reviews 2023 Clothing/Shoes/Jewelry category) and the organizer's 200-sample public development set
- **Development environment:** Cursor (editor + terminal), GitHub Desktop (version control)

## Team Contributions

Solo submission — architecture design, implementation, testing, debugging, and documentation all completed by Harshitha SureshKumar.

