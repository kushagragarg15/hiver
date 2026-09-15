# Golden evaluation set — how it was sampled and labelled

## What it is
`golden_set.jsonl` — 150–250 inbound customer messages for the chosen brand,
each hand-labelled with:

| field | meaning |
|---|---|
| `message` | the customer's opening message (the thing the agent sees) |
| `thread_context` | the full reconstructed conversation, for the labeller's reference |
| `reference_resolution` | what the brand actually said/did later in that thread |
| `gold_intent` | one of the 8 slugs in `src/intents.py` |
| `gold_action` | `auto` or `escalate` — should this have been handled without a human? |
| `gold_action_reason` | one line: why |
| `notes` | anything ambiguous |
| `label_source` | `human` (labelled/confirmed by a person) · `model_assisted` (drafted by an LLM, awaiting human review) · `heuristic_prelabel` (keyword guess only) |

## Current state (be honest about this)
233 rows. **169 are `human`** — 30 hand-labelled from scratch (the original
seed) plus 139 model-assisted drafts that have since been walked in
`python -m eval.label_tool --review` and confirmed or corrected (12 changed:
10 intents, 4 actions). **64 are still `model_assisted`** — drafted by
applying the protocol below and checked once, but not yet independently
reviewed. `--review` walks whatever is left and flips each row to `human`.
Every results table in `REPORT.md` notes which mix it was computed on.
Model-assisted labelling is allowed ("you may use AI coding assistants
freely") — not disclosing it would not be.

## Sampling (see `scripts/make_golden_candidates.py`)
1. **Source = held-out only.** Candidates are drawn exclusively from
   `{brand}_eval_pool.jsonl` — threads whose first message is on/after
   `split_date` (2017-11-15). The agent's retrieval memory is built only from
   threads *before* that date, so no golden message can be answered by
   retrieving its own thread.
2. **Stratify by intent guess × conversation length.** A keyword weak-labeller
   assigns a rough intent; length is bucketed 1 / 2–3 / 4+ customer turns.
   Every non-empty stratum gets a floor of 4; the rest is allocated roughly
   proportionally. This stops the head intent (usually update/hardware) from
   swamping rare-but-critical ones (security, billing, complaint).
3. **Quality gate.** Drop openings under 4 words, link-only / emoji-only tweets,
   and anything over 500 chars.
4. Target 240 candidates; label 150–250.

## Labelling protocol (see `eval/label_tool.py`)
- One labeller (the author). ~10–15 s per easy row, up to a minute for
  ambiguous ones.
- **Intent** = the customer's *primary* need. Tie-breakers, in order:
  security/identity → billing/money → the concrete technical bucket →
  complaint (only if there is no concrete ask) → praise/non-actionable.
- **`gold_action` answers: *should a good automated agent have handled this
  unsupervised?*** — NOT "what did AppleSupport actually do". This distinction
  matters for AppleSupport specifically: ~65% of its 2017 replies were "join us
  in DM" regardless of how routine the question was (it used Twitter as an
  intake funnel, not a resolution channel). Labelling `gold_action` by Apple's
  behaviour would make almost everything `escalate` and the task degenerate. So:
  - `auto` = there is a safe, standard, publicly shareable answer a competent
    bot could give now: known-bug workarounds (reset keyboard dictionary, update
    to 11.1.2), standard triage questions, documented how-tos, acknowledging
    feedback.
  - `escalate` = needs account / payment / identity verification; hardware
    inspection or replacement; legal / abuse / safety; the customer is too
    distressed or hostile for a bot; or it's genuinely novel with no known fix.
- Consequence, stated loudly in `REPORT.md` §5: these labels encode *my* view of
  what is safely automatable, which is **more generous than Apple's actual
  2017 behaviour**. A model that imitated Apple would escalate far more and
  score worse against this golden set — and a model that is too eager would look
  good here while being unsafe in production.
- `reference_resolution` is shown to the labeller for context but is frequently
  just "DM us"; the label is a judgement, not a copy of it.
- Pre-labels from the heuristic are shown but overridden freely; the recorded
  human/heuristic agreement rate is reported in `REPORT.md`.

## Known limitations
- Single annotator → no inter-annotator agreement number. Mitigation: a second
  pass over all `escalate`/`auto` calls (`--review-actions`) a day later; the
  self-agreement rate is reported.
- Labels use hindsight the live agent doesn't have (it can't see
  `reference_resolution`). This makes `gold_action` a slightly *harder* target
  than a real-time human would be held to — noted in the "misleading headline"
  section.
