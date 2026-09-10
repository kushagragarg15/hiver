# Results -- AppleSupport

_gemini/gemini-3.1-flash-lite, judge gemini-3.1-flash-lite, n=30, 1083.4s, generated 2026-09-10T16:12:31.178540+00:00_

## Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| agent (LLM) | 0.7 | 0.501 | 0.676 |
| baseline: keyword (simple) | 0.5 | 0.341 | 0.574 |
| baseline: majority (trivial) | 0.6 | 0.094 | 0.45 |

## Escalation decision (positive class = escalate)

| system | esc-precision | esc-recall | esc-F1 | missed-esc-rate | unnec-esc-rate | auto-rate |
|---|---|---|---|---|---|---|
| agent | 1.0 | 0.273 | 0.429 | 0.727 | 0.0 | 0.9 |
| baseline: always-auto | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| baseline: always-escalate | 0.367 | 1.0 | 0.537 | 0.0 | 1.0 | 0.0 |
| baseline: intent-prior | 1.0 | 0.273 | 0.429 | 0.727 | 0.0 | 0.9 |

## Reply quality -- LLM-as-judge (n=30)

| system | groundedness | relevance | correctness_safety | tone | completeness | pass-rate |
|---|---|---|---|---|---|---|
| agent | 4.8 | 4.533 | 4.833 | 4.867 | 4.5 | 0.9 |
| baseline: retrieval-only | 4.1 | 3.833 | 4.2 | 4.5 | 3.733 | 0.7 |
| baseline: canned | 4.9 | 3.9 | 4.9 | 4.567 | 3.9 | 0.933 |

## Headline

```json
{
  "intent_macro_f1": 0.501,
  "intent_accuracy": 0.7,
  "missed_escalation_rate": 0.727,
  "unnecessary_escalation_rate": 0.0,
  "auto_rate": 0.9,
  "reply_pass_rate": 0.9
}
```

See REPORT.md for what these numbers hide.