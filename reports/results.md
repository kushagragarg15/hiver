# Results -- AppleSupport

_ollama/llama3.1:8b, judge llama3.1:8b, n=30, 13.6s, generated 2026-09-10T15:36:39.054719+00:00_

## Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| agent (LLM) | 0.367 | 0.315 | 0.397 |
| baseline: keyword (simple) | 0.5 | 0.341 | 0.574 |
| baseline: majority (trivial) | 0.6 | 0.094 | 0.45 |

## Escalation decision (positive class = escalate)

| system | esc-precision | esc-recall | esc-F1 | missed-esc-rate | unnec-esc-rate | auto-rate |
|---|---|---|---|---|---|---|
| agent | 0.875 | 0.636 | 0.737 | 0.364 | 0.053 | 0.733 |
| baseline: always-auto | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| baseline: always-escalate | 0.367 | 1.0 | 0.537 | 0.0 | 1.0 | 0.0 |
| baseline: intent-prior | 0.857 | 0.545 | 0.667 | 0.455 | 0.053 | 0.767 |

## Reply quality -- LLM-as-judge (n=30)

| system | groundedness | relevance | correctness_safety | tone | completeness | pass-rate |
|---|---|---|---|---|---|---|
| agent | 3.267 | 4.133 | 4.467 | 4.767 | 3.367 | 0.6 |
| baseline: retrieval-only | 2.233 | 3.0 | 3.1 | 4.4 | 2.167 | 0.1 |
| baseline: canned | 2.033 | 2.533 | 3.933 | 4.6 | 1.8 | 0.033 |

## Headline

```json
{
  "intent_macro_f1": 0.315,
  "intent_accuracy": 0.367,
  "missed_escalation_rate": 0.364,
  "unnecessary_escalation_rate": 0.053,
  "auto_rate": 0.733,
  "reply_pass_rate": 0.6
}
```

See REPORT.md for what these numbers hide.