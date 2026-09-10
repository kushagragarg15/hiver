> ⚠️ **provider = `mock` — these numbers are a wiring check, not a quality signal.** Run `make eval` on a real backend (ollama/openai) to replace them.

# Results -- AppleSupport

_mock/mock, judge mock, n=30, 17.9s, generated 2026-09-10T14:28:31.654312+00:00_

## Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| agent (LLM) | 0.5 | 0.346 | 0.538 |
| baseline: keyword (simple) | 0.5 | 0.341 | 0.574 |
| baseline: majority (trivial) | 0.6 | 0.094 | 0.45 |

## Escalation decision (positive class = escalate)

| system | esc-precision | esc-recall | esc-F1 | missed-esc-rate | unnec-esc-rate | auto-rate |
|---|---|---|---|---|---|---|
| agent | 1.0 | 0.091 | 0.167 | 0.909 | 0.0 | 0.967 |
| baseline: always-auto | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| baseline: always-escalate | 0.367 | 1.0 | 0.537 | 0.0 | 1.0 | 0.0 |
| baseline: intent-prior | 1.0 | 0.091 | 0.167 | 0.909 | 0.0 | 0.967 |

## Reply quality -- LLM-as-judge (n=30)

| system | groundedness | relevance | correctness_safety | tone | completeness | pass-rate |
|---|---|---|---|---|---|---|
| agent | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 | 1.0 |
| baseline: retrieval-only | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 | 1.0 |
| baseline: canned | 4.0 | 4.0 | 4.0 | 4.0 | 3.0 | 1.0 |

## Headline

```json
{
  "intent_macro_f1": 0.346,
  "intent_accuracy": 0.5,
  "missed_escalation_rate": 0.909,
  "unnecessary_escalation_rate": 0.0,
  "auto_rate": 0.967,
  "reply_pass_rate": 1.0
}
```

See REPORT.md for what these numbers hide.