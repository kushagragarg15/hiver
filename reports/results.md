# Results -- AppleSupport

_gemini/gemini-3.1-flash-lite, judge gemini-3.1-flash-lite, n=233, 132.5s, generated 2026-09-14T21:16:00.879764+00:00_

## Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| agent (LLM) | 0.674 | 0.602 | 0.693 |
| baseline: keyword (simple) | 0.416 | 0.429 | 0.493 |
| baseline: majority (trivial) | 0.464 | 0.079 | 0.294 |

## Escalation decision (positive class = escalate)

| system | esc-precision | esc-recall | esc-F1 | missed-esc-rate | unnec-esc-rate | auto-rate |
|---|---|---|---|---|---|---|
| agent | 0.745 | 0.449 | 0.56 | 0.551 | 0.077 | 0.798 |
| baseline: always-auto | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| baseline: always-escalate | 0.335 | 1.0 | 0.502 | 0.0 | 1.0 | 0.0 |
| baseline: intent-prior | 0.791 | 0.436 | 0.562 | 0.564 | 0.058 | 0.815 |

## Reply quality -- LLM-as-judge (n=233)

| system | groundedness | relevance | correctness_safety | tone | completeness | pass-rate |
|---|---|---|---|---|---|---|
| agent | 4.798 | 4.567 | 4.841 | 4.918 | 4.562 | 0.906 |
| baseline: retrieval-only | 4.099 | 3.524 | 4.236 | 4.442 | 3.545 | 0.635 |
| baseline: canned | 4.798 | 3.678 | 4.811 | 4.416 | 3.682 | 0.811 |

## Headline

```json
{
  "intent_macro_f1": 0.602,
  "intent_accuracy": 0.674,
  "missed_escalation_rate": 0.551,
  "unnecessary_escalation_rate": 0.077,
  "auto_rate": 0.798,
  "reply_pass_rate": 0.906
}
```

See REPORT.md for what these numbers hide.