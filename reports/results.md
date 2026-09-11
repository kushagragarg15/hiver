# Results -- AppleSupport

_gemini/gemini-3.1-flash-lite, judge gemini-3.1-flash-lite, n=233, 3158.3s, generated 2026-09-11T11:06:39.423698+00:00_

## Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| agent (LLM) | 0.691 | 0.628 | 0.711 |
| baseline: keyword (simple) | 0.433 | 0.462 | 0.512 |
| baseline: majority (trivial) | 0.472 | 0.08 | 0.303 |

## Escalation decision (positive class = escalate)

| system | esc-precision | esc-recall | esc-F1 | missed-esc-rate | unnec-esc-rate | auto-rate |
|---|---|---|---|---|---|---|
| agent | 0.787 | 0.474 | 0.592 | 0.526 | 0.065 | 0.798 |
| baseline: always-auto | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| baseline: always-escalate | 0.335 | 1.0 | 0.502 | 0.0 | 1.0 | 0.0 |
| baseline: intent-prior | 0.837 | 0.462 | 0.595 | 0.538 | 0.045 | 0.815 |

## Reply quality -- LLM-as-judge (n=233)

| system | groundedness | relevance | correctness_safety | tone | completeness | pass-rate |
|---|---|---|---|---|---|---|
| agent | 4.798 | 4.567 | 4.841 | 4.918 | 4.562 | 0.906 |
| baseline: retrieval-only | 4.099 | 3.524 | 4.236 | 4.442 | 3.545 | 0.635 |
| baseline: canned | 4.798 | 3.678 | 4.811 | 4.416 | 3.682 | 0.811 |

## Headline

```json
{
  "intent_macro_f1": 0.628,
  "intent_accuracy": 0.691,
  "missed_escalation_rate": 0.526,
  "unnecessary_escalation_rate": 0.065,
  "auto_rate": 0.798,
  "reply_pass_rate": 0.906
}
```

See REPORT.md for what these numbers hide.