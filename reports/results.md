# Results -- AppleSupport

_groq/openai/gpt-oss-120b, judge None, n=150, 48.0s, generated 2026-09-10T18:10:49.105245+00:00_

## Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| agent (LLM) | 0.693 | 0.646 | 0.712 |
| baseline: keyword (simple) | 0.453 | 0.491 | 0.524 |
| baseline: majority (trivial) | 0.487 | 0.082 | 0.319 |

## Escalation decision (positive class = escalate)

| system | esc-precision | esc-recall | esc-F1 | missed-esc-rate | unnec-esc-rate | auto-rate |
|---|---|---|---|---|---|---|
| agent | 0.737 | 0.528 | 0.615 | 0.472 | 0.103 | 0.747 |
| baseline: always-auto | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 1.0 |
| baseline: always-escalate | 0.353 | 1.0 | 0.522 | 0.0 | 1.0 | 0.0 |
| baseline: intent-prior | 0.867 | 0.491 | 0.627 | 0.509 | 0.041 | 0.8 |

## Headline

```json
{
  "intent_macro_f1": 0.646,
  "intent_accuracy": 0.693,
  "missed_escalation_rate": 0.472,
  "unnecessary_escalation_rate": 0.103,
  "auto_rate": 0.747,
  "reply_pass_rate": null
}
```

See REPORT.md for what these numbers hide.