"""AI customer-support agent for a single Twitter brand.

Modules
-------
config      load config.yaml
data_prep   twcs.csv -> per-brand reconstructed threads + chronological split
intents     the intent taxonomy (defined from the data) + a keyword weak-labeller
llm         provider abstraction (ollama / openai) with an on-disk response cache
retrieve    BM25 over the brand's historical resolved threads
classify    intent classification (LLM + baselines)
draft       grounded reply drafting
escalate    auto-handle vs escalate decision, with a stated reason
agent       orchestrates classify -> retrieve -> draft -> escalate
pipeline    run the agent over a jsonl of messages
"""

__version__ = "0.1.0"
