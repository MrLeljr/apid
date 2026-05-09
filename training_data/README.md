# APID Training Data

Dataset version: `1.1.0`

`prompt_injection_dataset.json` is the default local training set for `PromptInjectionScanner`. It intentionally stays small enough for fast local tests but now includes:

- benign product, coding, writing, and safety-education prompts
- benign counterexamples that mention prompt-injection vocabulary without malicious intent
- public-benchmark-inspired rows from JailbreakBench, HarmBench, WildJailbreak, and Lakera Gandalf patterns
- synthetic prompt-injection attacks covering direct override, roleplay, system spoofing, indirect injection, obfuscation, and secret exfiltration

Records use this schema:

```json
{
  "text": "Prompt text",
  "label": 0,
  "source": "source_or_generation_family",
  "split": "train",
  "attack_type": "optional_attack_family"
}
```

Labels are binary: `0` means allow/benign, and `1` means prompt injection or jailbreak attempt. The public dataset references are in `public_sources.json`; examples marked `*_inspired` or `*_style` are local defensive test/training rows shaped after public benchmark categories, not bulk copies of those corpora.

Benchmark fixtures live under `training_data/benchmarks/` and are kept out of training so evals can catch regressions.
