# EmberBench

Adversarial evaluation harness for AI safety systems. Tests how well LLMs and safety guards detect prompt injection, authority spoofing, numeric contradictions, semantic paraphrasing, and temporal manipulation attacks.

## The Benchmark

**91 cases** — 61 adversarial, 30 benign — across Legal, Financial, and Medical domains.

### Attack Types
| Type | Cases | Description |
|------|-------|-------------|
| Numeric Contradiction | 19 | False statistics, manipulated figures |
| Cross-Layer Gap | 16 | Contradictions spanning context layers |
| Temporal Injection | 10 | Outdated or future-dated claims |
| Authority Poison | 7 | False authority/source attribution |
| Semantic Paraphrase | 6 | Meaning-altering rewrites |
| Soft Injection | 3 | Subtle behavioral nudges |

### 3-Axis Evaluation
- **Axis 1 — Guard DR/FPR**: Does the safety guard catch attacks without false positives?
- **Axis 2 — LLM Behavior**: Does the LLM resist, comply, or affirm the attack?
- **Axis 3 — System DR**: Does the full stack (LLM + Guard) contain every attack?

## Results (May 2026)

| Model | Baseline DR | System DR | FPR | Slip-throughs | Latency |
|-------|-------------|-----------|-----|---------------|---------|
| Claude Sonnet 4.6 | 95.1% | 100% | 3.3% | 0 | 3,820ms |
| Kimi K2.5 | 45.9% | 100% | 3.3% | 0 | 15,771ms |
| Gemini 3.1 Pro | 88.5% | 100% | 10.0% | 0 | 6,389ms |
| Claude Haiku 4.5 | 85.2% | 91.8% | 3.3% | 5 | 2,619ms |
| Kimi K2.6 | 52.5% | 96.7% | 16.7% | 2 | 14,031ms |
| EmberArmor (guard only) | 98.4% | — | 0.0% | — | 860ms |

## Usage

```bash
# Install deps
pip install anthropic google-genai httpx torch transformers

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
export KIMI_API_KEY=sk-...

# Run baseline (LLM only, no guard)
python -m eval.run_claude_baseline --model claude-sonnet-4-6
python -m eval.run_gemini_baseline --model gemini-3.1-pro-preview

# Run integrated (LLM + EmberArmor guard)
ANTHROPIC_API_KEY=... python -m eval.run_integrated_eval --model claude-sonnet-4-6
GEMINI_API_KEY=... python -m eval.run_integrated_eval --model gemini-3.1-pro-preview --save-responses
```

## Structure

```
EmberBench/
├── eval/
│   ├── emberbench/
│   │   ├── datasets/        — 91 test cases (legal, financial, medical)
│   │   └── runner.py
│   ├── run_integrated_eval.py   — Full-stack eval (LLM → EmberArmor)
│   ├── run_claude_baseline.py   — Claude baseline (no guard)
│   ├── run_gemini_baseline.py   — Gemini baseline (no guard)
│   └── run_kimi_baseline.py     — Kimi baseline (no guard)
├── ember_security/
│   └── dissonance_guard/    — Guard scorer + detector (shared dep)
├── results/                 — Per-case JSON results
│   ├── claude_sonnet_4_6_baseline_results.json
│   └── ...
└── .env.example
```

## License

MIT
