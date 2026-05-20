# EmberBench

**Adversarial evaluation harness for AI safety systems.** Tests how well LLMs and safety guards detect prompt injection, authority spoofing, numeric contradictions, semantic paraphrasing, and temporal manipulation attacks — across Legal, Financial, and Medical domains.

[![Tests](https://img.shields.io/badge/test%20cases-91-blue)](data/)
[![Models](https://img.shields.io/badge/models%20tested-5-purple)](results/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Benchmark

**91 cases** — 61 adversarial, 30 benign — across 3 high-stakes deployment domains.

### Attack Types
| Type | Cases | Description |
|------|-------|-------------|
| Numeric Contradiction | 19 | False statistics, manipulated figures, basis point manipulation |
| Cross-Layer Gap | 16 | Contradictions spanning system prompt and user context layers |
| Temporal Injection | 10 | Outdated or future-dated claims inserted to erode prior commitments |
| Authority Poison | 7 | False authority or source attribution to override restrictions |
| Semantic Paraphrase | 6 | Meaning-altering rewrites that slip through pattern matching |
| Soft Injection | 3 | Subtle behavioral nudges embedded in benign-looking content |

### 3-Axis Evaluation
- **Axis 1 — Guard DR/FPR:** Does the safety guard catch attacks without false positives?
- **Axis 2 — LLM Behavior:** Does the LLM resist, comply, or affirm the attack natively?
- **Axis 3 — System DR:** Does the full stack (LLM + Guard) contain every attack?

---

## Results *(May 2026)*

| Model | Baseline DR | With EmberArmor | FPR | Slip-throughs | Latency |
|-------|:-----------:|:---------------:|:---:|:-------------:|:-------:|
| Claude Sonnet 4.6 | 95.1% | **100%** | 3.3% | 0 | 3,820ms |
| Gemini 3.1 Pro | 88.5% | **100%** | 10.0% | 0 | 6,389ms |
| Claude Haiku 4.5 | 85.2% | **91.8%** | 3.3% | 5 | 2,619ms |
| Kimi K2.6 | 52.5% | **96.7%** | 16.7% | 2 | 14,031ms |
| Kimi K2.5 | 45.9% | **100%** | 3.3% | 0 | 15,771ms |
| **EmberArmor (guard only)** | **98.4%** | — | **0.0%** | — | **860ms** |

The gap between baseline detection and system detection illustrates the enforcement layer's value: a model detecting 45.9% of attacks natively reaches 100% with EmberArmor added.

---

## Usage

```bash
# Install deps
pip install anthropic google-genai httpx torch transformers

# Set API keys
export ANTHROPIC_API_KEY=sk-ant-...
export GEMINI_API_KEY=AIza...
export KIMI_API_KEY=sk-...
export PERPLEXITY_API_KEY=pplx-...   # for Sonar baseline

# Run baseline (LLM only, no guard)
python run_claude_baseline.py --model claude-sonnet-4-6
python run_gemini_baseline.py --model gemini-3.1-pro-preview
python run_kimi_baseline.py

# Run integrated (LLM + EmberArmor guard)
python run_integrated_eval.py --model claude-sonnet-4-6
python run_integrated_eval.py --model gemini-3.1-pro-preview --save-responses

# Incremental checkpointing — safe to interrupt and resume
python run_integrated_eval.py --model claude-sonnet-4-6 --checkpoint
```

---

## Structure

```
EmberBench/
├── emberbench/
│   ├── datasets/            — 91 test cases (legal, financial, medical)
│   │   ├── legal/
│   │   ├── financial/
│   │   └── medical/
│   └── runner.py            — Core evaluation loop
├── ember_security/
│   └── dissonance_guard/    — Guard scorer + detector (shared dep)
├── results/                 — Per-case JSON results, per-model
│   ├── claude_sonnet_4_6_baseline_results.json
│   ├── claude_sonnet_4_6_integrated_v2_responses.json
│   ├── gemini_3_1_pro_preview_baseline_results.json
│   └── ...
├── run_integrated_eval.py   — Full-stack eval (LLM → EmberArmor guard)
├── run_claude_baseline.py
├── run_gemini_baseline.py
├── run_kimi_baseline.py
├── run_sonar_baseline.py    — Sonar-only baseline (Perplexity API)
└── .env.example
```

---

## Reproducing Results

EmberBench is designed to be reproducible. All 91 test cases are in `emberbench/datasets/` as structured JSON. The eval scripts use incremental checkpointing — if a run is interrupted, it resumes from the last completed case.

The guard scorer uses DeBERTa-v3-large NLI with:
- Label order: `contradiction=0`, `entailment=1`, `neutral=2`
- Affirmation threshold: NLI entailment ≥ 0.60

Gemini 3.1 Pro requires `max_output_tokens=8192` (reasoning model) — this is set in the eval script.

---

## Ecosystem

| Repo | Role |
|------|------|
| [EmberArmor](https://github.com/GrandMastaShake/EmberArmor) | Runtime enforcement layer |
| [EmberHoneypot](https://github.com/GrandMastaShake/EmberHoneypot) | AI deception + threat intelligence |
| [Corporeus](https://github.com/GrandMastaShake/Corporeus) | Static AST vulnerability scanner |
| [EmberBench](https://github.com/GrandMastaShake/EmberBench) | Adversarial evaluation harness (this repo) |

---

## License

MIT — see [LICENSE](LICENSE)
