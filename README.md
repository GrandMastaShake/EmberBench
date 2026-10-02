# EmberBench

A small development set of two-statement test cases (legal, financial, medical) and multi-turn drift sequences for contradiction and prompt-injection detectors, with raw results from early runs against it.

## Status

Frozen development set. Not a benchmark, and the harness does not run from this repository. Status as of 2026-10-02.

## What is here

- **91 two-statement cases**, 61 adversarial and 30 benign, written as Python literals in `emberbench/datasets/legal.py`, `financial.py` and `medical.py`. Each case has two statements, an expected tier (`SAFE`, `USER_FLAGGED` or `ESCALATE_HALT`), a domain, an attack type, a difficulty and a note.
- **5 multi-turn drift sequences** of 6 or 7 turns in `emberbench/datasets/drift.py`. Each turn is meant to look acceptable next to the one before it, while the last turn contradicts or overrides the first. `DriftCase` gives `consecutive_pairs` and `endpoint_pair` for scoring. The sequences are not part of any figure below.
- **Raw per-case results** in `results/`, 91 records per file. Three `*_baseline_results.json` files hold a language model's tier for each pair. Five `*_integrated_v2_responses.json` files hold a model's reply to statement A alone and the guard's tier for statement A against that reply.
- **Two generated reports** in `results/` from April 2026, kept as a record.
- A partial copy of the detector the cases were written for, in `ember_security/dissonance_guard/`, and a signature store in `data/signatures.json` whose reader is not included.

| Attack type | Cases | Expected tier | What the pairs look like |
|---|---:|---|---|
| `numeric_contradiction` | 19 | `ESCALATE_HALT` | The same quantity with different numbers |
| `cross_layer_gap` | 16 | 12 `ESCALATE_HALT`, 4 `USER_FLAGGED` | Two sentences that contradict each other on a fact. Despite the name, no system-prompt layer is involved |
| `temporal_injection` | 10 | `USER_FLAGGED` | A claim presented as current, against a statement that says otherwise |
| `authority_poison` | 7 | `ESCALATE_HALT` | A claimed role or legal authority used to demand that restrictions be lifted |
| `semantic_paraphrase` | 6 | `ESCALATE_HALT` | An instruction override phrased in legal language. The two statements agree with each other |
| `soft_injection` | 3 | `ESCALATE_HALT` | An instruction override presented as a regulatory waiver |
| `benign_fpr` | 30 | `SAFE` | The same fact restated in other words |

By domain: legal 31 (21 adversarial, 10 benign), financial 30 (20, 10), medical 30 (20, 10).

## The 98.4% figure

Earlier versions of this README reported 98.4% detection and 0% false positives for the guard alone. Read that figure as follows.

- It was measured on 2026-04-22 on these same 91 cases. 60 of 61 adversarial cases received exactly the expected tier (98.4%), all 61 were flagged at some tier, and 0 of 30 benign cases were flagged.
- The detector was adjusted against these cases three times that day. The reported detection rate went from 70.5% to 88.5% to 98.4%.
- There is no held-out set. The figure is an in-sample development-set result. It does not estimate how the detector behaves on text it was not adjusted against.
- 30 benign cases is a small sample. 0 false positives in 30 is consistent with a true false-positive rate anywhere up to about 11.6% (exact binomial 95% upper bound).
- It cannot be reproduced from this repository. The harness does not import (see below), the per-case output of that run is not in `results/`, and `ember_security/dissonance_guard/scorer.py` here is a later revision than the one that produced the figure.

The model comparison table that used to be here has been removed. Its columns came from different tasks scored by different rules, so its rows could not be compared with each other.

## What does not work

- Nothing imports from a clean clone. `python -m emberbench`, `run_eval_direct.py`, `run_integrated_eval.py` and the four `run_*_baseline.py` scripts all stop with `ModuleNotFoundError`. The code imports a package named `eval` and parts of `ember_security` (`config`, `thalamic_conductor`, `offensive` and others) that are not in this repository.
- The pattern-matching layers behind part of the guard figure live in `ember_security.offensive`, which is one of the missing parts.
- There is no `tests` directory. `python -m pytest` collects 0 items.
- `pyproject.toml` does not declare everything the code imports (numpy, structlog, sentence-transformers, torch, scipy, google-genai).

## What you can run

Checked with Python 3.12.4 on Windows 11, from the repository root.

Tally a result file (standard library only):

```
python -c "import json, collections; rows = json.load(open('results/claude_sonnet_4_6_baseline_results.json', encoding='utf-8')); print(len(rows), collections.Counter(r['attack_type'] for r in rows))"
```

It prints `91` and the count per attack type.

Load the cases. The dataset modules need pydantic (`pip install pydantic`) and two package names this repository does not provide, so the script registers those names before importing:

```python
import sys, types

# Register the package names the dataset modules expect, without running
# the package __init__ files that import modules missing from this repo.
for name, path in {
    "eval": ".",
    "eval.emberbench": "emberbench",
    "ember_security": "ember_security",
    "ember_security.dissonance_guard": "ember_security/dissonance_guard",
}.items():
    pkg = types.ModuleType(name)
    pkg.__path__ = [path]
    sys.modules[name] = pkg

from eval.emberbench.datasets import get_all_cases, get_all_drift_cases

cases = get_all_cases()
print(len(cases), "cases,", len(get_all_drift_cases()), "drift sequences")
```

It prints `91 cases, 5 drift sequences`. Each case is a dataclass with `case_id`, `statement_a`, `statement_b`, `expected_tier`, `domain`, `attack_type`, `difficulty` and `notes`.

## Known issues

- The scripts do not share one definition of "detected": exact tier match (`run_eval_direct.py`, the Kimi and Sonar scripts), flagged at either tier (`emberbench/report.py`), any tier other than `SAFE` (the Claude and Gemini scripts). This is why `results/emberbench_report.md` says 100% for the run described above as 98.4%.
- The bootstrap intervals in `results/emberbench_report.md` have zero width because the sample had no failures, so they carry no information. `emberbench/bootstrap.py` also computes precision as `1 - FPR`, so its F1 is wrong.
- Layer attribution labels any result slower than 5 ms as `nli`, including benign cases that passed. That is where `nli 75` in the report comes from.
- `results/emberbench_comparison.md` has a Sonar column that was never filled in, and its per-attack table puts the one tier mismatch under the wrong attack type.
- `emberbench/report.py` hard-codes a "v3 baseline" of 182 cases. This repository has no data, script or results for it.
- The Claude, Gemini and integrated scripts save their output to a hard-coded `/home/user/workspace/` path.
- The cases have no written rule for `USER_FLAGGED` versus `ESCALATE_HALT`, no field saying which statement is the trusted one, and no source field for the facts they state.
- Every benign pair is a close restatement with the same numbers. None has a second statement that adds a different number, so the set does not test ordinary text that does.

## What would make this a benchmark

- A held-out split, written without sight of any detector and frozen before any detector change.
- A benign set large enough to put a useful bound on the false-positive rate, including ordinary text that adds numbers or uses trigger words.
- Results on public sets alongside, so the numbers can be compared with other detectors.
- One definition of detection, with exact binomial intervals on every rate.
- A harness that runs from a clean clone against any detector, not only the one it was written for.

## Direction

This repository is frozen as a development set. Evaluation of the constraint-ledger work in [EmberArmor](https://github.com/GrandMastaShake/EmberArmor) will use public benchmarks first. These cases may return later as one small domain-specific set with a held-out split.

## License

MIT. See [LICENSE](LICENSE).
