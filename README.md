# Kaggle Dominator

An evidence-driven strategy and execution skill for legitimate Kaggle competition
work across tabular, computer vision, NLP/audio, simulation and agent ladders,
code competitions, and judge-scored hackathons.

It is the competitive brain: live reconnaissance, validation design,
`BEST_KNOWN` protection, experiment portfolios, ensembling, campaign focus, and
submission decisions. Pair it with a separate `kaggle` infrastructure skill for
authentication, downloads, kernels, and API plumbing.

## What v3 adds

- Universal routing across all major Kaggle formats.
- Portfolio states (`ACTIVE`, `MONITOR_ONLY`, `PAUSED`, `CLOSED`) and explicit
  budgets for attention, compute, submissions, storage, and deadline slack.
- A candidate queue with exploit/explore/audit lanes and stop/continue gates.
- Fresh-score, metric-direction, submission-availability, and provenance checks.
- Clear authority boundaries for submissions and public actions.
- Compliance, attribution, reproducibility, and anti-leakage safeguards.
- Learning checklists, public-resource guidance, an account scorecard, measured
  cross-domain lessons, and bounded autonomous loops.
- Contract tests and Codex discovery metadata.

## Core loop

```text
live recon
  → front card and resource budget
  → reproduce strongest compliant baseline
  → trustworthy validation / arena
  → candidate batch with cheap gates
  → full measurement and diversity analysis
  → submission decision with authority + quota checks
  → fresh score, ledger update, next decision
```

A candidate never replaces `BEST_KNOWN` until the correct real metric confirms it.
If validation disagrees with the leaderboard or ladder, repair validation before
continuing to tune.

## References

| Need | Reference |
|---|---|
| Multiple fronts and resource focus | [`campaign-control.md`](references/campaign-control.md) |
| Cross-domain competitive process | [`grandmaster-playbook.md`](references/grandmaster-playbook.md) |
| Measured account history | [`scorecard.md`](references/scorecard.md) |
| Tabular | [`tabular.md`](references/tabular.md) |
| CV, NLP, audio, signals | [`deep-learning.md`](references/deep-learning.md) |
| Simulation and agents | [`simulation.md`](references/simulation.md) |
| Code competitions and hackathons | [`code-and-hackathon.md`](references/code-and-hackathon.md) |
| Autonomous bounded loops | [`autonomous.md`](references/autonomous.md) |
| Self-curation | [`self-improvement.md`](references/self-improvement.md) |
| ML craft | [`learning-craft.md`](references/learning-craft.md) |
| Learning sources and attribution | [`resources.md`](references/resources.md) |
| Winning solution patterns | [`winning-solutions.md`](references/winning-solutions.md) |
| Tools and communities | [`arsenal.md`](references/arsenal.md) |

## Install

Codex project-local installation:

```bash
git clone https://github.com/KapaSique/kaggle-dominator.git \
  .agents/skills/kaggle-dominator
```

Codex user installation:

```bash
git clone https://github.com/KapaSique/kaggle-dominator.git \
  ~/.agents/skills/kaggle-dominator
```

Invoke it explicitly as `$kaggle-dominator`, or let it trigger on competitive
Kaggle strategy requests. It deliberately does not trigger for badge collection,
account configuration, or CLI installation alone.

## Verify

```bash
python3 -m unittest -v tests/test_skill_contract.py
python3 /path/to/skill-creator/scripts/quick_validate.py .
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py tests/*.py
```

## License

[MIT](LICENSE)
