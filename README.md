# 🏆 kaggle-dominator

**A [Claude Code](https://www.anthropic.com/claude-code) skill that turns the agent into a relentless Kaggle Grandmaster.**

![Claude Code Skill](https://img.shields.io/badge/Claude%20Code-Skill-d97757)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/battle--tested-yes-success)

> Not another list of tips. A **battle constitution** + a **technique arsenal** that sets the operating mode for a whole competition session: recon the top public solutions, protect a `BEST_KNOWN`, run experiments in parallel GPU-kernel batches, **trust only the real metric**, and ensemble your way to the prize zone.

Most "ML assistant" prompts make the agent agreeable and verbose. This one makes it *win*. It encodes the habits that actually move a leaderboard — and, just as importantly, the anti-patterns that quietly wreck a score (chasing a local metric, replacing a working baseline with an unverified idea, believing a single noisy number).

---

## Why it's different

| Most prompts | kaggle-dominator |
|---|---|
| "Here are some ideas you could try" | Decide → run → measure → next. One number or it didn't happen. |
| One model, tuned forever | Dozens of OOF in parallel kernel batches, Hill-Climb the ensemble |
| Trusts local CV | **Validation must match the LB** — fix the proxy *first* if it drifts |
| Replaces the baseline with the new idea | Keeps a protected `BEST_KNOWN`; new ideas are *additional* attempts |
| Generic advice | Loads a **type-specific arsenal** (tabular / DL / simulation / code / hackathon) |

## The five iron rules

1. **Recon first, code second.** Reproduce the best public baseline before writing your own.
2. **Preserve the best.** `BEST_KNOWN` changes only after the *real* metric confirms it.
3. **Trust the real metric only.** "Should help" is a hallucination until measured.
4. **Validation must match reality.** If local says 70% and the LB drops, the validation is broken — fix it first.
5. **Proof-of-work.** "Done / improved" comes only with a number on the correct metric.

## What's inside

Progressive disclosure — the constitution loads on trigger, the detailed arsenal loads on demand by competition type:

| Competition type | Arsenal file |
|---|---|
| Tabular / Playground | [`references/tabular.md`](references/tabular.md) — OOF factory × Hill Climbing × multi-level stacking × pseudo-labeling × the flat-zone exit |
| Computer Vision / NLP | [`references/deep-learning.md`](references/deep-learning.md) — backbones, augmentation, TTA, pseudo-labeling, WBF/NMS |
| Simulation / Agent ladder | [`references/simulation.md`](references/simulation.md) — opponent pools, lookahead/MCTS, stability > cleverness |
| Code competition / Hackathon | [`references/code-and-hackathon.md`](references/code-and-hackathon.md) — regression guards, rubric-as-metric, the writeup |
| Tools & treasure troves | [`references/arsenal.md`](references/arsenal.md) — TabPFN, winning-solution repos, NVIDIA Grandmasters Playbook, cuML, W&B |
| Autonomous / hands-off | [`references/autonomous.md`](references/autonomous.md) — eval-driven loops, parallel agent teams, overnight headless runs — **with guardrails** (protected `BEST_KNOWN`, budget caps, gated submissions). Ships [`scripts/kaggle_eval_loop.sh`](scripts/kaggle_eval_loop.sh) + a [`scripts/nightly-agent.yml`](scripts/nightly-agent.yml) GitHub Actions template |

## Install

**Option A — clone into your skills directory (recommended):**

```bash
git clone https://github.com/KapaSique/kaggle-dominator.git ~/.claude/skills/kaggle-dominator
```

Then in any Claude Code session it auto-triggers on Kaggle work, or invoke explicitly with `/kaggle-dominator`.

**Option B — one-file install:** grab [`kaggle-dominator.skill`](kaggle-dominator.skill) and drop it in via your skill manager.

**Option C — project-local:** copy `SKILL.md` + `references/` into `<your-project>/.claude/skills/kaggle-dominator/`.

> Pairs cleanly with infrastructure skills (CLI/MCP/badges) — this one is the **strategy brain**, those are the **plumbing**. Use both.

## Battle-tested

This isn't theory-ware. On a live Playground competition (stuck in a flat zone at a hard score ceiling) the skill's "**confirm the metric, don't guess it**" rule caught a trap: the pipeline *looked* like it was optimizing the wrong objective. Instead of charging in to "fix" it, the constitution forced a probe — submit a constant prediction, read the score back. The number (`0.33333` for a 3-class problem) revealed the metric was **balanced accuracy**, the pipeline was already correct, and the "fix" would have *tanked* the score. Rule 3 (*trust only the real metric*) prevented a confident, plausible, wrong move.

That probe trick is now baked into the skill.

## Philosophy

Built from real losses, not vibes. Descended from a personal "Kaggle superprompt" and distilled with lessons from the [NVIDIA Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/) and the community [winning-solutions archive](https://github.com/faridrashidi/kaggle-solutions).

## License

[MIT](LICENSE) — use it, fork it, win with it. PRs welcome.
