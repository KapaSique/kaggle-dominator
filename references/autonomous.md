# Autonomous mode — eval-driven loops, agent teams, overnight runs

The deep end of the iceberg: the agent runs experiments, judges itself, and iterates with little or no human in the loop. Powerful and dangerous in equal measure. **Autonomy without guardrails burns your API quota and floods the leaderboard with garbage submissions.** What makes it safe is exactly the constitution in `SKILL.md` — a protected `BEST_KNOWN`, trust-only-the-real-metric, proof-of-work. Build autonomy *on top of* that discipline, never instead of it.

Read this when the user wants the agent to grind a competition hands-off (overnight, in a loop, or as a team), or asks about headless / eval-driven / self-improving runs.

## Contents
- [The prime directive of autonomy](#the-prime-directive-of-autonomy)
- [Level 1 — eval-driven loop (one agent)](#level-1--eval-driven-loop-one-agent)
- [Level 2 — parallel agent teams](#level-2--parallel-agent-teams)
- [Level 3 — headless overnight runs](#level-3--headless-overnight-runs)
- [Level 4 — orchestration (frontier)](#level-4--orchestration-frontier)
- [Guardrails (non-negotiable)](#guardrails-non-negotiable)

---

## The prime directive of autonomy

An autonomous agent amplifies whatever discipline it already has. A disciplined agent in a loop compounds gains; an undisciplined one compounds mistakes faster than you can watch. So before any loop runs, the invariants from the constitution must hold automatically:

- `BEST_KNOWN` is a file the loop **reads but never overwrites** except through the confirmed-by-real-metric gate.
- Every experiment writes a row to `results.csv` (slug, hypothesis, CV, LB) — the loop's memory.
- A hard budget exists (max iterations / max API spend / max kernels) and the loop halts when hit.
- Submissions are rate-limited and gated; an unattended loop **proposes** submissions, a human (or an explicit rule) **confirms** the final one.

## Level 1 — eval-driven loop (one agent)

The single most useful autonomous pattern for Kaggle. The agent cycles:

```
hypothesis → build/modify kernel → push → poll → read OOF/LB → log → decide → repeat
```

Stop conditions (any one): target metric reached, N iterations with no improvement, budget exhausted. The loop's state lives on disk (`results.csv` + the OOF files), so it survives restarts and stays honest.

Skeleton (headless, self-paced):

```bash
# scripts/kaggle_eval_loop.sh  — see scripts/ for the full version
TARGET=0.9720; BEST=$(cat BEST_KNOWN.txt); ROUNDS=8
for i in $(seq 1 $ROUNDS); do
  claude -p "Read results.csv and BEST_KNOWN=$BEST. Propose ONE new experiment that is
    different from everything tried (new FE / model / fold scheme). Build the kernel, push it
    under my account, poll to completion, read the OOF balanced-accuracy, append a row to
    results.csv. Do NOT submit. Print only the new OOF score." --allowedTools "Bash Read Write Edit"
  NEW=$(tail -1 results.csv | cut -d, -f5)
  awk "BEGIN{exit !($NEW > $BEST)}" && BEST=$NEW && echo "$BEST" > BEST_KNOWN.txt
  awk "BEGIN{exit !($BEST >= $TARGET)}" && { echo "target hit"; break; }
done
```

The constitution does the heavy lifting here: each iteration is an *additional* attempt, `BEST_KNOWN` only moves on a real improvement, and the loop never submits on its own.

## Level 2 — parallel agent teams

Instead of one agent iterating serially, fan out specialised agents with **isolated context** (the parallel-dispatch pattern). The general rule — *one agent per independent domain, only when there's no shared state* — maps onto Kaggle almost perfectly, because **each independent OOF predictor IS an independent domain**. Parallel agents ↔ parallel kernels ↔ diversity-of-OOF: three layers of the same lever that breaks the flat zone.

**What is independent (dispatch in parallel):** each base predictor is its own domain — one agent builds the LightGBM-with-count-encoding OOF, another the CatBoost-with-target-encoding OOF, another the TabPFN bag, another the NN. They never touch each other's work, so they run concurrently.

**What is NOT independent (keep serial — a barrier):** the **ensemble needs ALL OOF at once**, so Hill-Climbing / L2 stacking is a barrier *after* the fan-out, never a parallel peer. Likewise, an L2 model that consumes L1 OOF depends on L1 — sequential, not parallel.

**Two Kaggle-specific contracts the generic pattern doesn't mention:**
- **Shared fold scheme is a contract, not shared state.** Every agent MUST use the *same* `StratifiedKFold(seed)` and the *same* row order, or their OOF columns don't align and the ensemble is silently corrupt. Pin the seed in the dispatch prompt. (This is the opposite of "agents must not share state" — here they must share the *fold definition* while sharing nothing else.)
- **Isolate by kernel slug + output file.** Two agents writing to the same kernel slug or the same `oof.npy` collide. Give each its own slug (`<user>/s6e6-lgbm-cnt`, `<user>/s6e6-cat-tgt`, …) and its own output path. That's how you keep them from interfering.

A good team:

| Role | Job | Parallel? |
|------|-----|-----------|
| **Explorer** | Mine discussion + past-season writeups for diverse FE / model ideas | yes (by idea) |
| **Modeler** | Turn one idea into a kernel, push under its own slug, collect OOF | yes (by predictor) |
| **Critic** | Hunt leakage, verify CV↔LB correlation, reject overfit/misaligned OOF | barrier (gates Modeler output) |
| **Ensembler** | Hill-Climb the surviving OOF, propose the candidate | barrier (needs all OOF) |

**Dispatch prompt contract** (focused, self-contained, explicit output — same discipline as fixing one test file):
```
Scope:   ONE predictor — LightGBM, multiclass, count-encoded categoricals.
Data:    competition <slug>, target=class, metric=balanced_accuracy.
Fold:    StratifiedKFold(5, shuffle=True, random_state=42)  ← DO NOT change; OOF must align.
Goal:    produce oof_lgbm_cnt.npy + test_lgbm_cnt.npy, report OOF balanced-accuracy.
Constraints: do NOT touch BEST_KNOWN, do NOT submit, use your own kernel slug.
Return:  one results.csv row + the OOF score. Nothing else.
```

**Integrate like the generic pattern, plus a leakage spot-check.** When agents return: read each summary, confirm no two used a different fold seed (silent corruptor), let the **Critic veto** anything with suspicious CV↔LB gap before it reaches the Ensembler. Agents make *systematic* errors — a subtly leaked target encoding will look like a great OOF and wreck the LB. Spot-check before trusting.

**Common mistakes (Kaggle flavour):**
- ❌ "Build a bunch of models" → agent sprawls. ✅ "Build ONE LGBM-count-encoded OOF, 5-fold seed 42."
- ❌ Each agent picks its own folds → OOF don't align, ensemble is garbage. ✅ Pin the fold scheme in every prompt.
- ❌ Agents share a slug/output file → collisions. ✅ One slug + one output path per agent.
- ❌ Ensembler dispatched in parallel with modelers → it has nothing to ensemble yet. ✅ Ensemble is a barrier after the fan-out.

## Level 3 — headless overnight runs

`claude -p` (headless) on a schedule — cron locally or GitHub Actions in CI. The agent works while you sleep; you read the log in the morning.

```bash
# cron: 2am nightly eval loop, log to disk
0 2 * * *  cd ~/comp && ./scripts/kaggle_eval_loop.sh >> nightly.log 2>&1
```

**Safe to run unattended:** kernel experiments, OOF generation, EDA reports, leakage audits, ensemble search — anything that writes files and proposes, but doesn't act irreversibly.

**NOT safe unattended:** the final submission decision (limited daily budget, irreversible), anything that publishes or spends real money. Leave those gated for the morning review. An overnight run that *prepares* the best candidate and a one-line `submit` you approve at breakfast is the sweet spot.

See `scripts/kaggle_eval_loop.sh` (loop) and `scripts/nightly-agent.yml` (GitHub Actions template — needs an `ANTHROPIC_API_KEY` repo secret you add yourself).

## Level 4 — orchestration (frontier)

Honest framing: high cost, real fragility, only worth it when the work genuinely exceeds one context.

- **Agents managing agents.** An orchestrator spawns task-scoped subagents and synthesises their results — only when the task decomposes cleanly (independent subsystems). Nesting deeper than one level usually costs more than it returns.
- **Multi-repo orchestration.** One agent drives synchronized changes across repos (e.g. ML model + FastAPI backend + frontend). Use git worktrees for isolation; verify each repo independently before any cross-repo commit.
- **Agent-built tooling.** The agent writes skills/scripts for future agents. You already hand-write skills — the next step is generating a task-specific skill, then having a fresh agent consume it. Treat generated tooling as untrusted until reviewed.

These earn their keep on scale (large migrations, sweeps across many competitions), not on a single tabular comp. Don't reach for them to feel advanced — reach for them when one context honestly can't hold the work.

## Guardrails (non-negotiable)

- **API budget cap.** Headless loops spend real tokens and can hit provider daily limits (e.g. a bagged-TabPFN sweep can exhaust a cloud quota in one run). Set a max-iterations / max-spend ceiling and log spend.
- **Submission gate.** Unattended runs never spend the last daily submission. They prepare and rank candidates; the human or an explicit rule confirms.
- **`BEST_KNOWN` is sacred.** The loop may read it; it changes only through the real-metric gate. A regression overnight is worse than no run at all.
- **Kill switch.** A loop must check a stop flag (file/env) each iteration so you can halt it without hunting the process.
- **Everything logged.** `results.csv` + per-kernel logs. If you can't reconstruct what the agent did overnight, you can't trust the morning's number.
- **Insights persisted, not just scores.** When an unattended run discovers something measured (a mechanism, a debunked assumption, a ceiling), it appends it per "CAPTURE WHAT YOU LEARN" in SKILL.md — transferable → `grandmaster-playbook.md`, comp-specific → the competition's memory note. The morning review should read what the agent *learned*, not only the leaderboard delta.
