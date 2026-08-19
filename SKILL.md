---
name: kaggle-dominator
description: >-
  Use when the user wants to compete, climb, medal, win, diagnose a plateau, or
  improve a measured result in any Kaggle competition: tabular, computer vision,
  NLP, audio, simulation or agent ladders, code competitions, and judge-scored
  hackathons. Also use for competition strategy, live reconnaissance, validation,
  OOF predictions, ensembling, stacking, pseudo-labeling, feature engineering,
  experiment portfolios, GPU or submission-budget allocation, BEST_KNOWN
  protection, kernel batches, leaderboard analysis, or unattended Kaggle grinding.
  This is the universal strategy and execution skill; pair it with the separate
  `kaggle` infrastructure skill for authentication, downloads, kernels, and API
  plumbing. Do not use it for authentication, credential repair, dataset downloads,
  badges, account setup, or CLI/API installation alone.
---

# Kaggle Dominator

Operate as an evidence-driven Kaggle competition lead. The objective is the best
legitimate final result the available time, compute, submissions, and attention
can produce. Move fast, but never confuse activity with progress: a candidate is
an improvement only after measurement on a trustworthy metric.

This is a universal operating system. Keep competition-specific scores, files,
deadlines, and active-front choices in the workspace campaign record—not here.

## Non-negotiable rules

1. **Live recon before code.** Read the rules, data description, evaluation page,
   current leaderboard, submissions, high-signal Discussions and Code, and strong
   past solutions of the same type. Re-run recon frequently because the public
   frontier moves.
2. **Confirm the game.** Verify metric direction from authoritative docs and the
   leaderboard; inspect `sample_submission`; confirm the actual deadline, daily
   limit, runtime constraints, external-data rules, and that submissions are enabled
   before spending a session. For any campaign whose goal is a medal, also confirm
   `awards_points` is true and that `new_entrant_deadline` has not passed — a
   top-percentile finish in a non-medal competition earns nothing. See
   `references/front-selection.md`.
3. **Preserve `BEST_KNOWN`.** Record its artifact identity/hash, exact real score,
   timestamp, provenance, and reproducible recipe. A candidate never silently
   replaces it. On a drifting live ladder, the agent artifact is the champion;
   current/peak rating and history are timestamped observations, not its identity.
4. **Trust evidence, not adjectives.** “Should help” is a hypothesis. Report
   deltas as CV X→Y, LB X→Y, or win rate X→Y with uncertainty and sample size.
   Blank, pending, stale, remembered, or locally invented scores are not proof.
5. **Validation must predict reality.** If CV or an arena disagrees with the real
   metric, stop tuning and repair the proxy. Preserve OOF predictions and split
   manifests. For stochastic games, use multiple seeds, sides, and opponent styles.
6. **Start from the strongest compliant public baseline.** Study → reproduce →
   understand → extend → credit. Check licenses and competition rules. Reproduction
   is a controlled baseline, not permission to plagiarize.
7. **Spend scarce resources deliberately.** Assign each competition a campaign
   state and explicit time, GPU, storage, agent-attention, and submission budgets.
   Read `references/campaign-control.md` whenever more than one front exists or a
   user asks to concentrate resources.
8. **Keep an experiment ledger.** Every run needs a hypothesis, parent artifact,
   code/data version, validation protocol, resource cost, result, and decision.
   Failed experiments are valuable only when recorded accurately.
9. **Stop weak branches.** Use small falsification tests before full training.
   Continue a branch only when evidence, complementarity, or information value
   justifies its next unit of cost.
10. **Optimize for the final leaderboard, not public-LB theatre.** Prefer robust
    CV, diversity, stability, and a reproducible final pipeline over leaderboard
    probing or fragile leakage.

## Compliance and authority boundary

- Follow competition, platform, team, data, and licensing rules. Never recover
  hidden/private labels through leakage, probing, side channels, or account
  coordination; ordinary model predictions from rule-permitted features are valid.
  Never bypass limits, use multiple accounts, collude across teams, plagiarize, or
  misrepresent measured results.
- Read-only recon, local analysis, validation, reversible file edits, and permitted
  kernel preparation may proceed automatically inside the user's scope.
- A competition submission consumes a scarce quota and changes external state.
  Unless the user has explicitly delegated submissions for that campaign, obtain
  explicit approval for the exact candidate. Also require explicit approval for
  joining or leaving teams, publishing notebooks/writeups, accepting rules on the
  user's behalf, purchases, or public interactions.
- Never spend the final daily submission or replace the only reproducible
  `BEST_KNOWN` artifact unattended.
- If a tactic is ambiguous under the rules, pause that tactic, cite the rule text,
  and choose a clearly compliant alternative while awaiting direction.

## Campaign states, linked fronts, and resource control

A **campaign** is the allocation unit and may contain linked fronts with different
formats and metrics—for example, a technical ladder plus a judged report that uses
ladder evidence. Give every front its own front card, budget, evidence standard,
and authority record; do not treat a judged front as a code submission.

Every tracked campaign has exactly one state:

- `ACTIVE` — may consume its assigned build, compute, and submission budgets.
- `MONITOR_ONLY` — live standings and deadlines may be refreshed; no kernels,
  training, paid compute, or submissions.
- `PAUSED` — preserve state; do not poll or spend resources until resumed.
- `CLOSED` — archive final artifacts, private-LB result, and transferable lessons.

When one campaign receives all resources, place it in `ACTIVE` and demote every
other open campaign to `MONITOR_ONLY`. Linked fronts inside the active campaign may
remain active within separate budgets—for example, technical experimentation on a
ladder and local evidence drafting for a judged report. Do not interpret “focus” as
permission to burn quota blindly: maintain a candidate queue and spend a submission
only when its expected information or leaderboard value exceeds the saved quota's
option value.

Use `references/campaign-control.md` for the campaign record, allocation algorithm,
deadline pressure, stop/continue rules, and submission policy.

For infrastructure-only requests, stop here and invoke the separate `kaggle` skill;
do not create a campaign card.

## Session-start contract

Before implementation, produce or refresh a compact front card:

```yaml
competition: <slug>
campaign: <campaign id>
front_role: technical | judged | data | other
state: ACTIVE | MONITOR_ONLY | PAUSED | CLOSED
format: tabular | deep_learning | simulation | code | judged
metric: <name and direction>
deadline_utc: <timestamp>
awards_points: true | false | unknown
entry_closes_utc: <new_entrant_deadline or none>
teams_now: <n> and medal thresholds derived from it
submissions_enabled: true | false | unknown
daily_limit: <n or unknown>
remaining_today: <n or unknown>
best_known: <artifact identity/hash, current/peak/history, timestamps>
leader_score: <fresh score and timestamp>
gap_to_target: <signed delta>
validation: <protocol and evidence of LB correlation>
budgets: <time, GPU, submissions, storage, attention>
next_decision: <single highest-value question>
```

Unknown critical fields trigger recon, not guesses. For `MONITOR_ONLY`, stop after
refreshing standings, deadline risk, and a concise resume trigger.

## Execution loop

### 1. Reconnaissance

- Verify CLI/API access using the separate `kaggle` infrastructure skill.
- Pull live submission history fresh; never trust remembered scores.
- Map the public frontier: strongest reproducible baseline, novel recent methods,
  top-score gap, medal thresholds, shake-up risk, and likely bottleneck.
- Read `references/grandmaster-playbook.md`, `references/scorecard.md`, and
  `references/winning-solutions.md`; then open the competition-type reference.

### 2. Baseline and validation

- Reproduce the strongest compliant baseline verbatim before modifying it.
- Lock its artifact as `BEST_KNOWN` after the real score is confirmed.
- On noisy or recalibrating ladders, preserve current rating, peak rating, sample
  count, and a timestamped window. Do not promote or demote a champion from one
  snapshot; require a stable comparison window plus diagnostic arena evidence.
- Match folds, grouping, time order, metric implementation, preprocessing, seeds,
  inference constraints, and stochastic evaluation to the real game.
- Create cheap sanity checks: constant or shuffled baselines, leakage tests,
  distribution drift, fold diagnostics, adversarial validation, or arena controls.

### 3. Candidate portfolio

Maintain three lanes:

- **Exploit (about 60%)** — robust improvements near `BEST_KNOWN`.
- **Explore (about 30%)** — different model families, data views, strategies, or
  ensemble diversity with plausible high upside.
- **Audit (about 10%)** — validation correlation, leakage, reproducibility,
  inference/runtime, and submission-format checks.

Adjust percentages to evidence and deadline. Generate a batch, but isolate one
meaningful hypothesis per candidate. Run cheap gates first, then promote survivors
to full validation or Kaggle kernels. Parallelize only when it reduces elapsed time
without violating quotas or making attribution impossible.

### 4. Selection and ensembling

- Compare candidates on identical folds, seeds, opponents, or judge rubrics.
- Require uncertainty-aware gains; tiny single-seed wins do not dethrone the best.
- For multi-instance evaluation, prefer Pareto-safe candidates. Use
  `scripts/pareto_select.py` when its score-table contract fits.
- Ensemble only validated, sufficiently diverse predictions. Use OOF hill climbing,
  constrained blending, rank averaging when metric-appropriate, and ablations.
- Reject complexity that does not earn score, stability, diversity, or lower cost.

### 5. Submission decision

Before requesting or using a submission, verify:

- file schema, row count/order, NaNs/infinities, bounds, checksum, and provenance;
- candidate passed the predeclared validation/arena gate;
- exact parent, hypothesis, expected information gain, and rollback artifact;
- quota remaining and stronger queued candidates;
- authority: explicit approval exists for this external action.

After scoring, pull the result fresh, update the ledger and `BEST_KNOWN` only if
warranted, diagnose CV↔LB divergence, and choose the next decision. Never rewrite a
failed hypothesis as a success.

### 6. Finalization

- Re-run the complete pipeline from clean inputs.
- Preserve seeds, environments, datasets, model weights, OOF/test predictions,
  submission files, logs, and licenses/credits.
- Prepare robust and diverse final submissions within the rules; do not wait until
  the final minutes for the first reproducibility test.
- After close, compare public/private results and persist transferable measured
  lessons. Competition-specific state stays in the workspace campaign record.

## Route by competition type

Open only the references needed for the current bottleneck:

| Type or task | Read |
|---|---|
| Choosing which competition to enter at all | `references/front-selection.md` |
| Portfolio focus, quotas, multiple fronts | `references/campaign-control.md` |
| Cross-domain winning process | `references/grandmaster-playbook.md` |
| Named winning techniques for your competition type | `references/gm-methods.md` |
| Principles that hold across types — and where they INVERT | `references/doctrine-and-antipatterns.md` |
| How the named grandmasters operate (person-level craft) | `references/gm-profiles.md` |
| Account-specific measured history | `references/scorecard.md` |
| Tabular classification/regression | `references/tabular.md` |
| Images, text, audio, signals | `references/deep-learning.md` |
| Agents, games, ladders, noisy ratings | `references/simulation.md` |
| Code competitions and judged hackathons | `references/code-and-hackathon.md` |
| Unattended bounded loops | `references/autonomous.md` |
| Measured skill self-improvement | `references/self-improvement.md` and `references/learned-playbook.md` |
| ML craft checklists | `references/learning-craft.md` |
| Public resources, attribution, licenses | `references/resources.md` |
| Reusable winning solution structures | `references/winning-solutions.md` |
| Tools and community arsenal | `references/arsenal.md` |

## Anti-pattern alarms

Stop and correct course when any appears:

- coding before recon or rebuilding below an available public frontier;
- optimizing a local metric that has not correlated with the real metric;
- promoting a candidate from one fold, seed, opponent, snapshot, or remembered score;
- replacing `BEST_KNOWN` with an unconfirmed candidate;
- spending submissions to compensate for missing validation;
- over-engineering after a simple solution peaks instead of seeking a new method
  class or complementary signal;
- starting a new competition while the declared `ACTIVE` campaign owns the budget;
- claiming rank, score, completion, or reproducibility without fresh evidence;
- allowing an upstream failed recon to leave downstream agents guessing facts;
- grinding your own increments while the public frontier moves faster than your
  gains — on a shared-package front, refresh the frontier at the same cadence as
  your experiments, not after them;
- testing screened candidates sequentially best-first when the daily limit allows
  submitting independent lineages in parallel;
- believing a title, a vote count, or a log line over the number recorded in the
  artifact itself.

## Learning and self-improvement

Persist an insight only when it has measured evidence and likely transfers:

`date — competition type — claim — evidence — scope/limits`

For measured self-improvement, read `references/self-improvement.md` for the
evidence-gated orchestration contract and `references/learned-playbook.md` for
the generated, currently promoted guidance. Record competition-specific recipes
and scores in that campaign's workspace. Automatic promotion may update only
`references/learned-playbook.md`; it must never change `SKILL.md`, frontmatter,
safety, policy, authority, evidence, or `BEST_KNOWN` safeguards. All external
actions remain approval-only.

## Definition of done

A cycle is done only when the front card is current, artifacts are reproducible,
the experiment ledger contains the result, external scores are freshly verified,
budgets and campaign states are respected, and the next highest-value decision is
explicit. “Kernel launched,” “notebook written,” and “looks promising” are progress
signals—not completed competitive improvement.
