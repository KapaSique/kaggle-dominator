# Kaggle Dominator v3 design

Date: 2026-07-11
Status: approved in conversation

## Goal

Turn `kaggle-dominator` into the single universal strategy skill for improving
results in any legitimate Kaggle competition. Pokémon TCG is the current active
campaign, not a specialization baked into the skill.

## Source of truth and deployment

- Canonical source: this `kaggle-dominator-repo` repository.
- Installed project skill: `.agents/skills/kaggle-dominator`.
- Retire the divergent `.agents/skills/kaggle-practice-coach` installation.
- Preserve measured user-authored lessons in
  `references/grandmaster-playbook.md` and keep strategy distinct from the
  infrastructure-oriented `kaggle` skill.

## Behaviour

The skill must:

1. Trigger for competitive Kaggle strategy across tabular, deep learning,
   simulation/agent, code, and judge-scored formats.
2. Reconcile winning pressure with rules, attribution, honest validation, and
   reproducibility. It must never encourage hidden-label inference, multi-account
   limit bypasses, plagiarism, or other rule violations.
3. Start from live reconnaissance, confirm metric direction and submission
   availability, preserve `BEST_KNOWN`, and treat every improvement as unproven
   until measured on the correct metric.
4. Allocate scarce submissions, GPU quota, time, and agent attention across a
   portfolio of competitions. One campaign may receive all active resources while
   other fronts remain read-only monitors.
5. Choose methods by competition type and bottleneck, run bounded batches, stop
   weak branches early, and keep an auditable experiment ledger.
6. Separate automatic reversible actions from submissions, public publishing,
   team changes, and other consequential actions that require explicit authority.
7. Persist only measured, transferable lessons and keep competition-specific
   state outside the generic skill.

## Structure

- `SKILL.md`: concise operating contract, workflow, safety gates, routing table,
  and definition of done.
- `references/campaign-control.md`: portfolio allocation, resource budget,
  submission policy, stop/continue rules, and campaign state schema.
- Existing technique references: tabular, deep learning, simulation, code and
  hackathon, autonomous operation, self-improvement, learning craft, resources,
  scorecard, winning solutions, and cross-domain playbook.
- `scripts/`: reusable monitoring, evaluation, Pareto-selection, and curator
  helpers.
- `agents/openai.yaml`: product discovery metadata for implicit and explicit use.

## Pokémon campaign policy

The current workspace campaign sets Pokémon TCG as `ACTIVE`; other competitions
are `MONITOR_ONLY`. The generic skill explains the states but does not name or
hard-code Pokémon. Workspace automation and campaign notes carry that choice.

## Verification

- Contract tests initially fail against the old skill, then pass after migration.
- Validate frontmatter and skill structure with `quick_validate.py`.
- Syntax-check shell and Python helpers.
- Search for stale `kaggle-practice-coach` paths.
- Exercise positive scenarios for a generic tabular competition and Pokémon, plus
  negative scenarios for infrastructure-only requests and prohibited tactics.
- Compare canonical and installed trees after deployment.

