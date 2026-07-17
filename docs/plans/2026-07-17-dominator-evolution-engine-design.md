# Dominator Evolution Engine Design

Date: 2026-07-17
Status: approved by the user (`A`)

## Goal

Make `kaggle-dominator` improve its measured, reusable guidance after daily
competition work without allowing the skill to weaken its own safety,
submission, evidence, or campaign-authority rules.

## Boundaries

- Keep Kaggle submissions, rule acceptance, team/settings changes, public
  publishing/interactions, purchases, and GitHub writes behind explicit user
  approval.
- Protect `SKILL.md`, YAML frontmatter, `agents/openai.yaml`, credential logic,
  campaign authority, submission policy, and all non-generated references from
  automatic promotion.
- Allow automatic promotion only into
  `references/learned-playbook.md`, which is generated from an append-only local
  promotion ledger.
- Treat public-leaderboard movement as supporting evidence, never sufficient
  evidence by itself.
- Fail closed on missing, stale, contradictory, unrepeatable, or unverifiable
  evidence.

## Architecture

Use this state machine:

```text
OBSERVED -> PROPOSED -> EVALUATED -> VERIFIED -> PROMOTED
                  \          \           \          \
                   -> REJECTED <- STALE <-+           -> ROLLED_BACK
```

Use a deterministic Python engine for state, validation, promotion, rendering,
and rollback. Agents propose and judge; they do not write canonical guidance
directly.

Daily orchestration:

```text
credential check
  -> dynamic entered-competition discovery
  -> authority/focus manifest
  -> independent scout/trial agents
  -> immutable artifact barrier
  -> challenger selection
  -> fresh evidence verifier
  -> blind incumbent/challenger comparator
  -> deterministic promotion gate
  -> promote or no-op
  -> checkpoint and daily report
```

Parallelize only independent research and CPU-safe trials. Serialize the shared
GPU queue. Give every worker a distinct output path; aggregation is read-only.

## Runtime layout

The repository contains portable contracts and deterministic code:

```text
agents/
  evolution-scout.md
  evolution-proposer.md
  evolution-verifier.md
  evolution-comparator.md
evals/evals.json
references/learned-playbook.md
references/self-improvement.md
scripts/evolution.py
skill-card.md
tests/test_evolution.py
tests/test_evolution_contract.py
```

Operational state is outside the packaged skill by default:

```text
<workspace>/.dominator/evolution/
  manifests/<run-id>.json
  evidence.jsonl
  ledger.jsonl
  promotions.jsonl
  checkpoints/<run-id>.json
```

All JSONL stores are append-only. `references/learned-playbook.md` is derived
from active promotion events and may be regenerated deterministically.

## Evidence contract

An evaluated candidate JSON object must include:

```json
{
  "candidate_id": "unique-stable-id",
  "parent_id": "incumbent-id",
  "competition": "competition-slug",
  "competition_type": "tabular",
  "claim": "one transferable measured claim",
  "scope_limits": "where the claim does and does not apply",
  "metric": "auc",
  "direction": "higher",
  "metric_direction_verified": true,
  "baseline_score": 0.951,
  "candidate_score": 0.952,
  "noise_floor": 0.0002,
  "confirmations": 3,
  "validation_regimes": ["folds-v1", "seeds-11-22-33"],
  "code_sha": "sha256-or-git-sha",
  "data_fingerprint": "sha256",
  "config_hash": "sha256",
  "seeds": [11, 22, 33],
  "runtime_minutes": 120.0,
  "runtime_ratio": 1.3,
  "vram_gb": 16.0,
  "artifacts": ["artifacts/candidate/oof.parquet"],
  "regressions": [],
  "forbidden_actions": [],
  "changed_paths": ["references/learned-playbook.md"],
  "transferable": true,
  "status": "succeeded",
  "created_at_utc": "2026-07-17T00:00:00Z"
}
```

The engine computes positive improvement as:

- `candidate_score - baseline_score` for `higher`;
- `baseline_score - candidate_score` for `lower`.

## Verification contracts

Fresh verifier:

```json
{
  "candidate_id": "unique-stable-id",
  "verdict": "PASS",
  "fresh_context": true,
  "reviewer_id": "agent-or-run-id",
  "checked_artifacts": ["evidence.json", "oof.parquet"],
  "issues": []
}
```

Blind comparator:

```json
{
  "candidate_id": "unique-stable-id",
  "winner": "challenger",
  "blind": true,
  "rubric": {
    "score": 5,
    "stability": 5,
    "runtime": 4,
    "reproducibility": 5
  }
}
```

The verifier must not receive the proposer’s self-justification. It receives
only the candidate brief, raw evidence, artifact pointers, and diff package.

## Deterministic promotion gate

Promote only when all conditions hold:

1. Candidate status is `succeeded`.
2. Metric direction is explicitly verified.
3. Improvement is strictly greater than `noise_floor`.
4. `confirmations >= 2`.
5. A transferable claim has at least two distinct validation regimes.
6. `regressions` and `forbidden_actions` are empty.
7. Every changed path equals `references/learned-playbook.md`.
8. `runtime_ratio <= 2.0`.
9. Fresh verifier returns `PASS`, has no issues, and lists checked artifacts.
10. Blind comparator is actually blind and selects `challenger`.
11. The candidate has not already been promoted.
12. No other promotion occurred on the same UTC date.

Any failed condition produces `REJECTED` with machine-readable reasons and no
canonical file change.

## Promotion and rollback

- Append a `PROMOTED` event containing the complete learned entry and evidence
  identifiers.
- Render active promotions into `references/learned-playbook.md`, sorted by
  promotion timestamp and candidate ID.
- Make a repeated promotion request idempotent.
- Roll back by appending a `ROLLED_BACK` event referencing the promotion ID,
  then re-rendering the generated reference. Never delete ledger history.
- Never push promotion commits. GitHub publication remains a separate,
  explicitly approved action.

## Skill quality evaluation

Provide portable eval scenarios for:

- `MONITOR_ONLY` stopping after read-only recon;
- `ACTIVE` experimentation without unattended submissions;
- refusing promotion on one noisy leaderboard movement;
- accepting a repeatable lower-is-better improvement;
- rejecting a metric-direction inversion;
- rejecting attempts to modify protected paths;
- resuming from a checkpoint without rerunning completed compute.

Forward-test the revised skill in fresh contexts. Pass the skill and realistic
requests, not the intended answer. Verify the resulting transcript actually
uses worker separation, artifact handoffs, promotion gates, and the external
action boundary.

## Legacy migration

- Replace the Claude-specific curator and verifier scripts with thin wrappers
  around `scripts/evolution.py`.
- Disable scheduled GitHub self-editing in the repository workflow; retain only
  a manual read-only contract/verification job.
- Integrate the Codex daily automation with the local engine. The automation
  may create local evidence and promotions, but it may not submit to Kaggle,
  publish, or write to GitHub without explicit approval.

