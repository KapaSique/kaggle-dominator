# Evidence-gated self-improvement

Read this with `references/learned-playbook.md` when measured work may produce a
reusable lesson. The latter is generated from active promotions and is the only
file automatic promotion may change. `SKILL.md`, its frontmatter, safety and
policy rules, authority boundaries, credentials, campaign control, submission
policy, and every other reference are protected.

The local deterministic engine is `scripts/evolution.py`. It records and renders
evidence; it never submits to Kaggle, accepts rules, changes teams/settings,
publishes, spends money, or writes to GitHub. Those external actions always need
explicit user approval.

## Trusted orchestration and separated roles

A trusted root Codex orchestrator owns the run, campaign authority/focus manifest,
opaque candidate token, checkpoint, and unique immutable output paths. It keeps
these roles separated:

- **Scout** performs read-only, campaign-scoped recon. `MONITOR_ONLY` stops after
  recon and returns a resume trigger.
- **Proposer** turns immutable scout/trial inputs into one evaluated, transferable
  candidate; it does not modify guidance.
- **Fresh verifier** receives only candidate brief, raw evidence, artifact pointers,
  and diff package—not proposer justification or identity—and emits a verdict.
- **Blind comparator** receives only identity-free `incumbent` and `challenger`
  packages plus the opaque candidate token, then selects a winner or no-decision.

No role self-verifies, shares a mutable output path, promotes, submits, publishes,
or changes canonical guidance. Parallelize independent scouts/trials only; serialize
the shared GPU queue and all state transitions.

The engine verifies structural provenance: declared role/worker separation, unique
paths, source artifact types, hashes, UTC ordering, sealed payload equality, verifier
bindings, and protected changed paths. It does **not** attest actual Codex
fresh-context or worker identity, and it cannot prove semantic redaction inside
arbitrary free text. Those are trusted-root-orchestrator responsibilities, not
signed attestations.

## State and immutable storage

The approved full **orchestrator/checkpoint lifecycle** is:

```text
OBSERVED -> PROPOSED -> EVALUATED -> VERIFIED -> PROMOTED
               |            |            |
               +-----> STALE+------------+
               +-----> REJECTED <--------+
PROMOTED -> ROLLED_BACK
```

`PROPOSED` and `STALE` are orchestrator/checkpoint lifecycle labels, not claims that
the persisted engine ledger appends those events. The trusted root records
`PROPOSED` after sealing a candidate brief and records `STALE` when a source,
artifact, or review expires before a usable gate. The persisted engine ledger records
`OBSERVED` then `EVALUATED` for valid evidence, `VERIFIED` for a passing gate,
`REJECTED` for a failed (including stale) gate, and `PROMOTED`/`ROLLED_BACK` for
their corresponding operations. A verifier artifact with verdict `STALE` therefore
fails closed into persisted `REJECTED`; it is not falsely represented as an engine
`STALE` event. State lives outside the packaged skill by default:

```text
<workspace>/.dominator/evolution/
  manifests/<run-id>.json
  artifacts/<run-id>/...
  evidence.jsonl
  ledger.jsonl
  promotions.jsonl
  checkpoints/<run-id>.json
```

Manifests, artifacts, evidence, ledger, promotions, and checkpoints are immutable
or append-only. Every worker receives a distinct artifact path; aggregation reads
sealed artifacts rather than editing them. The root orchestrator records completed
work in a checkpoint, so a resumed run validates existing hashes/statuses and does
not rerun completed compute.

## Exact contracts

An evaluated candidate has exactly the engine evidence fields below (values must be
finite where numeric and timestamps are timezone-aware UTC):

```json
{
  "candidate_id": "unique-stable-id",
  "parent_id": "incumbent-id",
  "competition": "competition-slug",
  "competition_type": "tabular",
  "claim": "one transferable measured claim",
  "scope_limits": "where it applies and does not apply",
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
  "artifacts": ["artifacts/run-1/oof.parquet"],
  "regressions": [],
  "forbidden_actions": [],
  "changed_paths": ["references/learned-playbook.md"],
  "transferable": true,
  "status": "succeeded",
  "created_at_utc": "2026-07-17T00:00:00Z"
}
```

The engine calculates improvement as `candidate - baseline` for `higher` and
`baseline - candidate` for `lower`; public-leaderboard movement is supporting
evidence, never enough by itself. The fresh verifier and blind comparator schemas
are exactly:

```json
{"candidate_id":"unique-stable-id","verdict":"PASS","fresh_context":true,"reviewer_id":"verifier-worker","checked_artifacts":["artifacts/run-1/evidence.json"],"issues":[]}
```

```json
{"candidate_id":"unique-stable-id","winner":"challenger","blind":true,"rubric":{"score":5,"stability":5,"runtime":4,"reproducibility":5}}
```

Before a gate, the manifest registers exactly one proposer, verifier, and comparator
with pairwise distinct worker IDs and outputs. Each registration has a SHA-256,
UTC timestamp after the manifest, terminal status, and typed source inputs
(`raw_evidence`, `artifact_pointer`, or `diff_package`, all with origin `source`).
Verifier inputs cannot be proposer output or an alias; `reviewer_id` and
`checked_artifacts` must match its registration. Comparator inputs are exactly the
two sealed `incumbent`/`challenger` packages and contain no identity-bearing fields.

## Deterministic promotion gate

`PROMOTED` requires every condition below; otherwise append `REJECTED` and leave
the learned playbook unchanged:

1. Candidate `status` is `succeeded` and metric direction is explicitly verified.
2. Direction-aware improvement is strictly greater than `noise_floor`.
3. `confirmations >= 2`; a transferable claim has at least two distinct validation regimes.
4. `regressions` and `forbidden_actions` are empty.
5. Every changed path is exactly `references/learned-playbook.md`.
6. `runtime_ratio <= 2.0`.
7. The fresh verifier returns `PASS`, `fresh_context: true`, no issues, and registered checked artifacts.
8. The comparator is blind and selects `challenger`.
9. Sealed provenance is complete, unchanged, non-stale, correctly ordered, and role-separated.
10. The candidate has not already been promoted and no other promotion occurred on that UTC date.

Promotion appends a complete event with its evidence identifiers, then atomically
renders active promotions into `references/learned-playbook.md`, ordered by
promotion timestamp and candidate ID. Repeating the same request is idempotent.
There is at most one promotion per UTC day.

## Rollback and local commands

On regression, append a `ROLLED_BACK` event referencing the `promotion_id` and a
reason, then re-render. Never delete evidence or history. Resume from the latest
checkpoint/ledger state rather than recreating past events or recomputing sealed
artifacts.

The legacy curator entry points are deliberately thin local forwarders:

```text
scripts/skill_curator.sh [engine arguments...]        -> python3 scripts/evolution.py
scripts/curator_verify.sh [gate arguments...]         -> python3 scripts/evolution.py gate
```

Use `plan`, `record`, `gate`, `promote`, `rollback`, and `status` with local JSON
artifacts. The bundled curator workflow is manual, `contents: read`, and validation
only; it does not create commits, push, open/merge pull requests, use secrets, or
run model/Kaggle/GitHub actions.
