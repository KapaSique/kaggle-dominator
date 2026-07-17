# Autonomous work — bounded local evolution

Use unattended work only through the trusted Codex daily orchestration and
`scripts/evolution.py`. The orchestration may collect local, immutable
artifacts and ask independent workers for bounded analysis; it does not make
public, account, or repository changes on its own.

## Safe route

1. Define one read-only recon or one reproducible local trial with an explicit
   time, compute, and API budget.
2. Give each independent worker a separate artifact path. Keep shared compute
   queues and aggregation serialized.
3. Checkpoint completed work and stop at the first budget, safety, or evidence
   failure. A stop flag must halt before the next work unit.
4. Preserve raw evidence and artifact hashes under the evolution state root.
   Route state changes through `scripts/evolution.py`: `plan`, `record`,
   `gate`, and, only after the deterministic gate passes, `promote`.
5. Read the resulting status and report the decision, reasons, costs, and
   next bounded action. Missing, stale, contradictory, or unverifiable
   evidence is a no-op.

The engine serializes shared state. Workers must never edit evidence ledgers,
manifests, or generated guidance directly.

## Learning boundary

Only `references/learned-playbook.md` may receive an automatic learning update,
and it is generated solely from the append-only promotion history by
`scripts/evolution.py`. All other references, the skill contract, campaign
authority, and credentials are protected. A measured observation is not a
promotion until the fresh verifier, blind comparator, provenance checks, and
deterministic gate have accepted it.

## Exact approval boundary

Obtain explicit approval for the exact artifact and action before any Kaggle
submission, public publishing or interaction, account/settings change,
purchase, or GitHub write. Local evidence collection and a generated learning
update do not grant that approval. If approval is absent or ambiguous, stop
after the local report and present the proposed action for review.

## Useful operating discipline

- Keep work units narrow: one hypothesis, one reproducible trial, and one
  artifact handoff.
- Parallelize only independent research or CPU-safe trials. Use a barrier
  before any comparison or aggregation.
- Treat a fixed budget and checkpoint as correctness requirements, not merely
  cost controls.
- Prefer a clean no-op to an unverified conclusion. The morning report should
  make it possible to reconstruct every decision from immutable artifacts.

The retired `kaggle_monitor.sh`, `kaggle_eval_loop.sh`, and `re_recon.sh`
entry points intentionally fail closed. They are not orchestration routes.
