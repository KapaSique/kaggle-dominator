# Evolution Engine Governance Card

## Purpose

The evolution engine turns measured, reusable evidence into a generated learning
entry. Agents may propose and judge; only the deterministic engine promotes.
fail closed on missing, stale, contradictory, unrepeatable, or unverifiable evidence.

## Approval boundary

Kaggle submissions, rule acceptance, team or settings changes, public
publishing/interactions, purchases, and GitHub writes always require explicit user approval. Local read-only recon, validation, and immutable evidence
artifacts may proceed inside the authorized campaign scope.

## Protected surfaces

Automatic promotion must never modify `SKILL.md`, YAML frontmatter,
`agents/openai.yaml`, credential logic, campaign authority, submission policy,
or non-generated references. The sole automatic destination is
`references/learned-playbook.md`, rendered from append-only local promotion
events. Promotion never pushes a commit or performs a GitHub write.

## Promotion gate

Promote only a succeeded candidate with verified metric direction, improvement
strictly above its noise floor, sufficient confirmations and distinct validation
regimes, no regressions or forbidden actions, only the allowed changed path,
bounded runtime, a fresh PASS verifier, and a blind comparator that selects the
challenger. Otherwise record a machine-readable rejection and leave canonical
guidance unchanged.

## Sealed provenance manifest

Before gate or promotion, the deterministic engine saves exactly one immutable
manifest for the opaque stable candidate token. It registers a unique
role+worker identity and output path for the proposer, verifier, and comparator;
each completed output and declared input artifact must have a SHA-256 and UTC
timestamp. The workers and output paths are pairwise distinct. The verifier and
comparator payload supplied to the gate must byte-match their registered,
non-symlinked JSON outputs. Missing, changed, stale, shared, or self-produced
artifacts reject the candidate.

The comparator gets identity-free packages with only `incumbent` and
`challenger` labels plus the opaque stable candidate token. It never receives a
model, author, proposer, or requested winner identity, and there is no A/B
mapping adapter. The engine verifies the sealed package hashes and rejects
identity-leakage fields before accepting the exact comparator output schema.

```json
{
  "run_id": "run-cand-1",
  "candidate_id": "cand-1",
  "created_at_utc": "2026-07-17T00:00:00Z",
  "artifacts": [
    {
      "role": "proposer",
      "worker_id": "proposer-1",
      "output_path": "artifacts/run-cand-1/proposer.json",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "created_at_utc": "2026-07-17T00:00:00Z",
      "terminal_status": "succeeded",
      "input_artifacts": [{"path": "artifacts/run-cand-1/raw-evidence.json", "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"}]
    }
  ],
  "comparator_package": {
    "candidate_token": "cand-1",
    "inputs": [
      {"label": "incumbent", "path": "artifacts/run-cand-1/blind-incumbent.json", "sha256": "1111111111111111111111111111111111111111111111111111111111111111"},
      {"label": "challenger", "path": "artifacts/run-cand-1/blind-challenger.json", "sha256": "2222222222222222222222222222222222222222222222222222222222222222"}
    ]
  }
}
```

The example elides the verifier and comparator registrations only for space;
the engine requires exactly all three roles and validates their identical
registration fields.
