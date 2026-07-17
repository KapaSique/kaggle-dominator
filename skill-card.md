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
timestamp. The workers and output paths are pairwise distinct; `reviewer_id`
must equal the verifier worker ID and therefore differ from proposer and
comparator workers. Every artifact path component must be non-symlinked. The
verifier and comparator payload supplied to the gate must byte-match their
registered JSON outputs. Missing, changed, stale, shared, aliased, or
self-produced artifacts reject the candidate.

Verifier inputs are separately sealed source artifacts with `kind` one of
`raw_evidence`, `artifact_pointer`, or `diff_package`, origin `source`, a
SHA-256, and a UTC timestamp. Role outputs and byte-identical or inode-alias
copies of the proposer output are never valid verifier inputs. The proposer
output must be after the manifest; verifier and comparator outputs must be
strictly after the proposer and every declared input. The verifier's
`checked_artifacts` must normalize exactly to its registered input paths.

The comparator gets identity-free packages with only `incumbent` and
`challenger` labels plus the opaque stable candidate token. It never receives a
model, author, proposer, or requested winner identity, and there is no A/B
mapping adapter. The engine verifies the sealed package hashes and rejects
identity-leakage fields before accepting the exact comparator output schema.
It cannot prove that arbitrary free-text values contain no identity, so upstream
redaction and opaque-token generation remain an orchestration policy.

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
      "created_at_utc": "2026-07-17T00:02:00Z",
      "terminal_status": "succeeded",
      "input_artifacts": [{"kind": "raw_evidence", "origin": "source", "path": "artifacts/run-cand-1/raw-evidence.json", "sha256": "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789", "created_at_utc": "2026-07-17T00:01:00Z"}]
    }
  ],
  "comparator_package": {
    "candidate_token": "cand-1",
    "inputs": [
      {"label": "incumbent", "kind": "artifact_pointer", "origin": "source", "path": "artifacts/run-cand-1/blind-incumbent.json", "sha256": "1111111111111111111111111111111111111111111111111111111111111111", "created_at_utc": "2026-07-17T00:01:00Z"},
      {"label": "challenger", "kind": "artifact_pointer", "origin": "source", "path": "artifacts/run-cand-1/blind-challenger.json", "sha256": "2222222222222222222222222222222222222222222222222222222222222222", "created_at_utc": "2026-07-17T00:01:00Z"}
    ]
  }
}
```

The example elides the verifier and comparator registrations only for space;
the engine requires exactly all three roles and validates their identical
registration fields.
