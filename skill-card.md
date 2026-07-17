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
