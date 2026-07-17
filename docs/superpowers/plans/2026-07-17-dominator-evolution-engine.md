# Dominator Evolution Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, evidence-gated self-improvement engine for
`kaggle-dominator` and connect it to the daily Codex automation without
authorizing Kaggle submissions, public actions, or GitHub writes.

**Architecture:** Store operational history in append-only JSONL ledgers and
derive a single generated learned reference from active promotions. Agents
produce immutable evidence, verifier, and blind-comparison artifacts; a Python
CLI enforces the state machine and all promotion/rollback rules.

**Tech Stack:** Python 3 standard library, `unittest`, Markdown/JSON contracts,
shell wrappers, Codex automation.

## Global Constraints

- Automatic promotion may write only `references/learned-playbook.md`.
- Never autonomously modify `SKILL.md`, frontmatter, `agents/openai.yaml`,
  campaign authority, credentials, submission policy, or GitHub.
- Never submit to Kaggle, accept rules, modify teams/settings, publish, upvote,
  create public kernels, or spend paid compute.
- Public leaderboard movement is never sufficient promotion evidence by itself.
- All runtime JSONL ledgers are append-only and all commands are idempotent.
- Promotion is at most one per UTC date.
- Missing or contradictory evidence fails closed.
- Use tests first and confirm every new test fails for the intended reason.

---

### Task 1: Evidence registry and event-sourced state machine

**Files:**
- Create: `tests/test_evolution.py`
- Create: `scripts/evolution.py`

**Interfaces:**
- Produces: `EvolutionStore(root: Path, skill_root: Path)`
- Produces: `record_evidence(evidence: dict) -> dict`
- Produces: `latest_state(candidate_id: str) -> str | None`
- Produces: CLI commands `plan`, `record`, and `status`
- Stores: `manifests/*.json`, `evidence.jsonl`, and `ledger.jsonl`

- [ ] **Step 1: Write failing state-machine tests**

Add tests that import `scripts/evolution.py` with `importlib.util` and verify:

```python
store = EvolutionStore(state_dir, skill_root)
record = store.record_evidence(valid_evidence())
self.assertEqual(record["state"], "EVALUATED")
self.assertEqual(store.latest_state("cand-1"), "EVALUATED")
self.assertEqual(len(read_jsonl(state_dir / "evidence.jsonl")), 1)
```

Also verify duplicate `candidate_id` recording is a no-op, manifests are copied
under their `run_id`, UTC timestamps are normalized, and malformed JSON/schema
raises `EvolutionError`.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python3 -m unittest -v tests/test_evolution.py
```

Expected: import/file failure because `scripts/evolution.py` does not exist.

- [ ] **Step 3: Implement the minimal store and CLI**

Implement these exact public names:

```python
class EvolutionError(ValueError): ...
class EvolutionStore:
    def __init__(self, root: Path, skill_root: Path) -> None: ...
    def save_manifest(self, manifest: dict) -> Path: ...
    def record_evidence(self, evidence: dict) -> dict: ...
    def latest_state(self, candidate_id: str) -> str | None: ...
    def status(self) -> dict: ...

def read_json(path: Path) -> dict: ...
def read_jsonl(path: Path) -> list[dict]: ...
def append_jsonl(path: Path, payload: dict) -> None: ...
def validate_evidence(evidence: dict) -> None: ...
def build_parser() -> argparse.ArgumentParser: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

`record_evidence` appends one evidence record and two ledger events:
`OBSERVED`, then `EVALUATED`. A duplicate candidate returns the existing
record without appending.

- [ ] **Step 4: Run Task 1 tests and full baseline**

```bash
python3 -m unittest -v tests/test_evolution.py
python3 -m unittest discover -v tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/evolution.py tests/test_evolution.py
git commit -m "feat: add evolution evidence registry"
```

### Task 2: Fail-closed verification, promotion, and rollback

**Files:**
- Modify: `tests/test_evolution.py`
- Modify: `scripts/evolution.py`
- Create: `references/learned-playbook.md`

**Interfaces:**
- Produces: `gate_candidate(candidate_id, verification, comparison) -> GateResult`
- Produces: `promote(candidate_id, verification, comparison) -> dict`
- Produces: `rollback(promotion_id: str, reason: str) -> dict`
- Produces: CLI commands `gate`, `promote`, and `rollback`
- Stores: `promotions.jsonl`

- [ ] **Step 1: Write failing gate and rollback tests**

Cover every deterministic rule from the design:

```python
result = store.gate_candidate("cand-1", passing_verifier(), passing_comparator())
self.assertTrue(result.passed)

promotion = store.promote("cand-1", passing_verifier(), passing_comparator())
self.assertEqual(store.latest_state("cand-1"), "PROMOTED")
self.assertIn("cand-1", learned_playbook.read_text())

store.rollback(promotion["promotion_id"], "regression discovered")
self.assertEqual(store.latest_state("cand-1"), "ROLLED_BACK")
self.assertNotIn(valid_evidence()["claim"], learned_playbook.read_text())
```

Add one test each for wrong metric direction, delta at/below noise,
confirmations below two, insufficient transferable regimes, regressions,
forbidden actions, protected path changes, runtime ratio above two, stale or
non-fresh verifier, verifier issues, non-blind comparator, incumbent winner,
duplicate promotion, and second same-day promotion.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
python3 -m unittest -v tests/test_evolution.py
```

Expected: missing gate/promotion methods.

- [ ] **Step 3: Implement gate, generated rendering, and rollback**

Add:

```python
@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: tuple[str, ...]
    improvement: float

ALLOWED_PROMOTION_PATH = "references/learned-playbook.md"
MAX_RUNTIME_RATIO = 2.0
MIN_CONFIRMATIONS = 2
```

Render the learned reference from active `PROMOTED` events minus referenced
`ROLLED_BACK` events. Write through a temporary file followed by
`Path.replace()` for atomicity. A failed gate appends `REJECTED` and changes no
canonical file.

- [ ] **Step 4: Run Task 2 tests and full suite**

```bash
python3 -m unittest -v tests/test_evolution.py
python3 -m unittest discover -v tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/evolution.py tests/test_evolution.py references/learned-playbook.md
git commit -m "feat: gate and promote measured learnings"
```

### Task 3: Agent contracts, portable evals, and skill governance

**Files:**
- Create: `agents/evolution-scout.md`
- Create: `agents/evolution-proposer.md`
- Create: `agents/evolution-verifier.md`
- Create: `agents/evolution-comparator.md`
- Create: `evals/evals.json`
- Create: `skill-card.md`
- Create: `tests/test_evolution_contract.py`

**Interfaces:**
- Agent outputs use the exact evidence, verification, and comparator schemas in
  the approved design.
- `evals/evals.json` contains realistic prompts with expected behavior and
  success criteria.

- [ ] **Step 1: Write failing contract tests**

Verify all files exist; JSON parses; eval IDs are unique; the four required
scenarios `monitor-only`, `active-no-submit`, `reject-noisy-lb`, and
`reject-protected-path` exist; verifier requires fresh context and raw
artifacts; comparator requires blindness; governance protects external actions
and core skill files.

- [ ] **Step 2: Run tests and confirm RED**

```bash
python3 -m unittest -v tests/test_evolution_contract.py
```

Expected: missing contract files.

- [ ] **Step 3: Add minimal contracts**

Keep each agent prompt focused on one role and one immutable output file.
Require terminal status, artifact paths, UTC timestamps, and machine-readable
JSON. Prohibit shared mutable output paths and self-verification.

- [ ] **Step 4: Run contract and full tests**

```bash
python3 -m unittest -v tests/test_evolution_contract.py
python3 -m unittest discover -v tests
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add agents/evolution-*.md evals/evals.json skill-card.md tests/test_evolution_contract.py
git commit -m "docs: add evolution agent and eval contracts"
```

### Task 4: Replace legacy curator and integrate the skill

**Files:**
- Modify: `SKILL.md`
- Modify: `references/self-improvement.md`
- Modify: `scripts/skill_curator.sh`
- Modify: `scripts/curator_verify.sh`
- Modify: `scripts/skill-curator.yml`
- Modify: `.github/workflows/skill-curator.yml`
- Modify: `tests/test_skill_contract.py`
- Modify: `tests/test_workflow_policy.py`

**Interfaces:**
- `scripts/skill_curator.sh` forwards arguments to `python3 scripts/evolution.py`.
- `scripts/curator_verify.sh` forwards to `python3 scripts/evolution.py gate`.
- GitHub workflow is manual-only, `contents: read`, and runs validation without
  commits, pushes, PRs, or merges.

- [ ] **Step 1: Write failing integration-policy tests**

Require `SKILL.md` to route measured self-improvement to both
`references/self-improvement.md` and `references/learned-playbook.md`.
Require runtime scripts to contain no `claude -p`, no fixed competition slug,
and no GitHub write commands. Require both curator workflow copies to have no
`schedule`, `contents: write`, `git push`, `gh pr create`, or `gh pr merge`.

- [ ] **Step 2: Run tests and confirm RED**

```bash
python3 -m unittest -v tests/test_skill_contract.py tests/test_workflow_policy.py
```

Expected: failures on the legacy Claude curator and scheduled write workflow.

- [ ] **Step 3: Update documentation, wrappers, and workflows**

Keep `SKILL.md` concise: add only the route and promotion boundary. Put the full
orchestration, state machine, schemas, gates, checkpointing, and rollback
workflow in `references/self-improvement.md`.

- [ ] **Step 4: Run integration tests and validation**

```bash
python3 -m unittest discover -v tests
python3 /Users/artemcike/.codex/skills/.system/skill-creator/scripts/quick_validate.py .
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py tests/*.py
```

Expected: all commands exit zero.

- [ ] **Step 5: Commit**

```bash
git add SKILL.md references/self-improvement.md scripts/skill_curator.sh \
  scripts/curator_verify.sh scripts/skill-curator.yml \
  .github/workflows/skill-curator.yml tests/test_skill_contract.py \
  tests/test_workflow_policy.py
git commit -m "feat: integrate evidence-gated self-improvement"
```

### Task 5: Package, forward-test, and connect the Codex automation

**Files:**
- Regenerate: `kaggle-dominator.skill`
- External update: Codex automation `kaggle-daily-competition-report`

**Interfaces:**
- Automation calls the local evolution workflow after daily experiment
  verification and before the Russian summary.
- Automation reports `PROMOTED`, `REJECTED`, or `NO-OP`.

- [ ] **Step 1: Build the package deterministically**

Create a fresh archive containing the skill runtime files while excluding git
metadata, tests, caches, local state, and docs.

- [ ] **Step 2: Run full verification**

```bash
python3 -m unittest discover -v tests
python3 /Users/artemcike/.codex/skills/.system/skill-creator/scripts/quick_validate.py .
bash -n scripts/*.sh
python3 -m py_compile scripts/*.py tests/*.py
unzip -t kaggle-dominator.skill
```

- [ ] **Step 3: Run fresh-context forward tests**

Dispatch fresh agents with realistic `MONITOR_ONLY`, `ACTIVE`, and
self-improvement requests. Pass only the skill path and task. Verify they honor
dynamic discovery, artifact separation, promotion gates, and the external
action boundary.

- [ ] **Step 4: Update the Codex automation**

Use the automation tool, preserve the daily schedule and resource envelope, and
add the deterministic evolution stage. Do not enable any GitHub workflow.

- [ ] **Step 5: Commit package**

```bash
git add kaggle-dominator.skill
git commit -m "build: package dominator evolution engine"
```

