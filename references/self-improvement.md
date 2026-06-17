# Self-improvement — the skill that curates itself

The deepest level of the iceberg (HUMANITY: *agents managing agents*, *agent-built tooling*, *self-improving loops*) turned on the skill itself. "CAPTURE WHAT YOU LEARN" in SKILL.md is the *manual, in-session* reflex — write a measured insight the moment you hit it. This file closes that into an *autonomous loop*: a curator agent periodically harvests the measured lessons from real battles and wires them into the skill on its own.

**This is genuinely dangerous** — an agent editing its own instructions can drift, bloat, or corrupt itself. So the entire design is discipline-first. The same thing that makes the skill *ascended* and not *degenerate*: autonomy built on hard guardrails, never instead of them.

Read this when the user wants the skill to keep improving hands-off, or asks about self-curation / the curator loop.

## Contents
- [The two paths an insight enters the skill](#the-two-paths-an-insight-enters-the-skill)
- [The curator loop](#the-curator-loop)
- [Orchestration shape (agents managing agents)](#orchestration-shape-agents-managing-agents)
- [Guardrails (non-negotiable — it edits itself)](#guardrails-non-negotiable--it-edits-itself)
- [Running it](#running-it)

---

## The two paths an insight enters the skill

1. **In-session (manual).** Mid-battle, Opus hits a measured "oh — *that's* what actually works" and writes it immediately (the CAPTURE rule). Fast, but only catches what one session noticed.
2. **Curator (autonomous).** A scheduled agent re-reads the *whole* battle history across all competitions and extracts what the live sessions missed — especially **cross-competition patterns** that no single session can see. The "over-engineering past the peak = 6/8 comps" anti-pattern only became visible when an agent audited eight competitions at once. That is the curator's unique value: the aggregate prozrenie.

Both write to the same places (transferable → `grandmaster-playbook.md`; comp-specific → memory) under the same rule: **only what's measured.**

## The curator loop

A headless agent (`scripts/skill_curator.sh`), on a schedule or on demand:

```
1. SCAN    — read fresh signal: kaggle competitions submissions for each active comp
             (score JUMPS and DROPS + the technique in the submission description),
             memory notes, results.csv. One scanner sub-agent per competition.
2. EXTRACT — for each measured movement not yet reflected in the playbook, formulate
             ONE transferable bullet: "date — [type] insight (evidence with the number)".
3. DEDUPE  — barrier: gather all candidates, drop any already in Battle-proven additions,
             and detect repeats (a pattern seen in a new comp bumps an anti-pattern counter).
4. WRITE   — append the survivors to Battle-proven additions (newest first); update the
             anti-pattern counts in SKILL.md if a known pattern recurred.
5. SHIP    — repackage the .skill (must VALIDATE), commit to the skill repo with a clear
             diff. The git history is the audit trail; a human can read or revert any edit.
```

The loop never invents — it only transcribes what the leaderboard already proved.

## Orchestration shape (agents managing agents)

The curator is an orchestrator, not a monolith:

- **Scanner agents** — one per active competition, dispatched in parallel (isolated context, the dispatch contract from `autonomous.md`). Each returns `{comp, [new score movements], technique, evidence}`. They're independent domains → genuine parallelism.
- **Synthesizer** — a barrier after the scanners (it needs *all* findings to dedupe against the current playbook and to spot cross-comp repeats). It classifies transferable vs comp-specific, writes the bullets, bumps counters.
- **Validator** — repackages and confirms the skill still validates before anything is committed.

This mirrors the skill's own teaching: parallel where independent (scanners), barrier where you need the whole set (synthesis), diversity of inputs → a pattern no single view catches.

## Guardrails (non-negotiable — it edits itself)

An agent rewriting its own brain is the most dangerous thing in this skill. These are not optional:

- **Only measured facts.** A number, an LB movement, a confirmed mechanism. Never a hunch. If there's no evidence, it doesn't get written.
- **Append-only to insights; never autonomously rewrite or delete existing rules.** The curator may *add* to Battle-proven additions and *increment* an anti-pattern counter. Changing the constitution, the iron rules, or any existing guidance is a **human** decision. Structural edits are out of scope for the loop.
- **Never touch the `description`.** Triggering is critical and easy to break; the curator leaves frontmatter alone.
- **Git is the audit trail.** Every self-edit is a separate commit with a readable diff. Prefer a PR over a direct push so a human reviews before merge; at minimum, a human can revert.
- **Anti-bloat / consolidation.** Battle-proven additions has a soft cap (~25 bullets). When it overflows, the curator's job flips from *adding* to *consolidating* — merge several specific bullets into one general principle (still measured, citing the cases). The skill must get *sharper* over time, not just longer. A skill that only grows is a skill that rots.
- **Validate before commit.** Run the packager's validation after every edit; never commit an invalid skill (e.g. a description pushed over the 1024-char limit by an unrelated edit).
- **Idempotent.** Re-running the curator with no new battles changes nothing — it must recognize what it already wrote.

## Running it

- **On demand:** `scripts/skill_curator.sh` — point it at your active competition slugs; it harvests, writes, validates, and commits.
- **Scheduled:** `scripts/skill-curator.yml` (GitHub Actions) — runs weekly, opens a PR with that week's measured insights for you to merge. Needs the same secrets as `nightly-agent.yml` (`ANTHROPIC_API_KEY`) plus push/PR permission on the skill repo.
- **The honest expectation:** most weeks it adds 0–2 lines, occasionally consolidates. That's success — the skill compounds slowly from real results, not from generated filler. If it ever wants to add ten bullets in a week, that's the bloat alarm, not a win.
