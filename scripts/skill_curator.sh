#!/usr/bin/env bash
# skill_curator.sh — the skill curates itself.
#
# A headless agent harvests MEASURED insights from your real battles and wires them
# into the skill: it scans submission histories for score jumps/drops + the technique
# behind them, appends transferable lessons to the playbook's "Battle-proven additions",
# bumps anti-pattern counters when a pattern recurs, repackages (must validate), and
# commits with a readable diff. See references/self-improvement.md for the full protocol.
#
# It NEVER rewrites existing rules, never touches the description, only adds MEASURED facts.
#
# Usage: ./skill_curator.sh <skill-dir> <comp-slug> [<comp-slug> ...]
# Example: ./skill_curator.sh ~/.claude/skills/kaggle-dominator playground-series-s6e6 orbit-wars
set -euo pipefail

SKILL_DIR="${1:?usage: skill_curator.sh <skill-dir> <comp-slug> [more...]}"; shift
COMPS="$*"
[ -n "$COMPS" ] || { echo "[curator] no competition slugs given"; exit 1; }

# Gather fresh signal: per-comp submission history (jumps/drops live in the score column + description).
SIGNAL="$(mktemp)"
for c in $COMPS; do
  echo "===== $c =====" >> "$SIGNAL"
  kaggle competitions submissions "$c" 2>/dev/null | grep -v "Warning" | head -20 >> "$SIGNAL" || true
done

PLAYBOOK="$SKILL_DIR/references/grandmaster-playbook.md"

claude -p "You are the kaggle-dominator curator. Your job: wire MEASURED battle insights into this skill — nothing invented.

Recent submission signal across active competitions is below between <signal> tags. The skill's
already-captured insights are in $PLAYBOOK under 'Battle-proven additions'.

<signal>
$(cat "$SIGNAL")
</signal>

Do exactly this:
0. For EACH competition, FIRST confirm the metric direction: run 'kaggle competitions leaderboard <slug> --show' and look at the top vs bottom score. Higher-at-top = higher-is-better; lower-at-top = lower-is-better. A 'jump' or 'drop' is meaningless — and easy to record backwards — until you know which way the metric runs. NEVER infer direction from a submission description.
1. In the signal, using the confirmed direction, find score JUMPS and DROPS and read the technique from each submission's description.
2. For each MEASURED movement NOT already reflected in the playbook, draft ONE transferable bullet:
   'YYYY-MM — [comp type] insight (the evidence with the number)'. Skip anything already present.
3. Append the survivors to 'Battle-proven additions' (newest first). If a known pattern recurs in a new
   comp (e.g. over-engineering past the peak), increment its counter in SKILL.md (e.g. 6/8 -> 7/9).
4. HARD RULES: only measured facts; do NOT rewrite or delete existing rules; do NOT touch the YAML
   description; if 'Battle-proven additions' exceeds ~25 bullets, CONSOLIDATE several specific ones into
   one general measured principle instead of adding more.
5. If nothing new and measured exists, change NOTHING and say so.

Report: the exact bullets you added or 'no change'." --allowedTools "Bash Read Edit Grep"

rm -f "$SIGNAL"

# Repackage + validate before committing. Adjust SKILL_CREATOR_DIR to your environment.
if [ -n "${SKILL_CREATOR_DIR:-}" ] && [ -d "$SKILL_CREATOR_DIR" ]; then
  ( cd "$SKILL_CREATOR_DIR" && python -m scripts.package_skill "$SKILL_DIR" 2>&1 | grep -E "valid|Successfully|too long" ) || {
    echo "[curator] VALIDATION FAILED — not committing"; exit 1; }
fi

# Do NOT commit here — leave edits in the working tree for the caller (CI opens a PR; a human
# reviews and merges). Committing here too caused a double-commit / branch race in CI.
if [ -n "$(git -C "$SKILL_DIR" status --porcelain 2>/dev/null)" ]; then
  echo "[curator] edits staged in the working tree — review the diff, then commit/PR:"
  git -C "$SKILL_DIR" --no-pager diff --stat
else
  echo "[curator] no change — skill already reflects the battles"
fi
