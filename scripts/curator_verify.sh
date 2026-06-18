#!/usr/bin/env bash
# curator_verify.sh — INDEPENDENT fresh-context verifier for a curator change.
#
# It did NOT write the change. It reads the git diff on the current branch, applies the
# self-improvement guardrails, and prints VERDICT=SAFE or VERDICT=UNSAFE. The workflow
# auto-merges ONLY on SAFE; anything else is left as a PR for a human. This is the
# "agents-managing-agents" check that stops the curator writing a wrong insight to the skill
# (it once inverted a metric direction — this is the guard against a repeat).
#
# Requires: claude CLI (ANTHROPIC_API_KEY), kaggle CLI (read-only), run inside the repo on the curator branch.
set -uo pipefail

PROMPT="You are an INDEPENDENT verifier for an autonomous skill-curator change on the current git branch. You did NOT write it; do not assume it is correct.

First run: git diff origin/main...HEAD   (to see exactly what changed).

Apply these guardrails STRICTLY:
1. ONLY references/grandmaster-playbook.md 'Battle-proven additions' or anti-pattern bullets may change. Any change to SKILL.md constitution text, the skill description, or other structural areas => UNSAFE.
2. Every added insight must cite a REAL measured number. Vague claims or hunches => UNSAFE.
3. For any 'X beat Y' / direction claim, CONFIRM the metric direction (top row = best) by running: python3 -m kaggle competitions leaderboard <slug> --show | head. If the diff's claim contradicts the leaderboard ordering, or you cannot confirm it => UNSAFE.
4. Append-only and small (<= 4 bullets). Mass rewrite, reordering, or deletions => UNSAFE.

Use the Bash tool for 'git diff' and 'kaggle ... leaderboard' reads ONLY. Never run 'kaggle competitions submit' or 'kaggle kernels push'.

Output EXACTLY one final line, starting with the token:
VERDICT=SAFE <one-sentence reason>
or
VERDICT=UNSAFE <one-sentence reason>"

OUT=$(claude -p "$PROMPT" --allowedTools "Bash Read Grep Glob" --disallowedTools "Write Edit WebSearch WebFetch" 2>&1)
echo "----- verifier output -----"
echo "$OUT" | tail -25
echo "---------------------------"
# Emit a machine-readable gate the workflow greps on.
if echo "$OUT" | grep -q "VERDICT=SAFE"; then
  echo "GUARD=PASS"
else
  echo "GUARD=FAIL"
fi
