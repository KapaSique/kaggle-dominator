#!/usr/bin/env bash
# kaggle_monitor.sh — headless multi-front Kaggle monitor (NO submit).
#
# For each front slug it reads the REAL Kaggle state (submission scores, ladder rank,
# deadline) via the API, writes a dated digest to STATUS.md, and appends any MEASURED
# transferable insight to the playbook. It never submits. Driven by the nightly workflow.
#
# Usage: ./kaggle_monitor.sh "<slug1> <slug2> ..."
# Requires: claude CLI (ANTHROPIC_API_KEY env), kaggle CLI authenticated.
set -uo pipefail

SLUGS="${1:?usage: kaggle_monitor.sh \"<slug1> <slug2> ...\"}"
echo "Monitoring fronts: $SLUGS"

PROMPT="You are the autonomous nightly Kaggle monitor. Fronts to check (space-separated): ${SLUGS}

For EACH front slug, using the Bash tool with 'python3 -m kaggle':
1. Run: python3 -m kaggle competitions submissions <slug> --csv | head -6   (our latest submissions + REAL publicScore; blank means pending or failed).
2. Run: python3 -m kaggle competitions leaderboard <slug> --show | head -8   (top scores; note the gap from our best).
3. Run: python3 -m kaggle competitions list -s <one keyword of the slug>   (read the deadline; flag if under 7 days away).

Then WRITE a dated digest to STATUS.md (OVERWRITE the whole file). One section per front containing: our best REAL score, our ladder position vs the top, deadline and days left, whether submissions look enabled, and ONE concrete recommended next action. Be terse and factual: only numbers you actually read, never invented ones.

If — and only if — you observe a MEASURED, transferable insight proven by a real number you just read, append ONE bullet to the 'Battle-proven additions' section of references/grandmaster-playbook.md. Before claiming any 'X beat Y', confirm the metric DIRECTION from the leaderboard ordering (top row = best). When unsure, do NOT write it.

HARD RULES: Do NOT run any 'kaggle competitions submit' or 'kaggle kernels push' command. Monitoring and writing only. Stay within about 20 Bash calls total.

End your reply with: MONITOR_DONE"

claude -p "$PROMPT" \
  --allowedTools "Bash Read Write Edit Grep Glob" \
  --disallowedTools "WebSearch WebFetch" \
  || echo "[monitor] claude exited non-zero — committing whatever digest exists"
