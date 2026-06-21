#!/usr/bin/env bash
# re_recon.sh — every-few-days re-RECON sweep (playbook rule 7: stale-base trap).
#
# The public frontier moves fast (neurogolf 6507→7800s in 2 days); reproducing a fresh
# best-public base dwarfs hand-grinding (+606 measured vs +1.35). This scans each front for a
# PUBLIC solution that beats our current best, and flags it in STATUS_RERECON.md. NO submit.
#
# Cloud-friendly: reads OUR best from the Kaggle API (submissions), not local memory.
# Usage: ./re_recon.sh "<slug1> <slug2> ..."
set -uo pipefail

SLUGS="${1:?usage: re_recon.sh \"<slug1> <slug2> ...\"}"
echo "Re-RECON fronts: $SLUGS"

PROMPT="You are the periodic Kaggle re-RECON scout. Apply playbook rule 7 (stale-base trap): reproducing a fresh best-public base dwarfs hand-grinding (neurogolf gave +606). Fronts (space-separated): ${SLUGS}

For EACH front slug, using Bash with 'python3 -m kaggle':
1. OUR best: 'python3 -m kaggle competitions submissions <slug> --csv | head -4' → read our top REAL score.
2. Leaderboard: 'python3 -m kaggle competitions leaderboard <slug> --show | head -12' → top public scores.
3. Public solutions: 'python3 -m kaggle kernels list --competition <slug> --sort-by voteCount --page-size 12' and 'python3 -m kaggle datasets list -s <one keyword of the slug> --sort-by hotness' → look for a NEW public notebook/dataset whose stated score beats OUR best.

Then WRITE a dated digest to STATUS_RERECON.md (overwrite). One line per front: our best, top public found, and VERDICT — 'REPRODUCE: <name> scores <x> > our <y>, steps: ...' if a reproducible public base beats us, else 'hold — none newer'. Only REAL scores you actually read; never invent. If a front gained reproduce-upside, make that line start with '🔴'.

HARD RULES: do NOT submit or push kernels. Read + write only. Stay within ~25 Bash calls.

End with: RERECON_DONE (and list any 🔴 fronts)."

claude -p "$PROMPT" \
  --allowedTools "Bash Read Write Edit Grep Glob" \
  --disallowedTools "WebSearch WebFetch" \
  || echo "[re-recon] claude exited non-zero"
