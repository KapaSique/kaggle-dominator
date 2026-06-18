# Nightly Kaggle Monitor — 2026-06-18

Metric direction confirmed per front from leaderboard ordering (top row = best). All four fronts are **higher = better**.

## playground-series-s6e6
- **Our best REAL public score:** 0.97183 (cand_A_anchor, 2026-06-16; raw public best-single).
- **Ladder position:** Top = 0.97283 (yuki #2). Top-5 floor shown = 0.97233. Our 0.97183 sits **below the top-5**, gap **0.00100** to #1, ~0.00050 to 5th. Field = 1923 teams.
- **Deadline:** 2026-06-30 23:59 → **12 days left.**
- **Submissions enabled:** Yes — recent submits all COMPLETE with scores.
- **Next action:** Blending the anchor has repeatedly net-hurt (softvote 0.97165, coalition 0.97159, TabPFN3 standalone 0.97061 all < 0.97183). Stop blending the same public pool; re-RECON for a *newer/stronger* public anchor or build a richer GPU base — only a stronger standalone signal will clear 0.972.

## orbit-wars
- **Our best REAL score:** 684.1 (v8 Tempo, 2026-06-10). Latest submit (v6 resubmit, 2026-06-18) scored only **610.7** — well below the 737.9 the v6 lever historically earned (live-ladder variance / opponent drift).
- **Ladder position:** Top = 1767.9 (Isaiah @ Tufa Labs); 5th = 1643.6. We are **far back** (~684 vs 1768, ≈39% of top). Field = 4665 teams.
- **Deadline:** 2026-06-23 23:59 → **5 days left — ⚠️ UNDER 7 DAYS.**
- **Submissions enabled:** Yes — COMPLETE statuses.
- **Next action:** Time-critical. Re-submit the proven peak v6/v8 line to lock a clean ladder score (the 610.7 resubmit underperformed); confirm which agent version is actually active before the window closes. The 1768 top implies a structurally deeper planner — no time to rebuild, so protect the best proven agent.

## neurogolf-2026
- **Our best REAL score:** 6287.25 (v5, 2026-06-14). v6/v7 mega-multisource (6241.54 / 6239.36) both scored *below* v5.
- **Ladder position:** Top = 7800.75 (neurogolf team); 5th = 7676.92. We trail the top by **~1513** (6287 vs 7801). Field = 2064 teams.
- **Deadline:** 2026-07-15 23:59 → **27 days left.**
- **Submissions enabled:** Yes — COMPLETE statuses.
- **Next action:** v5 (holdout-validated conv fits) is the protected peak; the multisource layer overfit. Pursue a richer base toward the 7700+ cluster rather than more conv-fit stacking on the v2 base.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL score:** 30.530 (guide24 N=350, 2026-06-17). Prior completed: 18.11/18.06 (N≈900 mass), 14.54.
- **Ladder position:** Top = 77.670 (Kohei); 2nd = 77.650; then a cliff to 66.645 / 58.500. Our 30.530 is **below the top cluster**, roughly mid-field. Field = 772 teams.
- **Submissions enabled:** Yes, but the four newest v14 cliff-probes (N=420/500/580 + safe-point) are **PENDING** (no score yet) — re-check next run.
- **Deadline:** 2026-09-01 23:59 → **75 days left** (ample).
- **Next action:** Wait for the PENDING v14 N-sweep to score, then keep the best N. Recipe quality (guide24 at N=350 = 30.53) already beat crude mass (N=900 = 18.11); next lever is matching the top's recipe, not raising N further.

---
_Monitoring only — no submissions made. Scores read directly from `kaggle competitions submissions/leaderboard/list`._
