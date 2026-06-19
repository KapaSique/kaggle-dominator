# Kaggle Monitor — nightly digest (2026-06-19)

Metric direction confirmed per front from leaderboard ordering (top row = best ⇒ higher = better on all four). Pareto rebuilt tonight from fresh scored submissions only (pending/blank skipped).

## playground-series-s6e6
- **Our best REAL:** 0.97183 (`cand_A_anchor` — public best-single, submitted verbatim).
- **Ladder:** top = 0.97283 (yuki #2); top-5 down to 0.97233. Our 0.97183 sits ~0.00050 below 5th, ~0.00100 below #1.
- **Deadline:** s6e6 NOT shown in `competitions list` output (only s6e1–s6e4 listed). Playground monthly cadence implies ~2026-06-30 (~11 days) — NOT directly read, treat as unconfirmed.
- **Submissions enabled:** yes — latest COMPLETE 2026-06-16.
- **PARETO (10 scored versions):** front = {anchor 0.97183} alone; all 9 blends/stackers DOMINATED (softvote 0.97165, override_wide 0.97159, consensus 0.97148, cdeotte_LR 0.97101, mega_stack 0.97085, hillclimb 0.97073, TabPFN3_standalone 0.97061, enriched 0.96941, 2level 0.96934). Tool: **HOLD** anchor (sole front member). Blending decorrelated-but-weaker signals into it net-hurt.
- **Next action:** stop blending the existing public pool; passing the 0.972 cluster needs a richer BASE (GPU rebuild / fresh public-SOTA recon), not more votes over the same anchor.

## orbit-wars
- **Our best REAL:** 737.9 (`v6_Oracle`) — but **non-reproducible**: re-submitting identical v6 scored 585.9 (−152). Active/defended version is `v8_Tempo` 684.1 (also re-scored 655.8 on resubmit). Ladder is high-variance.
- **Ladder:** top = 1720.0 (Isaiah @ Tufa Labs); top-5 1599–1720. Our best ~737.9 is ~982 below #1 — far down.
- **Deadline:** 2026-06-23 23:59 → **4 DAYS LEFT — FLAG (<7).**
- **Submissions enabled:** yes — latest COMPLETE 2026-06-19.
- **PARETO (7 versions):** front = {v6_Oracle 737.9}; v8 684.1, v7b 663.6, v11 643.6, v9 636.4, v5 573.0, v1 542.7 DOMINATED. Tool: **REPLACE v8_Tempo → v6_Oracle.** CAVEAT: this is the single-submission-fluke trap — v6 only reproduced 585.9, so 737.9 is NOT a real ceiling; do not chase it on faith.
- **Next action:** with 4 days left and a ~1000-pt gap to the top, don't burn remaining slots reverting to the noisy 737.9. Re-submit v8_Tempo (and/or v6) 2–3× to estimate the true mean, then lock the higher-mean version for final scoring.

## neurogolf-2026
- **Our best REAL:** 6507.21 (`kojimar_audited` base — grader-verified local 6507.21 exact).
- **Ladder:** top = 7843.05 (neurogolf team); top-5 7687–7843. Our 6507.21 is ~1336 below #1.
- **Deadline:** 2026-07-15 23:59 → ~26 days.
- **Submissions enabled:** yes — latest COMPLETE 2026-06-19.
- **PARETO (8 versions):** front = {kojimar_audited 6507.21}; v5 6287.25, v4 6281.20, v2 6275.70, baseline_kojimar 6275.09, v6_MEGA 6241.54, v7_overlay 6239.36, v3 6102.00 DOMINATED — the heavier multisource overlays (v6/v7) scored BELOW the simpler audited base. Tool: **HOLD** kojimar_audited.
- **Next action:** protect the 6507.21 base; gap to top is large, so pursue a stronger public base / new arch rather than more conv-fit overlays (which dominate-down).

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL:** 45.000 (`v14` guide24 N=500). Several N=530/550/570 (v15) probes still PENDING.
- **Ladder:** top = 90.000 (Kohei), 89.910, 82.300, 77.65, 66.645. Our 45.0 is exactly half the top, below 5th (66.645).
- **Deadline:** 2026-09-01 23:59 → ~74 days.
- **Submissions enabled:** yes — submissions accepted 2026-06-19 (pending).
- **PARETO (3 scored versions):** front = {v14_N500 45.0}; v14_N420 37.8, v11_N350 30.53 DOMINATED — score rises with N up to the cliff. Tool: **HOLD** v14_N500.
- **Next action:** wait on the pending v15 N=530/550/570 results to map the cliff above N=500; if they don't beat 45.0, the lever is recipe quality (top is at 90), not more N.

MONITOR_DONE
