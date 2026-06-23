# Kaggle Monitor Digest — 2026-06-23

Metric direction confirmed per front from leaderboard ordering (top row = best); all four are higher-is-better.

## playground-series-s6e6
- **Our best REAL score:** 0.97183 (`cand_A_anchor`, raw public best-single, 06-16).
- **Ladder position:** top = yuki #2 0.97284; gap from our best ≈ **0.00101**. Cluster of leaders at 0.9725.
- **Deadline:** 2026-06-30 23:59 → **7 days left** (borderline — watch).
- **Submissions enabled:** yes (all recent statuses COMPLETE).
- **Pareto verdict:** 6 scored versions. Front = {`cand_A_anchor` 0.97183} only. Dominated/over-engineered: softvote 0.97165, override_wide 0.97159, consensus 0.97148, hillclimb 0.97073, TabPFN3_standalone 0.97061. Tool says **HOLD** anchor (sole front member; every blend is strictly worse).
- **Next action:** Stop blending into the anchor. To close the 0.001 gap, RECON the *current* public SOTA (kernels list by votes/date) for a stronger BASE than the 0.97183 anchor — the leaders already moved to 0.9728.

## orbit-wars
- **Our best REAL score:** latest active `v8` re-measure = **649.8** (06-23); highest single read in window 655.4 (`v11` Horizon, 06-19). TrueSkill is ±150-noisy across re-measures.
- **Ladder position:** top = Isaiah @ Tufa Labs 1829.5; we are ~650 → **massive gap (~1180)**, far down the ladder.
- **Deadline:** 2026-06-23 23:59 → **TODAY, hours left** (URGENT — submission window closes tonight).
- **Submissions enabled:** yes (recent COMPLETE).
- **Pareto verdict:** 4 distinct versions (latest score each). Front = {`v8` 649.8} only. Dominated: v11 625.1, v7b 614.0, v6 596.3. Tool says **HOLD** v8.
- **Next action:** Deadline is tonight — do NOT swap the active agent on noisy single reads. Confirm `v8` is the locked active submission for final scoring and leave it; no time to validate a challenger.

## neurogolf-2026
- **Our best REAL score:** 7128.81 (`mark-b`, franksunp audited single base + ARM-only, 06-22).
- **Ladder position:** top = Fritz & Tony 7893.17; gap from our best ≈ **764**. (neurogolf team 7876.91, Pavel 7850.31 also ahead.)
- **Deadline:** 2026-07-15 23:59 → **22 days left**.
- **Submissions enabled:** yes (recent COMPLETE).
- **Pareto verdict:** 7 versions. Front = {`mark-b_7128` 7128.81} only. Dominated: kojimar_7114, hand-rebuild_6508, kojimar_6507, v5 6287, v6_MEGA 6241, v7_overlay 6239 (the cross-base merges sit at the bottom). Tool says **HOLD** mark-b.
- **Next action:** Single audited base keeps climbing (6507→7114→7128); cross-merges are dominated. RECON for the next stronger public single base toward the 7893 top — keep the verbatim+ARM-only safety wrapper, don't merge bases.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL score:** 51.750 (`N=575` cliff-edge probe, 06-21).
- **Ladder position:** top = Victor Merckle 100.490; gap from our best ≈ **48.7** (roughly half the leader's score).
- **Deadline:** 2026-09-01 23:59 → **70 days left**.
- **Submissions enabled:** yes; note 2 recent pending/blank probes (N=720, N=800 — likely timed out at high N).
- **Pareto verdict:** 5 scored N-points. Front = {`N=575` 51.75} only. Dominated: N=570 51.30, N=550 49.50, N=530 47.70, N=500 45.00 — monotone N-curve, higher N = higher score up to the timeout wall. Tool says **HOLD** N=575.
- **Next action:** N=720/800 returned BLANK (timeout wall above ~580). Don't push raw N higher — the gap to 100 needs a better *recipe* (more severity/cells per finding), not more candidates. Find what the leaders' attack does differently before re-scaling N.
