# Kaggle Monitor Digest — 2026-06-24

Metric direction confirmed per front from leaderboard ordering (top row = best); all four are higher-is-better.

## playground-series-s6e6
- **Our best REAL score:** 0.97183 (`cand_A_anchor`, raw public best-single, 06-16).
- **Ladder position:** top = yuki #2 0.97284; next Hamachi/nybbler 0.97252, vedant pol 0.97251. Gap from our best ≈ **0.00101**; we sit below the 0.9725 leader cluster.
- **Deadline:** 2026-06-30 23:59 → **6 days left** (⚠ under 7 — flagged).
- **Submissions enabled:** yes (all recent statuses COMPLETE).
- **Pareto verdict:** 5 scored versions tonight. Front = {`cand_A_anchor` 0.97183} only (sole non-dominated). Dominated/over-engineered: softvote 0.97165, override_wide 0.97159, consensus 0.97148, TabPFN3_standalone 0.97061 — every blend strictly worse than the raw anchor. Tool says **HOLD** anchor.
- **Next action:** Stop blending into the anchor — all 4 blends are dominated. With 6 days left, RECON current public SOTA kernels (sort by votes/recent) for a stronger BASE than 0.97183; the leaders already cleared 0.9725.

## orbit-wars
- **Our best REAL score:** `v11` re-measure **658.3** (06-22), now the highest live read; `v8` re-measured to 640.5 on 06-23. (TrueSkill ±~150-noisy across re-measures — single reads unreliable.)
- **Ladder position:** top = Isaiah @ Tufa Labs 1693.8; Jake Will 1616.8, TonyK 1602.7. We are ~658 → **massive gap (~1035)**, far down the ladder.
- **Deadline:** 2026-07-07 23:59 → **13 days left** (NOTE: deadline is 07-07, not 06-23 as the prior digest assumed — the window is NOT closing tonight).
- **Submissions enabled:** yes (recent COMPLETE).
- **Pareto verdict:** 4 distinct versions (latest score each). Front = {`v11` 658.3} only. Dominated: v8 640.5, v7b 614.0, v6 596.3. Tool says **REPLACE** current best `v8` → `v11` Pareto-dominates it on tonight's measure.
- **Next action:** Per the noise caveat (±150), do NOT hard-swap on one read — re-measure `v8` and `v11` head-to-head on the current field 2–3× before flipping the active agent. With 13 days now available there is time to confirm v11>v8 robustly rather than trust a single 658.3 vs 640.5.

## neurogolf-2026
- **Our best REAL score:** 7128.81 (`mark-b_7128`, franksunp audited single base + ARM-only, 06-22).
- **Ladder position:** top = Fritz & Tony 7899.04; neurogolf team 7876.91, Pavel 7850.31 also ahead. Gap from our best ≈ **770**.
- **Deadline:** 2026-07-15 23:59 → **21 days left**.
- **Submissions enabled:** yes (recent COMPLETE).
- **Pareto verdict:** 7 versions. Front = {`mark-b_7128` 7128.81} only. Dominated: kojimar_7114 7114.66, hand-rebuild_6508 6508.55, kojimar_6507 6507.21, v5 6287.25, v6_MEGA 6241.54, v7_overlay 6239.36 (cross-base merges sit at the bottom). Tool says **HOLD** mark-b.
- **Next action:** Single audited base keeps climbing (6507→7114→7128); every cross-merge is dominated. RECON for the next stronger public single base toward the ~7899 top — keep the verbatim + ARM-only safety wrapper, don't merge bases.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL score:** 51.750 (`N=575` cliff-edge probe, 06-21).
- **Ladder position:** top = Victor Merckle 100.490; Team name placeholder 95.310, Kohei 93.760. Gap from our best ≈ **48.74** (under half the leader's score).
- **Deadline:** 2026-09-01 23:59 → **69 days left**.
- **Submissions enabled:** yes; 2 recent blank probes (N=720, N=800 — timed out at high N).
- **Pareto verdict:** 5 scored N-points. Front = {`N=575` 51.75} only. Dominated: N=570 51.30, N=550 49.50, N=530 47.70, N=500 45.00 — monotone N-curve up to the timeout wall (~580). Tool says **HOLD** N=575.
- **Next action:** N=720/800 returned BLANK (timeout wall above ~580); raw-N scaling is exhausted. The ~49pt gap to 100 needs a better *recipe* (more severity/cells per finding), not more candidates — study what leaders' attacks do differently before re-scaling N.
