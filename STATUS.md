# Nightly Kaggle Monitor — 2026-06-20

All four fronts have recent COMPLETE submissions → **submissions enabled everywhere**.
Metric direction confirmed per front from leaderboard ordering (top row = best ⇒ higher = better on all four).
All Pareto fronts are single-axis (public only); each best is the sole non-dominated
member → tool says **HOLD** on all four. No challenger dominates the active best.

---

## playground-series-s6e6
- **Our best REAL:** 0.97183 (`cand_A_anchor` = raw public best-single).
- **Ladder vs top:** top = yuki 0.97283; gap **−0.00100**. We sit in the 0.971 band just under the 0.972 cluster.
- **Deadline:** 2026-06-30 23:59 → **10 days left**.
- **Submissions:** enabled (last COMPLETE 06-16).
- **Pareto:** front = `anchor_candA` (1/6). Dominated/over-engineered: softvote 0.97165, override_wide 0.97159, consensus 0.97148, hillclimb 0.97073, TabPFN3_standalone 0.97061. Every blend OF the anchor scored below it. → **HOLD anchor_candA.**
- **Next action:** Don't blend further into the anchor. Re-RECON current public Code for a *fresher/stronger BASE* (the 0.972 cluster anchor may have advanced); a richer base, not more voting, is the only lever above the cluster.

## orbit-wars  ⚠ DEADLINE <7 DAYS
- **Our best REAL (current 1720-field):** 651.6 (`v8`). Historical peak v6=737.9 re-measured to only 596.3 on the current field (local-mirror inflation), so trust the 651.6 floor.
- **Ladder vs top:** top = Jake Will 1696.1; we are ~651 → gap **≈ −1044**. Deep in the field.
- **Deadline:** 2026-06-23 23:59 → **3 days left — FLAG.**
- **Submissions:** enabled (last COMPLETE 06-20).
- **Pareto:** front = `v8` 651.6 (1/4). Dominated: v7b 645.0, v11 639.9, v6 596.3 (all current-field re-measures). → **HOLD v8.** Stop re-measuring old variants on the new field — they keep landing below v8.
- **Next action:** With 3 days, freeze v8 as the active final-ladder entry and stop the re-measure churn. Only replace if a challenger Pareto-dominates v8 on the current field — none does. Top is ~2.6× our score; the gap is an algorithmic-depth gap, not a tuning gap, and won't close in 3 days.

## neurogolf-2026
- **Our best REAL:** 6508.55 (`blend6508` = kojimar 6507 public base + 6 our true-rule hand-rebuilds, +1.35 over raw base).
- **Ladder vs top:** top = Fritz & Tony 7865.68; gap **≈ −1357**. Mid-field.
- **Deadline:** 2026-07-15 23:59 → **25 days left**.
- **Submissions:** enabled (last COMPLETE 06-19).
- **Pareto:** front = `blend6508` 6508.55 (1/7). Dominated: kojimar6507 6507.21, v5 6287.25, v4 6281.20, v2 6275.70, v6_MEGA 6241.54, v7_overlay 6239.36. The verbatim-public-base jump (6287→6507, +220) dwarfs all our own multisource builds. → **HOLD blend6508.**
- **Next action:** The lever is the public SOTA base, not our overlays (which all scored ≤6287). RECON for a fresher public base above 6507; keep stacking honest grader-verified hand-rebuilds on top of it (the +1.35 edge is real but tiny — find a higher base).

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL:** 51.300 (`v15_N570`).
- **Ladder vs top:** top = Kohei 93.160; gap **≈ −41.9**. Public score rises monotonically with N (420→37.8, 500→45.0, 530→47.7, 550→49.5, 570→51.3); N=580 went blank (cliff).
- **Deadline:** 2026-09-01 23:59 → **73 days left**.
- **Submissions:** enabled (last COMPLETE 06-19).
- **Pareto:** front = `v15_N570` 51.3 (1/5). Dominated: v15_N550 49.5, v15_N530 47.7, v14_N500 45.0, v14_N420 37.8. → **HOLD v15_N570.**
- **Next action:** We're climbing the N-ramp toward the ~580 cliff with diminishing returns; 51.3 vs top 93.2 means N-tuning alone won't reach the top. Pursue recipe/severity quality (per playbook: recipe > N volume) and verify the mechanism transfers to the PRIVATE guardrail, not just public — that's where the ~42-pt gap to Kohei likely lives.

---
_No Battle-proven playbook addition this run: every observed signal (anchor>blend, orbit over-engineering past peak, ai-sec recipe>N / over-return cliff, neurogolf verbatim-base jump) is already recorded. Nothing new + direction-confirmed to add._
