# Nightly Kaggle Monitor — 2026-06-21

All four leaderboards are ranked **higher = better** (top row = best score, confirmed per front). Pareto run on the single `public` axis, so the highest-scoring version always dominates → tool says **HOLD** on all four (each best is the sole non-dominated member). No challenger dominates the active best.

---

## playground-series-s6e6
- **Our best REAL score:** 0.97183 (`candA-anchor`, raw public best-single).
- **Ladder vs top:** top = yuki #2 0.97283; #2 junhaofu1 0.97250. Gap to #1 ≈ **−0.00100**. We sit just under the 0.972 cluster.
- **Deadline:** 2026-06-30 23:59 → **9 days left** (not flagged).
- **Submissions:** enabled (userHasEntered=True; last COMPLETE 06-16).
- **Pareto:** front = `candA-anchor` 0.97183 (1/5). Dominated/over-engineered: softvote 0.97165, candC-override 0.97159, consensus 0.97148, TabPFN3-standalone 0.97061. Every blend scored below the raw anchor. → **HOLD.**
- **Next action:** Stop blending into the anchor; only a single candidate that beats 0.97183 outright moves us. RECON for a fresher/stronger public base.

## orbit-wars  ⚠ DEADLINE <7 DAYS
- **Our best REAL score:** 671.3 (`v8-tempo`, finals re-measure #2). v8 also measured 651.6 on 06-19 (opponent field shifted between runs).
- **Ladder vs top:** top = Isaiah @ Tufa Labs 1733.6; #2 Jake Will 1679.4. Gap to #1 ≈ **−1062**. Deep in the field.
- **Deadline:** 2026-06-23 23:59 → **2 days left — FLAG.**
- **Submissions:** enabled (userHasEntered=True; last COMPLETE 06-21).
- **Pareto:** front = `v8-tempo` 671.3 (1/4). Dominated: v11-horizon 655.4, v7b-oracle+ 637.5, v6-oracle 596.3. → **HOLD.** Stop re-measuring old variants — they keep landing below v8.
- **Next action:** With 2 days, freeze `v8-tempo` (671.3) as the final-ladder entry now and stop the re-measure churn. The ~2.6× gap to the leader is an algorithmic-depth gap, not closeable in 2 days.

## neurogolf-2026
- **Our best REAL score:** 7114.66 (`kojimar-7114-base`, audited single public base).
- **Ladder vs top:** top = Fritz & Tony 7876.00; #2 neurogolf team 7861.42 (not us). Gap to #1 ≈ **−761**. Mid-field.
- **Deadline:** 2026-07-15 23:59 → **24 days left** (not flagged).
- **Submissions:** enabled (userHasEntered=True; last COMPLETE 06-20).
- **Pareto:** front = `kojimar-7114-base` 7114.66 (1/5). Dominated: 6508-rebuild 6508.55, 6507-base 6507.21, v6-mega 6241.54, v7-overlay 6239.36. The verbatim public-base jump (+606 over 6508) dwarfs all our cross-merge builds (~6240). → **HOLD.**
- **Next action:** The lever is the public SOTA base, not overlays. RECON for a fresher single public base above 7114; keep stacking honest grader-verified rebuilds on top.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL score:** 51.300 (`v15-N570`). Latest submission (guide24-NOREPORT N=800) is **PENDING** — no score yet.
- **Ladder vs top:** top = Team name placeholder 95.310; #2 Kohei 93.760. Gap to #1 ≈ **−44**. Public score rises monotonically with N (500→45.0, 530→47.7, 550→49.5, 570→51.3); no cliff observed up to 570.
- **Deadline:** 2026-09-01 23:59 → **72 days left** (not flagged).
- **Submissions:** enabled (userHasEntered=True; latest accepted, pending scoring).
- **Pareto:** front = `v15-N570` 51.3 (1/4). Dominated: v15-N550 49.5, v15-N530 47.7, v14-N500 45.0. → **HOLD.**
- **Next action:** Wait for the N=800 score to land; if it beats 51.3 it becomes the new front, else hold v15-N570. N-tuning alone won't reach 95 — pursue recipe/severity quality. Plenty of runway (72 days).

---
_No Battle-proven playbook addition this run: anchor>blend, orbit over-engineering, neurogolf verbatim-base jump, and ai-sec N-ramp are already recorded; the orbit v8 re-measure spread (651.6→671.3) is confounded by the shifting opponent field, so direction is not cleanly attributable — withheld._
