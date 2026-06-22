# Nightly Kaggle Monitor — 2026-06-22

All four leaderboards rank **higher = better** (top row = best, confirmed per front). Pareto runs on the single `public` axis, so the highest-scoring version dominates → tool says **HOLD** on all four (each best is the sole non-dominated member). No challenger dominates the active best.

---

## playground-series-s6e6
- **Our best REAL score:** 0.97183 (`cand_A_anchor`, raw public best-single, 06-16).
- **Ladder vs top:** top = yuki #2 0.97284; #2 junhaofu1 0.97250. Gap to #1 ≈ **−0.00101**. We sit just under the 0.972 cluster.
- **Deadline:** 2026-06-30 23:59 (per prior run; s6e6 absent from `list -s playground` top rows tonight, not re-read) → ~**8 days left** (not flagged).
- **Submissions:** enabled (last COMPLETE 06-16).
- **Pareto:** front = `cand_A_anchor` 0.97183 (1/5). Dominated/over-engineered: softvote 0.97165, cand_C_override_wide 0.97159, consensus 0.97148, TabPFN3_standalone 0.97061. Every blend scored below the raw anchor. → **HOLD.**
- **Next action:** Only a single candidate beating 0.97183 outright moves us; stop blending into the anchor. RECON for a fresher/stronger public base.

## orbit-wars  ⚠ DEADLINE <7 DAYS
- **Our best REAL score:** 667.7 (`v8` Tempo finals re-measure #2, 06-21). v8 also measured 651.6 on 06-19 — scores are TrueSkill re-measures with opponent-field variance.
- **Ladder vs top:** top = Isaiah @ Tufa Labs 1703.5; #2 Jake Will 1696.7. Gap to #1 ≈ **−1036**. Deep in the field.
- **Deadline:** 2026-06-23 23:59 → **1 day left — FLAG.**
- **Submissions:** enabled (last COMPLETE 06-22).
- **Pareto:** front = `v8` 667.7 (1/3). Dominated: v11 655.4, v7b 614.0. Re-measured old variants keep landing below v8. → **HOLD.**
- **Next action:** With ~1 day left, lock the two highest re-measured agents (`v8` 667.7 + `v11` 655.4) as the active finals pair NOW and stop the re-measure churn. The ~2.5× gap to the leader is algorithmic depth, not closeable in 1 day.

## neurogolf-2026
- **Our best REAL score:** 7128.81 (`mark-b`, franksunp audited single base = local 7114.56 + ARM-only, 06-22; +14 over kojimar 7114.66).
- **Ladder vs top:** top = Fritz & Tony 7884.76; #2 neurogolf team 7876.91 (not us). Gap to #1 ≈ **−756**. Mid-field.
- **Deadline:** 2026-07-15 23:59 → ~**23 days left** (not flagged).
- **Submissions:** enabled (last COMPLETE 06-22).
- **Pareto:** front = `mark-b` 7128.81 (1/5). Dominated: kojimar_base 7114.66, our_6508 6508.55, kojimar_6507 6507.21, v7_overlay 6239.36. Single-base public jumps dwarf all cross-merge builds (~6240). → **HOLD.**
- **Next action:** Lever is the public SOTA base, not overlays. RECON for a fresher single public base above 7128; keep banked mark-b safe.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL score:** 51.750 (`N=575` cliff-edge probe, 06-21). Latest two probes (N=720, N=800) returned **BLANK** = pending/failed (timeout wall above N≈580 per descriptions).
- **Ladder vs top:** top = Victor Merckle 100.490; #2 Team name placeholder 95.310. Gap to #1 ≈ **−48.7**. Far back. Linear N-curve slope ~0.094/N is too shallow to close the gap.
- **Deadline:** 2026-09-01 23:59 → ~**71 days left** (not flagged).
- **Submissions:** enabled, but high-N runs (N≥580) keep returning blank.
- **Pareto:** front = `N=575` 51.750 (1/3). Dominated: N570 51.300, N550 49.500. → **HOLD.**
- **Next action:** N-scaling has hit the timeout wall AND is too shallow to reach the top cluster; pivot from N-probes to recipe/severity quality. Plenty of runway (71 days).

---
_No Battle-proven playbook addition this run: anchor>blend, orbit re-measure over-engineering, neurogolf single-base-jump dominance, and ai-sec N-ramp/timeout-wall are already recorded; the orbit v8 re-measure spread (651.6→667.7) stays confounded by the shifting opponent field, so direction is not cleanly attributable — withheld._

MONITOR_DONE
