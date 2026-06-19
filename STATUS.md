# Kaggle Monitor — nightly digest 2026-06-19

All metrics higher = better (confirmed: leaderboard top row = best score). Pareto rebuilt tonight from fresh scored submissions only (pending/blank skipped).

---

## playground-series-s6e6
- **Our best REAL score:** 0.97183 (`cand_A_anchor`, raw public best-single, COMPLETE).
- **Ladder vs top:** top = 0.97283 (yuki #2). Gap = **0.00100**. We sit below the visible top-5 (5th shown 0.97233) — outside top 5.
- **Deadline:** 2026-06-30 — **11 days left**.
- **Submissions enabled:** yes (recent subs all COMPLETE).
- **Pareto:** 1/5 on front. Front = `anchor` (0.97183). DOMINATED/over-engineered: softvote 0.97165, cand_C_override 0.97159, consensus 0.97148, TabPFN3_standalone 0.97061. Tool says **HOLD anchor** (sole front member; every blend OF the anchor scored below it).
- **Next action:** Don't submit any blend that doesn't beat 0.97183 on CV first. To climb the 0.001 gap you need a richer BASE (fresh public SOTA / GPU rebuild), not more blending of the existing pool.

## orbit-wars
- **Our best REAL score:** 684.1 (`v8_Tempo`, COMPLETE).
- **Ladder vs top:** top = 1723.9 (Jake Will). Gap ≈ **1040**; 5th visible 1601.5 — far below the top cluster.
- **Deadline:** 2026-06-23 — **4 days left ⚠️ UNDER 7 DAYS**.
- **Submissions enabled:** yes (latest v6 re-submit COMPLETE 2026-06-18).
- **Pareto:** 1/5 on front. Front = `v8_Tempo` (684.1). DOMINATED: v7b 663.6, v11_Horizon 643.6, v9_Diplomat 636.4, v6_Oracle_resubmit 596.1. Tool says **HOLD v8_Tempo**. Note: tonight's v6 re-submit (claimed "proven peak 737.9 local") scored only **596.1** on the ladder — local mirror score did not transfer.
- **Next action:** 4 days out — lock/confirm v8 (the real measured peak); stop over-engineering. If burning a slot, re-run v8 to verify placement given ladder score noise, not a new variant.

## neurogolf-2026
- **Our best REAL score:** 6507.21 (`kojimar_6507_base`, fresh public RECON base, COMPLETE 2026-06-19).
- **Ladder vs top:** top = 7843.05 (neurogolf team). Gap ≈ **1336**; 5th visible 7687.71.
- **Deadline:** 2026-07-15 — **26 days left**.
- **Submissions enabled:** yes.
- **Pareto:** 1/5 on front. Front = `kojimar_6507_base` (6507.21). DOMINATED: v5 6287.25, v4 6281.20, v6_MEGA 6241.54, v7_overlay 6239.36 — our own MEGA/overlay engineering scored BELOW even simple v5. Tool says **HOLD kojimar base** (+220 over prior best v5).
- **Next action:** Build holdout-validated conv-fit overlays ON TOP of the new 6507 base (not the stale 6372 base). The over-engineered overlays that lost ~220 were built on a weaker base.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL score:** 45.000 (`v14_guide24_N=500`, COMPLETE). v15 N=530/550/570 are PENDING (no score yet).
- **Ladder vs top:** top = 90.000 (Kohei). Gap = **45.0** (we're at exactly half); 5th visible 61.81.
- **Deadline:** 2026-09-01 — **74 days left**.
- **Submissions enabled:** yes (subs flowing; several PENDING tonight).
- **Pareto:** 3 scored versions → 1/3 on front. Front = `v14_N=500` (45.000). DOMINATED: v14_N=420 37.800, v11safe_N=350 30.530. Tool says **HOLD v14_N=500**. The live lever is N: 350→30.53, 420→37.8, 500→45.0, 580→blank (timeout) — score climbs with N up to a cliff.
- **Next action:** Wait for tonight's PENDING v15 N=530/550/570 to score, then pick the highest N that still scores (below the ~580 timeout cliff) as the new best — don't push N past where it blanks.
