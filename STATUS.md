# Nightly Monitor Digest — 2026-06-27

All four fronts: Pareto says **HOLD** the current best (each best is the sole non-dominated front member; every challenger is strictly dominated). Metric direction confirmed per front from leaderboard ordering (top row = best, all higher-is-better).

## playground-series-s6e6
- **Our best REAL:** 0.97183 (`cand_A_anchor`, raw public best-single).
- **Ladder:** top = yuki #2 0.97284; #2 Hamachi 0.97267; #3 0.97259. Gap to top ≈ **0.00101**.
- **Deadline:** 2026-06-30 23:59 → **3 days left** ⚠️ (under 7).
- **Submissions:** enabled (recent submits COMPLETE with real scores).
- **Pareto:** front = {`cand_A_anchor` 0.97183}. Dominated/over-engineered: softvote 0.97165, cand_C_override 0.97159, consensus 0.97148, hillclimb 0.97073, TabPFN3-standalone 0.97061. Verdict: **HOLD** `cand_A_anchor`.
- **Next action:** With 3 days left, RECON the *current* public best-single again (`kaggle kernels list --competition playground-series-s6e6 --sort-by dateRun`); only a fresher/stronger raw anchor moves us — blending the same pool has repeatedly cost score.

## orbit-wars
- **Our best REAL:** 658.8 (`v11_horizon`, latest measure).
- **Ladder:** top = Isaiah @ Tufa Labs 1611.3; #2 Hober Malloc 1534.4; #3 1514.7. Gap to top ≈ **952.5** (far behind).
- **Deadline:** 2026-07-07 23:59 → **10 days left**.
- **Submissions:** enabled (recent submits COMPLETE). Note TrueSkill ±150 variance across re-measures (v8 read 625.3/636.2/651.6 on different fields).
- **Pareto:** front = {`v11_horizon` 658.8}. Dominated: v8 625.3, v7b 614.0, v6 596.3 (this window). Verdict: **HOLD** `v11_horizon`.
- **Next action:** Hold the two-active double-insurance (v8 floor + v11). We are ~950 below the top — heuristic/horizon tweaks won't close that; only a stronger search/planner would. Do not burn final slots re-measuring noise.

## neurogolf-2026
- **Our best REAL:** 7129.07 (`franksunp7128_plus_rebuilds`, banked).
- **Ladder:** top = Matheus & Fritz & Tony 7959.75; #2 neurogolf team 7929.44; #3 7910.90. Gap to top ≈ **830.7**.
- **Deadline:** 2026-07-15 23:59 → **18 days left**.
- **Submissions:** enabled (recent submits COMPLETE).
- **Pareto:** front = {`franksunp7128_plus_rebuilds` 7129.07}. Dominated: franksunp_markb 7128.81, **crossdump_harvest 7119.59** (latest — local grader claimed 7150.41 but LB regressed below banked), kojimar_7114 7114.66, 6508/6507, v7_overlay 6239.36. Verdict: **HOLD** 7129.07; do NOT replace with the crossdump harvest.
- **Next action:** Stop banking local-grader-only gains; reconcile the local↔LB metric before any merge. We are 830 behind the top — pull the current top public base (sort by dateRun/voteCount) and reproduce it verbatim before adapting.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL:** 51.750 (`N575`).
- **Ladder:** top = Victor Merckle 100.490; #2 Team name placeholder 95.310; #3 Kohei 93.760. Gap to top ≈ **48.7** (far behind).
- **Deadline:** 2026-09-01 23:59 → **66 days left**.
- **Submissions:** enabled, BUT N≥720 / N=800 attempts returned BLANK (timeout wall); only N≤575 score.
- **Pareto:** front = {`N575` 51.75}. Dominated: N570 51.3, N550 49.5, N530 47.7, N500 45.0 (clean monotonic N-curve up to the cliff). Verdict: **HOLD** `N575`.
- **Next action:** Don't keep pushing N into the timeout wall (≥720 = blank). The top is ~2× us — that is a recipe/severity-per-finding gap, not a volume gap. RECON the current top public guide/recipe and lift its per-finding quality rather than scaling N.
