# Nightly Monitor Digest — 2026-06-28

Scores read live tonight via `kaggle`. Higher = better on every front. Pareto fronts rebuilt from tonight's scored submissions (single `public` axis each).

## playground-series-s6e6
- **Our best REAL score:** 0.97244 (ridge-flip-refinement, danushkumarv, 69 votes; submitted 06-27, COMPLETE).
- **Ladder:** below the visible top-5. Top = yuki #2 0.97284; 5th visible = Bang sion 0.97256. **Gap to top ≈ 0.00040**, gap to 5th ≈ 0.00012.
- **Deadline:** 2026-06-30 23:59 — **2 DAYS LEFT ⚠️ (<7, near-closing).**
- **Submissions enabled:** yes — latest 06-27 COMPLETE.
- **Pareto:** front = {ridge-flip-refinement-danushkumarv} (sole non-dominated). DOMINATED/over-engineered: raunakdey07-v3 (0.97243), anchor+152-flips (0.97159), TabPFN-3-standalone (0.97061), RECON-diversify-odvut1 (0.96522). Tool says **HOLD**.
- **Next action:** With 2 days left and 0.97244 on the front, do NOT chase the 0.0004 gap with risky flips. Confirm 0.97244 is one of the two FINAL selected subs before 06-30 close.

## orbit-wars
- **Our best REAL score:** 655.4 (v11-Horizon re-measure, 06-19) / 654.6 (v11 finals, 06-22). Best ≈ 655.4.
- **Ladder:** far below the visible top-5. Top = Isaiah @ Tufa Labs 1606.7; 5th = Felix M Neumann 1522.3. **Gap to top ≈ 951** (we're ~41% of leader's rating).
- **Deadline:** 2026-07-07 23:59 — **9 days left.**
- **Submissions enabled:** yes — last COMPLETE 06-23 (none since).
- **Pareto:** front = {v11-Horizon} (sole). DOMINATED: v8-Tempo (618.3), v7b-Oracle+ (614.0). Tool says **HOLD** v11.
- **Next action:** Score gap to the field is enormous (655 vs 1606); incremental v8/v7b variance-chasing is futile. Re-scout top public approaches for a structurally different agent before 07-07; otherwise hold v11+v8 as the locked active pair.

## neurogolf-2026
- **Our best REAL score:** 7129.07 (7128-franksunp base + 6 true-rule rebuilds, grader-verified 400/400; 06-24).
- **Ladder:** below the visible top-5. Top = Matheus & Fritz & Tony 7967.19; 5th = Carlos/Egor/Rubempre/Bill 7889.12. **Gap to top ≈ 838.**
- **Deadline:** 2026-07-15 23:59 — **17 days left.**
- **Submissions enabled:** yes — last COMPLETE 06-26 (7119.59, regressed vs 7129.07, correctly NOT adopted).
- **Pareto:** front = {7128-franksunp+6-true-rule-rebuilds} (sole). DOMINATED: franksunp-7128.81 (7128.81), cross-dump-harvest+44 (7119.59), kojimar-7114.66 (7114.66), 6508-base (6508.55). Tool says **HOLD**.
- **Next action:** The 06-26 cross-dump +44-graph harvest LOST ~10 pts (7119.59 < 7129.07) — over-merging hurt. Hold 7129.07; pursue a fresher/stronger public base (top is 7967) rather than adding more cheap graphs to the current base.

## ai-agent-security-multi-step-tool-attacks
- **Our best REAL score:** 51.750 (N=575 cliff-edge probe; 06-21).
- **Ladder:** below the visible top-5. Top = Victor Merckle 100.490; 5th = shimacos 89.550. **Gap to top ≈ 48.7** (we're ~52% of leader).
- **Deadline:** 2026-09-01 23:59 — **65 days left.**
- **Submissions enabled:** yes for completed N — but N=720 and N=800 returned BLANK (timeout/over-budget wall); N≤575 score cleanly.
- **Pareto:** front = {N575-cliff-edge-probe} (sole). DOMINATED: N570-v15 (51.300), N550-v15 (49.500). Tool says **HOLD**.
- **Next action:** N-curve confirmed near-linear (N=570→51.30, N=575→51.75 ≈ +0.09/N) but N≥720 hits a timeout wall (blank). To close the ~49-pt gap, fix the per-candidate runtime (suppress report/output) so high-N runs complete, rather than nudging N near the current cliff. Plenty of runway (65 days).

---
Pareto verdict across all four fronts: **HOLD** — each front's banked best is the sole non-dominated member; every challenger is strictly dominated (over-engineered). No REPLACE warranted tonight.
