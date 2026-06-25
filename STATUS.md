# Nightly Monitor Digest — 2026-06-25

All four fronts: Pareto says **HOLD** the current best (each best is the sole non-dominated front member; all other versions strictly dominated/over-engineered). Submissions appear enabled everywhere (recent COMPLETE rows). Higher = better on every front (LB top row = best).

---

## playground-series-s6e6  (tabular Playground)
- **Our best REAL score:** 0.97183 (`cand_A_anchor` = raw public best-single).
- **Ladder:** top = 0.97284 (yuki #2). Gap to top ≈ **0.00101**. Cluster is dense (top-5: 0.97284 / 0.97258 / 0.97254 / 0.97253 / 0.97251).
- **Deadline:** 2026-06-30 23:59 → **5 days left — UNDER 7, FLAG.**
- **Submissions:** enabled.
- **Pareto verdict:** front = {`cand_A_anchor` 0.97183}. Dominated/over-engineered: softvote 0.97165, coalition-override 0.97159, consensus 0.97148, TabPFN3-standalone 0.97061. Tool says **HOLD** — never replace front-member best with a non-dominating challenger.
- **Next action:** Comp closes in 5 days. Blending the same public pool has plateaued (all blends < anchor). Do one fresh RECON of *current* public Code best-single; submit a richer BASE verbatim only if it standalone-beats 0.97183 — otherwise lock anchor as final.

---

## orbit-wars  (RTS simulation, $50k)
- **Our best REAL score:** 655.4 (a v11 Horizon re-measure, 06-19); active floor is v8 ≈ 636–645 (latest v8 = 644.6). NOTE: TrueSkill ±150 variance — these are noisy re-measures of the same agents, not gains.
- **Ladder:** top = 1616.5 (Isaiah @ Tufa Labs). Gap ≈ **+960 above us** — far behind; 4729 teams.
- **Deadline:** 2026-07-07 23:59 → **12 days left.**
- **Submissions:** enabled.
- **Pareto verdict (latest measure per label):** front = {`v8` 644.6}. Dominated: v11 637.1, v7b 614.0. Tool says **HOLD v8**.
- **Next action:** Hand-tuning/re-measuring agents has not closed a ~960-pt gap; per playbook the lever is search/RL ABOVE the heuristic teacher (PPO self-play that finishes within the time limit), not more variance-chasing re-submits. Hold v8 as floor; invest remaining 12 days in a trained planner, not re-measures.

---

## neurogolf-2026  (Research, $50k)
- **Our best REAL score:** 7129.07 (`franksunp_7128` base + 6 true-rule rebuilds, 400/400 solved).
- **Ladder:** top = 7940.11 (Matheus & Fritz & Tony). Gap to top ≈ **811**. Top-5: 7940 / 7911 / 7879 / 7839 / 7823.
- **Deadline:** 2026-07-15 23:59 → **20 days left.**
- **Submissions:** enabled.
- **Pareto verdict:** front = {`franksunp_7128_truerule_rebuild` 7129.07}. Dominated: 7128.81-audited, kojimar 7114.66 / 6508.55 / 6507.21. Tool says **HOLD**.
- **Next action:** Our edge over public base is only +0.26 (7129.07 vs 7128.81); 811 to the top. RECON the current public SOTA notebook (top moved to ~7940) and audit whether a fresher base transfers, rather than squeezing the franksunp base further.

---

## ai-agent-security-multi-step-tool-attacks  (Featured, $50k)
- **Our best REAL score:** 51.750 (`N=575`). Two recent attempts (N=720, N=800) returned **BLANK** — the timeout/N-cliff wall (avoid N≥580).
- **Ladder:** top = 100.490 (Victor Merckle). Gap to top ≈ **48.7**. Top-5: 100.49 / 95.31 / 93.76 / 93.32 / 89.55.
- **Deadline:** 2026-09-01 23:59 → **68 days left.**
- **Submissions:** enabled (but high-N submits fail/blank on timeout).
- **Pareto verdict:** front = {`N575` 51.75}. Dominated: N570 51.30, N550 49.50. Linear N-curve holds in the 550–575 band (~0.09/N), but the score cliffs to blank at N≥580. Tool says **HOLD N575**.
- **Next action:** Pure N-scaling is capped by the timeout wall well short of the top (we ~51.8 vs top ~100). Cut per-candidate runtime (the report-suppression path was already exploring this) so a higher N fits the budget without blanking, OR change the attack mechanism — N-tuning alone won't double the score.
