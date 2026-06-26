# Nightly Monitor Digest — 2026-06-26

All scores below are REAL publicScores read tonight. Leaderboards are sorted
descending (top row = best), so higher = better on every front here.

## playground-series-s6e6 (tabular)
- **Our best real:** 0.97183 (candA_anchor).
- **Ladder vs top:** top public 0.97284 (yuki #2); gap −0.00101. 0.972 cluster (0.97263/0.97254/0.97253) sits above us.
- **Deadline:** 2026-06-30 23:59 — **4 days left ⚠ UNDER 7**.
- **Submissions:** appear enabled (last accepted+scored 06-16).
- **Pareto:** candA_anchor is the SOLE non-dominated member (1/5). Dominated/over-engineered: softvote 0.97165, candC_override 0.97159, consensus 0.97148, TabPFN3_stacker 0.97061 — every blend lost to the raw anchor. Tool says **HOLD candA_anchor**.
- **Next action:** lock candA_anchor (0.97183) as one of the 2 final selections before 06-30; only a richer GPU base (not more blending of the same pool) can climb the 0.972 cluster.

## orbit-wars (RTS simulation)
- **Our best real:** 645.3 (v11) — note TrueSkill scores are noisy (v8 measured 627.9/636.2, v11 645.3/655.4).
- **Ladder vs top:** top 1633.0 (Isaiah @ Tufa Labs); gap ≈ −988. Top-5 all 1511–1633.
- **Deadline:** 2026-07-07 23:59 — 11 days left.
- **Submissions:** appear enabled (last accepted+scored 06-23).
- **Pareto:** v11 sole non-dominated member (1/3); dominated v8 627.9, v7b 614.0. Tool says **HOLD v11**.
- **Next action:** no new submit; keep v8+v11 as the double-insurance active set. Large gap to top suggests algorithmic depth, not heuristic tuning, is still the lever.

## neurogolf-2026 (program-synthesis / ARC-like)
- **Our best real:** 7129.07 (franksunp_7129 base + 6 true-rule rebuilds).
- **Ladder vs top:** top 7951.41 (Matheus & Fritz & Tony); gap ≈ −822. Top-5 7848–7951.
- **Deadline:** 2026-07-15 23:59 — 19 days left.
- **Submissions:** enabled (last accepted+scored TODAY 06-26 = 7119.59).
- **Pareto:** franksunp_7129 sole non-dominated member (1/5). Dominated: crossdump_7150attempt **7119.59** (tonight's new sub — claimed local 7150.41 / 400-of-400, scored BELOW the banked base), franksunp_markb 7128.81, kojimar_7114 7114.66, kojimar_6508 6508.55. Tool says **HOLD franksunp_7129**.
- **Next action:** keep franksunp_7129 (7129.07) as active best; the local↔LB gap on the cross-dump harvest is real — diagnose why local-verified 7150 dropped to 7119 LB before any further per-task harvesting.

## ai-agent-security-multi-step-tool-attacks (agent security)
- **Our best real:** 51.750 (N=575 probe).
- **Ladder vs top:** top 100.490 (Victor Merckle); gap ≈ −48.74. Top-5 89.5–100.5.
- **Deadline:** 2026-09-01 23:59 — 67 days left.
- **Submissions:** enabled, but two recent N=720 / N=800 runs returned BLANK (timeout/failed) — N-cliff/timeout wall confirmed.
- **Pareto:** N575 sole non-dominated member (1/3); dominated N570 51.300, N550 49.500 (monotone in N up to the cliff). Tool says **HOLD N575**.
- **Next action:** probe N between 575 and 720 (e.g. ~620–650) to find the timeout cliff edge without blanking; 51.75 → top 100 needs a different mechanism, not just more N.

MONITOR_DONE
