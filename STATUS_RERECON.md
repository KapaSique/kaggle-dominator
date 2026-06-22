# Re-RECON Scout Digest — 2026-06-22

Playbook rule 7 (stale-base trap): reproducing a fresh best-public base dwarfs hand-grinding.
Only REAL scores actually read from submissions CSV / leaderboard / kernel titles. No invented numbers.

## Fronts

- **playground-series-s6e6** — our best REAL **0.97183** (cand_A_anchor = public best-single). LB top **0.97284** (yuki #2). Top public found: stacks cdeotte/GPU-LR (06-08), philippsinger/TabPFN-3 (06-03), meenalsinha ensemble — none state a score > 0.97183; newest kernels (kospintr/odvut1/anasriaz, 06-21/22) list no beating score. VERDICT: hold — none newer.

- 🔴 **orbit-wars** — our best REAL **633.6** (v11 finals active-2; v8 re-measured 658.2/651.6). LB top **1707.6** (Isaiah @ Tufa Labs); LB field broadly 1500–1700. Public bases openly state **1000–1224**: romantamrazov "Orbit (Star) Wars | LB: MAX 1224" (1224, 144 votes), alycemiki "[Light ver. & 1200+]" (06-15), djenkivanov "Passed 1,000" — plus the dominant Producer meta: slawekbiel "The Producer V2" (255 votes, 06-12) & romantamrazov "I'M STRONGER" (374 votes, 06-08). All dwarf our 633.6 on the same LB metric. VERDICT: REPRODUCE: slawekbiel/the-producer-v2 (Producer meta, 255 votes) — or romantamrazov/orbit-star-wars-lb-max-1224 scores ~1224 > our 633.6, steps: fork the public agent kernel verbatim → run → submit agent → re-measure on current field, then graft onto our v8/v11 set. Huge stale-base upside.

- **neurogolf-2026** — our best REAL **7128.81** (franksunp audited mark-b, already a public-base repro; +606 lineage from kojimar). LB top **7889.45** (Fritz & Tony). Highest STATED public score = kojimar "[7116.91 LB] Audited ONNX Overrides" — 7116.91, *below* our 7128.81; others stated lower (vyanktesh 6425.76, octaviograu 6154.71). Watch: seddiktrk "Graph Surgeon" (152 votes, re-run today 06-22) lists no score. VERDICT: hold — none newer with a stated score above us.

- **ai-agent-security-multi-step-tool-attacks** — our best REAL **51.750** (N=575 cliff probe; N=570=51.3 banked). LB top **100.490** (Victor Merckle). Real headroom but NO public kernel states a score > 51.75: pilkwang Replay-Dense (204 votes, no number), nawfeelrahman "Baseline 4.900" (4.9), recent yaroslavkholmirzayev "Replay Dense Boundary Aggressive" / "k1-short" (06-22, no stated number). VERDICT: hold — none with a stated score beating us.

- **pokemon-tcg-ai-battle** — our best REAL **1062.2** (makthanithin 1084.5 baseline repro; Alakazam decorrelation shot landed 778.4, worse). LB top **1367.5** (keidroid). Recent stated public bases below us: romanrozen "Baseline Agent V10 | LB 950+" (950, 101 votes, 06-21). makthanithin 1084.5 already reproduced (we land 1062.2, variance under stated). VERDICT: hold — none newer with a stated score clearly above 1062.2.

## Summary
🔴 reproduce-upside front: **orbit-wars** (our 633.6 vs public 1000–1224 / LB 1707).
Holds: playground-series-s6e6, neurogolf-2026, ai-agent-security-multi-step-tool-attacks, pokemon-tcg-ai-battle.
