# Re-RECON digest — 2026-06-25

Periodic stale-base scout (playbook rule 7: reproducing a fresh best-public base
dwarfs hand-grinding — neurogolf gave +606). Only REAL scores actually read below.

- **playground-series-s6e6** — our best **0.97183** (cand_A_anchor, REAL). Top public LB **0.97284** (yuki #2). Top reproducible public kernels = cdeotte GPU-LR stacker / philippsinger TabPFN-3 (already our bases); none state >0.97183. VERDICT: **hold — none newer** (we sit at public-best-single; leaders ensemble privately).

- **orbit-wars** — our best **636.2** (v8, finals-locked 06-23, past deadline). Top public LB **1558.7** (Isaiah @ Tufa Labs). Highest public kernel stated ~**1224** (romantamrazov, 2026-04-20, old/known). Agent comp now in finals — no submit possible, reproduce no longer actionable (was 🔴 on 06-22). VERDICT: **hold — none newer**.

- **neurogolf-2026** — our best **7129.07** (REAL, grader-verified 400/400). Top public LB **7940.11** (private, climbing). Highest *reproducible public kernel* = vyanktesh **6425.76** / octaviograu 6154.71 / kojimar audited-onnx refs **6029.09 / 5800.55** — all BELOW us. VERDICT: **hold — none newer** (we lead the public-kernel pool; the 7900s are private).

- 🔴 **ai-agent-security-multi-step-tool-attacks** — our best **51.750** (REAL; N=570 banked). Top public LB **100.490** (Victor Merckle). NEW public base `caoyupeng/v23-alpha2co-667-break60` (2026-06-22, 71 votes): prior V22 URLCompact 642 **attested 57.78 LB**, V23 targets **60.03** via 667 candidates × 0.09/replay using shorter aa.co-style replay encoding (no search/gateway/GPU). VERDICT: **REPRODUCE: caoyupeng V23 (60.03 target, base 57.78 attested) > our 51.75** — steps: adopt URLCompact+Alpha2CO replay encoding, push candidate count toward ~667 within time_budget_s, cross-check against our N-curve (slope 0.094/N), avoid the N≥580 timeout wall.

- **pokemon-tcg-ai-battle** — our best **1062.2** (makthanithin baseline repro, REAL). Top public LB **1353.4** (keidroid). Highest public agent kernel = romanrozen V10 **LB 950+** < our 1062; kiyotah/pilkwang deck baselines lower. None stated >1062. VERDICT: **hold — none newer**.

## Summary
🔴 reproduce-upside front: **ai-agent-security-multi-step-tool-attacks** (our 51.75 vs caoyupeng 57.78→60 reproducible).
Holds: playground-series-s6e6, orbit-wars (finals-locked), neurogolf-2026, pokemon-tcg-ai-battle.

RERECON_DONE — 🔴 fronts: ai-agent-security-multi-step-tool-attacks
