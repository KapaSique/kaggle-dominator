# Winning-solutions reference — recon base for "steal the structure, not the code"

Top Kagglers don't start from scratch — they recon the best public/winning solutions and graft the
*structure* (CV scheme, feature ideas, ensemble shape, action schedule), then add their own edge.
This file is the recon base: how to mine winning solutions on RECON, and the proven structures
distilled so far. Grow it (the curator appends measured ones). Inspired by KaggleGrandMaster-LM (a
model trained on scraped top-solution notebooks) and validated by NVIDIA's 2026 Grandmaster-playbook
win.

## How to mine on RECON (do this for every new comp)

1. **Discussion → Most Votes** + **Code → Most Votes**: read the top 5 solution writeups/notebooks.
   Note CV scheme, features, model nature, ensemble shape, post-processing.
2. **Past winning solutions of similar comps**: search `competition-name + "1st place solution"` and
   the Kaggle "Competition Solutions" wiki. The structure transfers even when the data doesn't.
3. **Pull public OOF**: top notebooks publish out-of-fold predictions — fold them into your
   Hill-Climb for free diversity. Recon collects FILES, not just ideas.
4. **On ladder comps, mine EPISODES too** — the leaders' source is hidden but their actions are
   public. See "Replay distillation" below.
5. **Deduplicate sources by hash before counting them as agreement.** Two public notebooks
   presented as independent confirmations turned out to share one SHA-256 anchor — they were one
   strategy, not two votes for it.
6. **Record** each into the table below: comp-type, the structure, the measured result.

## Proven winning STRUCTURES (distilled — graft these)

| Comp type | Winning structure | Evidence |
|---|---|---|
| Tabular (Playground) | EDA → trustworthy CV → magic-feature search (groupby aggs, OOF target-enc) → many diverse base models → **Hill-Climb / multi-level stack** → metric-matched post-proc | NVIDIA 2026: 850 experiments → 4-level/150-model stack = **1st**; Deotte 3-level cuML stack |
| Tabular (small/noisy) | "Raw is Law": shallow stumps + rank-averaging, watch CV↔LB gap, don't add complexity CV doesn't pay for | S6E2 4th = max_depth=2 + rank-avg = 0.95534 |
| CV / image | EfficientNet + Swin/ViT ensemble (complementary biases) + TTA + EMA + pseudo-labeling | 4419-writeup study: EfficientNet+LGBM+augmentation most-mentioned |
| DL on a **shared public weights package** | Adopt the freshest public frontier fast, carry ONE own measured delta onto each new base, diversify by **pretraining regime** (self-supervised + domain-pretrained), run lineages in parallel | RSNA Knee: two adoptions = +0.011 (195 ranks) and +0.005 (133 ranks) while own-ensemble work returned +0.001 |
| NLP | DeBERTa-v3-large + varied pooling heads/prompts → stack; MLM-pretrain in-domain | CommonLit winner |
| Simulation/agent (ladder) | **Replay distillation**: download top teams' episodes → find the schedule that REPEATS across matches → distil into a deterministic tape → keep your own measured wrapper on top → gate paired-seat over untouched seeds | Kaggriculture leaders: distilled tapes gated 27-3, 35-5/20 seeds, 90-10/10 seeds |
| Simulation (open frontier, no strong public agent) | RL self-play (PointNet/attention) > hand heuristic at the top; behaviour-cloning a teacher plateaus AT teacher strength | orbit top 1723 (RL-class) vs our 684 (heuristic) |
| Economic / market simulation | Model the **price impact of your own actions**: metered batches over dumps, profit per *action-turn* over profit per item, liquidation before the horizon ends | Kaggriculture: premium goods crash to floor after ~60-80 units; ballast goods need ~3000 |
| Code-golf (ONNX) | TRUE-rule rebuilds (cheapest ops), grader-verified on held-out — NEVER memorize visible pixels | neurogolf: our convfit 6102 private (memorized) vs true-rule rebuilds |
| Red-team / agent security | Replay-DENSE severity stacking; but confirm WHICH board carries the prize gap before committing | JED: public NBs cap ~58-61, prize cluster 72-100 sat on the private board |
| Judge-scored hackathon | Get the RUBRIC first, write strictly to its weights, lead with honest measured findings (integrity > faked edge) | Pokemon $240k: Model 70%/Deck 20%/Report 10% |

## Replay distillation — the ladder analogue of "reproduce the best public notebook"

On a fixed-test comp you copy the leaders' *code*. On a ladder you cannot — but episodes are
public, so you copy their *behaviour*. The public top of Kaggriculture runs this loop, and it is
the single highest-yield ladder method observed:

1. **Download episodes for the displayed top teams**, matching each episode to the submission that
   actually produced the displayed score. A team can have two active submissions; taking their
   newest games samples their *weaker* newer agent instead of the ranked one.
2. **Accept repetition, not brilliance.** One replay is one match — the loser may be excellent and
   the winner may have had the seed. Require the same schedule against several opponents.
3. **Distil into a tape** (a deterministic per-turn action schedule) plus a small reactive
   controller for the endgame.
4. **Gate from both seats over untouched seeds** before promoting.
5. **Keep your own measured wrapper** on top of each new base you adopt, exactly as on DL fronts.

**Where the edge lives, once everyone converges.** When top teams converge on the same plan,
attribute the remaining difference by *counting changed decisions per subsystem*: one promoted
Kaggriculture candidate changed **20 field turns but 112 market turns**, proving the live edge was
sale timing rather than production. Swap subsystems between agents (A's field + B's market and the
reverse) to locate which half carries the strength.

## The single transferable meta-lesson
**Scale + trustworthy validation + diversity beats a clever single model.** NVIDIA's win wasn't a
genius model — it was 850 disciplined experiments hill-climbed into a stack, with a human steering.
Our gap to the top is almost always *depth/scale of the right structure*, not a missing trick.
Recon tells you which structure; the grind supplies the scale; the Pareto/held-out/paired-seat gate
keeps it honest.

**And the corollary that decides ladders:** reproduce-best-public wins there too. A reproduced
public agent beat this account's own hand-built agent by **+289 rating**. The exception is a front
with no strong public base — then, and only then, is original method work the cheapest path.
