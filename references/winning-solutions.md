# Winning-solutions reference — recon base for "steal the structure, not the code"

Top Kagglers don't start from scratch — they recon the best public/winning solutions and graft the
*structure* (CV scheme, feature ideas, ensemble shape), then add their own edge. This file is the
recon base: how to mine winning solutions on RECON, and the proven structures distilled so far. Grow
it (the curator appends measured ones). Inspired by KaggleGrandMaster-LM (a model trained on scraped
top-solution notebooks) and validated by NVIDIA's 2026 Grandmaster-playbook win.

## How to mine on RECON (do this for every new comp)

1. **Discussion → Most Votes** + **Code → Most Votes**: read the top 5 solution writeups/notebooks. Note CV scheme, features, model nature, ensemble shape, post-processing.
2. **Past winning solutions of similar comps**: search `competition-name + "1st place solution"` and the Kaggle "Competition Solutions" wiki. The structure transfers even when the data doesn't.
3. **Pull public OOF**: top notebooks publish out-of-fold predictions — fold them into your Hill-Climb for free diversity. Recon collects FILES, not just ideas.
4. **Record** each into the table below: comp-type, the structure, the measured result.

## Proven winning STRUCTURES (distilled — graft these)

| Comp type | Winning structure | Evidence |
|---|---|---|
| Tabular (Playground) | EDA → trustworthy CV → magic-feature search (groupby aggs, OOF target-enc) → many diverse base models → **Hill-Climb / multi-level stack** → metric-matched post-proc | NVIDIA 2026: 850 experiments → 4-level/150-model stack = **1st**; Deotte 3-level cuML stack |
| Tabular (small/noisy) | "Raw is Law": shallow stumps + rank-averaging, watch CV↔LB gap, don't add complexity CV doesn't pay for | S6E2 4th = max_depth=2 + rank-avg = 0.95534 |
| CV / image | EfficientNet + Swin/ViT ensemble (complementary biases) + TTA + EMA + pseudo-labeling | 4419-writeup study: EfficientNet+LGBM+augmentation most-mentioned |
| NLP | DeBERTa-v3-large + varied pooling heads/prompts → stack; MLM-pretrain in-domain | CommonLit winner |
| Simulation/agent | RL self-play (PointNet/attention) > hand heuristic at the top; validate on a POOL of opponents not one | orbit top 1723 (RL-class) vs our 684 (heuristic) |
| Code-golf (ONNX) | TRUE-rule rebuilds (cheapest ops), grader-verified on held-out — NEVER memorize visible pixels | neurogolf: our convfit 6102 private (memorized) vs true-rule rebuilds |
| Judge-scored hackathon | Get the RUBRIC first, write strictly to its weights, lead with honest measured findings (integrity > faked edge) | Pokemon $240k: Model 70%/Deck 20%/Report 10% |

## The single transferable meta-lesson
**Scale + trustworthy CV + diversity beats a clever single model.** NVIDIA's win wasn't a genius model — it was 850 disciplined experiments hill-climbed into a stack, with a human steering. Our gap to the top is almost always *depth/scale of the right structure*, not a missing trick. Recon tells you which structure; the grind supplies the scale; the Pareto/held-out gate keeps it honest.
