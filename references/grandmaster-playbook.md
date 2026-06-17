# Grandmaster Playbook — how the top actually operates

This is the cross-cutting craft that separates gold from bronze, distilled from how real Grandmasters work (bestfitting, Chris Deotte, Abhishek Thakur, Jean-François Puget, Gilberto Titericz, Marios Michailidis) and from public winning writeups. The type-specific arsenals (`tabular.md`, `deep-learning.md`, …) tell you *which techniques*; this tells you *how the winners think*. **Read it on RECON, every competition.** The single highest-leverage habit in all of it: a trustworthy CV is half the win.

## Contents
- [1. The CV doctrine — the #1 thing](#1-the-cv-doctrine--the-1-thing)
- [2. Ensembling craft](#2-ensembling-craft)
- [3. Feature-engineering craft](#3-feature-engineering-craft)
- [4. Deep-learning craft](#4-deep-learning-craft)
- [5. Meta-game and post-processing](#5-meta-game-and-post-processing)
- [6. The operating system](#6-the-operating-system)
- [Top 10 decisive edges](#top-10-decisive-edges)

---

## 1. The CV doctrine — the #1 thing

> *"A good CV is half of success."* — bestfitting (former world #1). *"If you have a good CV scheme in which validation data is representative of training and real-world data, you will build a model that's highly generalizable."* — Abhishek Thakur (4× GM)

- **Build the validation BEFORE the model.** Thakur's rule: split the data first, decide *how* to split (StratifiedKFold vs GroupKFold vs time split) before touching features. Everything downstream is guesswork if the CV is wrong.
- **Every change must move BOTH CV and public LB.** If a change lifts LB but not CV, distrust it — you're probably overfitting the public slice. Keep only changes the CV confirms.
- **Public LB is a lodestar, not the target.** Establish a positive CV↔LB *correlation* on a few submissions, then optimize CV and largely ignore the public number. A widening CV↔LB gap is the #1 overfitting red flag.
- **Adversarial validation** when you suspect train≠test drift: label train=0 / test=1, drop the target, train a binary classifier, read AUC. AUC≈0.5 → same distribution, trust random KFold. AUC≫0.5 → distributions differ; rebuild your validation set from the training rows the classifier most confidently thinks are "test-like", so your CV mirrors the real test.
- **Simulate the shakeup.** Estimate the public/private split and use multiple CV folds to model how much your rank could move — a sanity check against public overfit (an NVIDIA-GM habit).
- **GroupKFold the leakage.** If a group (patient, user, image series, time) spans multiple rows, a random fold leaks it across train/val and inflates CV. Split by the group. This is the most common silent CV-breaker.
- **Accelerate the loop.** Deotte's #2 (after "robust local validation"): run experiments as fast as possible — cuML / cuDF on GPU turns a day of folds into an hour, so you test 10× more ideas.
- **Exceptions to trust-CV:** tiny datasets, a public LB that is a large/representative fraction of the test, or a metric where CV is genuinely unstable. Know which regime you're in before deciding how hard to lean on CV.

## 2. Ensembling craft

> *"I love hill climbing because it takes lots of models and picks the best small subset — like Lasso — and computes the weights. It chooses DIVERSE models, not the best-CV ones."* — Chris Deotte

- **Hill-Climb over OOF for the blend.** Greedily add the predictor that most improves the CV metric, with repeats (= weights). It drops harmful models and rewards diversity over individual strength. Seconds on CPU; beats manual averaging almost always. (`hillclimbers` module → 4th place S3E14.)
- **Diversity > individual accuracy.** The best ensemble members are often mediocre alone but decorrelated. Generate diversity deliberately: different model natures (GBDT/NN/TabPFN), feature sets, seeds, encodings, fold schemes.
- **Multi-level stacking when HC plateaus.** L1 = many diverse base models (GBDT + NN + SVR + KNN + …); L2 = meta-models (Ridge/LGB/RF) trained on L1 OOF; L3 = blend. Deotte won the April-2025 Podcast comp with a 3-level cuML stack (private RMSE 11.44). A classic deep stack: 33 models → +3 → weighted mean.
- **Match the ensembling to the metric.** AUC → **rank-average** (not raw average). LogLoss → simple average of probabilities, then clip/scale (×0.99–1.01) toward safety. Sometimes geometric mean > arithmetic. Getting this wrong leaves free score on the table.
- **Regularize the weight search.** When optimizing many weights (Optuna/Nelder-Mead over 27 model combos), bound weights to e.g. [0.1, 1] instead of [0, 1] — a regularizer that stops the optimizer overfitting the OOF (CIBMTR 1st place).
- **"Raw is Law."** Simplicity can win: an S6E2 4th place used max_depth=2 stumps + RealMLP distillation + **rank averaging** to 0.95534, beating complex stacks by strictly watching the CV gap. Don't add complexity the CV doesn't pay for.
- **Pull public OOF.** Top notebooks publish predictions — fold them into your HC for free diversity. Recon collects files, not just ideas.

## 3. Feature-engineering craft

> Deotte's "magic feature" search: brute-force `groupby(COL1)[COL2].agg(STAT)` across thousands of (COL1, COL2, STAT) combos using cuDF-pandas — mean/std/count/min/max/nunique/skew — and let the metric find the gold.

- **Out-of-fold (nested) target encoding, always.** Compute the category→target statistic using only the other K-1 folds, never the row's own fold, or you leak the answer. Add smoothing/credibility for rare categories. Puget: "there was a barrier to entry; now everyone does it — master it and it's still an edge."
- **Aggregations are where tabular gold lives.** Group-by statistics, distribution descriptors (not just one stat per group — encode the whole distribution), categorical interactions, ratios/differences of numerics. Search broadly, keep what the CV rewards.
- **Know when raw beats engineered.** Strong GBDTs and especially TabPFN often extract interactions themselves; over-engineering can add noise. Always A/B engineered vs raw on the CV.

## 4. Deep-learning craft

> A study of 4,419 winning writeups: the most-mentioned techniques are **EfficientNet, LightGBM, data augmentation, ensembling**, plus cross-validation, label smoothing, and Mixup.

- **Ensemble different architectures, don't tune one.** CNN (ResNet/EfficientNet/ConvNeXt) + Transformer (Swin/ViT) ensembles beat a single tuned backbone — complementary inductive biases decorrelate errors.
- **NLP: DeBERTa-v3-large is the default backbone** (RTD pretraining + disentangled attention). Build diversity through varied **custom pooling heads, inputs, and prompts**, then stack (LGBM/XGB/CatB meta). CommonLit winner did exactly this.
- **MLM-pretrain on in-domain text** before fine-tuning NLP — small but real (NBME +0.002 CV; AI4Code ran 40 days of MLM on DeBERTa-v3-large).
- **Pseudo-labeling in rounds.** Predict the test with your best ensemble, add confident predictions to train, retrain, repeat. Decisive in CV/NLP ("more data" dominates). Keep a non-pseudo version as insurance.
- **TTA + EMA + the right schedule.** Average predictions over augmented test views (original + flips + multi-scale). EMA of weights, cosine schedule, label smoothing, Mixup/CutMix — these recur in golds.
- **What repeatedly does NOT work** (a 3rd-place ranked list): decoder models for encoding tasks, AWP, FGM, weight decay, constant LR, handcrafted features bolted onto transformers, and GBT for the stacking layer. Don't burn time re-discovering these.
- **OOF for DL too.** Save per-fold out-of-fold predictions so DL models feed the same Hill-Climb/stack as everything else; watch group/patient leakage in images.
- **Iteration speed first.** Defer ideas that slow training to late in the comp; early on, the ability to try many things matters more than any single heavy model.

## 5. Meta-game and post-processing

- **Hunt for leaks first.** Famous wins came from leakage: SETI (file timestamps + row order encoded the label), ASHRAE (public/"leaked" external data). Check id patterns, timestamps, row order, duplicates before modeling — and drop duplicates / keep external data out of validation so leaks don't corrupt your CV.
- **Post-process to the exact metric.** Threshold-tune for F1 (sweep, not 0.5), ideally **per group/source** (e.g. 0.005 steps per source). Rank-transform for AUC. Class-weight / threshold tuning for balanced accuracy. Calibrate probabilities when the metric is threshold-based.
- **Shakeup survival = submission selection.** You pick 2 finals: one **safe** (best CV, robust) + one **aggressive**. In high-shakeup comps, bet on CV and refuse the seductive public-LB-overfit notebook — the competitor hovering at rank 1700 on public CV-discipline routinely wins the private board.
- **Legitimate LB probing.** A constant/sample submission reveals the metric's nature and test class balance (e.g. all-one-class → 1/N means a macro/balanced metric). Use it to estimate distributions, never to reverse-engineer labels.

## 6. The operating system

> bestfitting's loop: (1) find similar past competitions, (2) read their winning solutions + relevant papers, (3) analyze data & build a stable CV, (4) preprocess → FE → train, (5) error analysis (prediction distribution, hard examples), (6) design models to add diversity / solve hard cases, (7) iterate. He opens a **solution document in week 1** and updates it throughout.

- **Recon before code.** Spend the first days on EDA, reading *all* the discussion, finding the past-competition analog, and reproducing the best public baseline as your floor. Never start from scratch when a public 0.95 exists.
- **One change at a time, measured.** Isolate each idea so you know what moved the metric. Keep a single source of truth (a results table: slug, features, model, params, CV, LB) — by experiment #30, memory fails.
- **Volume of experiments.** GMs run dozens-to-hundreds of variants and let the ensemble choose. Your norm is dozens per session, not three. Seed-discipline and reproducibility make the volume trustworthy.
- **Know when to stop tuning and start ensembling.** Past a point, the marginal hour is better spent generating a *different* model than squeezing one. Recognize a dead idea fast and drop it.
- **Ego vs the leaderboard.** Don't fall in love with a clever idea the metric doesn't reward. Keep a protected best; a new idea is an additional attempt, never a replacement. The board is the only judge.

## Top 10 decisive edges

1. A trustworthy CV that correlates with LB — built before any model. (bestfitting / Thakur)
2. Adversarial validation when train≠test; rebuild val from test-like rows.
3. GroupKFold wherever a group spans rows — kill the silent leak.
4. Hill-Climb a *diverse* OOF pool for the blend; diversity > individual CV.
5. Multi-level stacking (cuML/GPU) when the blend plateaus.
6. Out-of-fold target encoding + brute-forced group-agg "magic" features.
7. Ensemble different architectures; pseudo-label in rounds; TTA — for DL.
8. Post-process to the exact metric (rank-avg AUC, threshold-tune F1 per group).
9. Two finals = safe (CV) + aggressive; trust CV through the shakeup.
10. Recon the top, reproduce the baseline, log every experiment, keep a protected best.

---

## Battle-proven additions (grow this)

Append MEASURED, transferable insights here as battles teach them — newest first. Format: `date — [comp type] insight (the evidence)`. Only what's proven by a number/result, never a hunch. This section is the skill learning from its own fights (see "CAPTURE WHAT YOU LEARN" in SKILL.md).

- 2026-06 — [any comp, process] **Confirm the metric DIRECTION by the leaderboard top vs bottom before claiming "X beat Y" — never infer it from a description.** The first autonomous curator run inverted neurogolf-2026 (assumed lower=better; the LB top was 7795 = higher-better, so 6287 > 6239 — its insights were backwards). A wrong direction silently records the opposite of the truth. One glance at the leaderboard ordering settles it.

- 2026-06 — [code comp] **Reproduce the top notebook VERBATIM before adapting.** Repeatedly adapting a top public notebook into your own framework scored BLANK or fell back to baseline; a verbatim copy of the best public notebook (wrapped only in a thin serve/fallback layer) was what actually scored. Adapt only *after* the verbatim version is confirmed working.
- 2026-06 — [blind/no-clean-label comp] **Calibrate a local surrogate on your measured submit→LB points before trusting it.** A pseudo-clean proxy showed Spearman 0.000 vs the real LB across known points — dead proxy. Always verify rank-correlation of any local validation against your handful of real LB points first; if it doesn't correlate, don't optimize against it.
- 2026-06 — [code comp, runtime-limited] **Over-returning causes TIMEOUT, not a higher score.** Returning far more candidates than the runtime/replay budget fits silently times out (= zero), while a count matched to the budget scores. Size N to the budget, not to the maximum.
- 2026-06 — [asymmetric-metric comp] **Asymmetric metrics are hypersensitive to a single axis — probe it.** Changing only the confidence of identical predictions moved the LB by ~+70; the metric's FP penalty scaled with confidence. Before modeling, probe how the score responds to one isolated axis (confidence, threshold, calibration).
- 2026-06 — [tabular Playground] **In a flat zone, the freshest public best-single notebook IS the lever, not your own models.** S6E6: weeks of own stacking plateaued at 0.97106; pulling the current public "anchor" (best-single, the whole 0.972 cluster converges on it) and submitting it *verbatim* jumped to 0.97183 (rank ~198→~91, top 11%→5%). RECON the *current* public SOTA repeatedly — it moves under you.
- 2026-06 — [tabular] **Voting/diversifying on top of the best-single net-HURTS when its disagreements live in coin-flip rows.** S6E6: raw anchor 0.97183 > my TabPFN-3 softvote 0.97165 > public voted-consensus 0.97148. The anchor's disagreements sat in rows where every signal had confidence <0.80 — irreducibly ambiguous. Diversification beat the *voted blend* but not the *raw best-single*. Don't blend a weaker signal into the best-single unless it's confidently right where the anchor is wrong.
- 2026-06 — [RTS simulation] **Algorithmic depth (exact forward simulator + lookahead rollouts) is the real lever, not heuristic tweaks.** orbit-wars: heuristic v1 542 → v5 573 → exact-simulator + rollout-planner v6 **737.9** (+165 in one step). Every later "improvement" (FFA fixes, deeper horizon, diplomacy) scored *below* v6. Build the lookahead; stop hand-tuning heuristics.
- 2026-06 — [cross-domain, MEASURED 6/8 comps] **Over-engineering past the peak is the #1 self-inflicted loss.** After finding a strong simple solution, adding complexity repeatedly degraded the real score: orbit v6 737.9 → v7–v11 ≤684; maze v18 1046 → economy-experiments v19/v20/v21 all lower (author reverted to v18); neurogolf holdout-validated v5 6287 → mega-multisource v6/v7 6239; freuid simple convnext_tiny384+TTA 0.354 → fullres/ensemble/multi-res 0.02–0.15; S6E6 anchor 0.97183 > softvote/consensus/override; security single-post-mass 18.11 > multi-hop stacking 6–15. When a simple thing scores, *protect it and stop* — the next clever layer usually costs you. "Raw is Law."
- 2026-06 — [simulation/ladder] **Local winrate vs a SINGLE bot does not predict the ladder.** maze-crawler: agents at 70% local winrate vs one LB-tier bot scored *below* the simpler proven agent on the real ladder ("same lesson as v11" — it happened twice). orbit echoed it: 4-player ladder rank diverged from 1v1 local. Validate against a *pool* of 3–5 distinct opponents and a past version of yourself, never one bot — rule 4, paid for in real rank.
- 2026-06 — [asymmetric-metric, unlearning] **On a confidence-scaled metric, a pure confidence transform can be the biggest single lever.** neural-debris: the public baseline at 259.79 + a sqrt confidence-boost (no model change) jumped to 329.53 (+70). When the metric rewards/penalizes by confidence magnitude, probe a monotonic conf transform (sqrt/power) on your best base *before* building new models — it's free and often dominates.
- 2026-06 — [code/security comp] **A reliable own harness beats a verbatim top notebook that BLANKs.** Running the top author's code verbatim returned empty/zero; the same *recipe* re-expressed inside a harness that always serves+falls-back scored (18.11). Reproduce the top's recipe, but inside a wrapper you've proven scores — verbatim the idea, not necessarily their exact runtime.
- 2026-06 — [tabular, private-LB] **Own honest CV models rarely beat the public consensus but hedge the shakeup.** s6e5 (closed, private known): public rank-avg consensus 0.95460-private was the best; own honest RealMLP+GBDT blends landed 0.95425-0.95454 — slightly behind but decorrelated from the public crowd, insurance for the private split. Submit one public-best + one own-honest for the shakeup.
- 2026-06 — [dense flat-zone tabular] **The raw public best-single can beat every blend OF it — verify a decorrelated signal's STANDALONE strength before blending it in.** S6E6: the whole 0.97218 cluster shared one public "anchor" submission = 0.97183. Blending it DOWN with my diverse signals net-HURT every time (softvote 0.97165, coalition-override 0.97159 < anchor 0.97183) because the anchor's disagreements concentrate in irreducibly-ambiguous rows (the diverse coalition there had stacker-confidence <0.80 = coin-flips). The "decorrelated" TabPFN-3 stacker I was mixing in was 0.97061 STANDALONE — weaker, so it dragged. Lesson: submit the raw best-single first; measure each blend member's standalone LB before trusting its diversity; diversity only helps when the member is also individually strong. Climbing past the cluster needed a richer BASE (GPU rebuild), not more blending of the same public pool.
- 2026-06 — [live comp] **Re-run RECON when the comp is still active — the public SOTA moves.** Stuck at an old BEST_KNOWN (0.97106) for a week; a fresh pull of public Code outputs revealed the ceiling had advanced to a shared 0.97183 anchor + a GM's TabPFN-3 stacker. Pulling current public OOF/submissions (not last week's) was a free +0.00077 (rank ~198→~91). `kaggle kernels list --competition <c> --sort-by voteCount/dateRun` + `kaggle kernels output <ref>` is the recon loop; re-walk it periodically, not once.
