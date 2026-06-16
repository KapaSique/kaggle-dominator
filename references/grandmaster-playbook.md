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
