# Tabular — Playground and tabular competitions

Here the winner is **volume of out-of-fold predictions × diversity of sources × Hill Climbing**. Not one genius model, but dozens of different OOF the ensemble picks the best subset from.

## Contents
- [Core principle: the OOF factory](#core-principle-the-oof-factory)
- [Hill Climbing ensemble](#hill-climbing-ensemble)
- [Multi-level stacking](#multi-level-stacking)
- [Pseudo-labeling](#pseudo-labeling)
- [Predictors — the diversity pool](#predictors--the-diversity-pool)
- [Feature engineering](#feature-engineering)
- [Validation](#validation)
- [The flat zone](#the-flat-zone)

---

## Core principle: the OOF factory

The goal is dozens of OOF predictions from combinations of:

```
[feature sets] × [TabPFN / XGB / LGBM / CatBoost / NN / AutoGluon] × [hyperparams / seeds]
```

Each combination = one kernel = one OOF file (train predictions by fold) + test predictions. Run them in **parallel batches** under different slugs. Small tabular data trains in minutes → dozens of runs an evening is realistic.

**Pull other people's OOF.** Top public notebooks often publish predictions — that's free diversity for your ensemble. During RECON, collect not just ideas but the actual submission/OOF files.

Keep a single registry: `results.csv` (slug, features, model, params, CV, LB) in a private dataset. Without it you'll be lost by experiment #30.

## Hill Climbing ensemble

The finale is NOT a manual average. Hill Climbing over OOF picks the weights/subset itself:

1. Start with the best single OOF.
2. Greedily add the predictor that moves the CV metric the most (with repeats — you can add the same one several times = weights).
3. Stop when adding stops helping.
4. Apply the same weights to the test predictions.

It beats a manual `mean()` almost always, because it drops harmful predictors and weights useful ones. Runs in seconds on CPU.

## Multi-level stacking

When you have many OOF and Hill Climbing has plateaued — stack in levels (this is what top Playground 2025 teams did):

- **L1:** base — TabPFN, LGB, XGB, CatB on different features / with pseudo-labels. Dozens of models.
- **L2:** meta-models (LGB + RF + Ridge) over L1 OOF (+ optionally the raw features).
- **L3:** ElasticNet / Ridge over L2, then blend with the non-pseudo L2 stack; weights via Nelder-Mead.

Deep stacks are expensive (hundreds of fits), so compute on **cuML (GPU)** — hours instead of days. See `arsenal.md`.

## Pseudo-labeling

Turns the unlabeled test into training signal:

1. Train your best model, predict the test.
2. Take confident predictions (or all, with soft weights) as "labels."
3. Retrain on train + pseudo-labeled test.

Gives a boost where the number of examples matters. Careful: only from an honest CV scheme, or you leak. Pairs well with stacking — pseudo on L1, a clean stack as insurance on L2. For an imbalanced + balanced-metric setup, pseudo-label confident **minority** classes to lift their recall instead of flooding the majority.

## Predictors — the diversity pool

Different model natures → different errors → a stronger ensemble:

- **TabPFN v2.5** — foundation model, no training, seconds; often the strongest single model on small/medium data. A must-have in the pool. (If you subsample its context too hard on large data, it gets crippled — bag several fits on a larger context instead.)
- **GBDT** — LightGBM, XGBoost, CatBoost. The workhorses. Different seeds/depths/features = different OOF.
- **NN** — tabular MLP / TabNet / FT-Transformer; errs differently from trees → valuable in the ensemble even if weaker alone.
- **AutoGluon** — a whole stack out of the box; another independent OOF.

Rule: one OOF predictor = death in the flat zone. Minimum 3 different natures.

## Feature engineering

- Count / target / frequency encoding of categoricals (target strictly out-of-fold, or you leak).
- Arithmetic interactions (ratios, differences, products of numerics).
- Group aggregations (mean/std/min/max of target or numerics by category).
- Look at what worked in **past seasons of the same series** — Playground repeats patterns (arsenal.md → farid.one).

## Validation

- Correct KFold / StratifiedKFold, **the same metric as the LB** (don't guess — confirm), zero leakage between folds.
- CV must correlate with LB. They diverge → fix CV (grouping, time split, duplicates) FIRST.
- A slightly pessimistic but correlated CV beats an optimistic broken one.
- **Probe the metric via a known point.** Before building local validation, work out what the LB actually computes: look at the score of `sample_submission` or a constant prediction. Battlefield example: 3 classes, sample = all one class, LB = **0.33333** = 1/3 → the metric is **balanced_accuracy** (macro-recall), not plain accuracy. Had you optimized accuracy with `class_weight=None`, you'd have regressed. One probe saves days of optimizing the wrong function. (Plain `accuracy` from a constant would give the class share, not 1/N — the number itself reveals the metric's nature.)
- **The metric IS the `class_weight`.** balanced_accuracy / macro-F1 → "balanced" class weights are justified (minority classes weigh as much as the majority). Plain accuracy → `class_weight=None`, don't sacrifice the majority for the minority. Get this wrong and you systematically lose score.

## The flat zone

Hundreds of teams on the same score = everyone submitting one public notebook. The exit is not "tune that same notebook better" but **introduce predictions it doesn't have**: TabPFN + your own FE + others' OOF + a stack. Unique diversity is what shifts you above the plateau.
