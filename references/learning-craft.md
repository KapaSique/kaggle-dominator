# Learning craft — step-by-step checklists for practising the fundamentals

This file is the *how-to* companion to the technique files. Where `tabular.md`,
`deep-learning.md`, and `grandmaster-playbook.md` explain **which techniques win
and why**, this file gives a calm, ordered checklist for **doing each discipline
well while you are still learning it**. Work through the checklists in order the
first few times; later they become reflex.

Everything here uses only public data, the provided training labels, and
competition-compliant methods. Avoiding leakage is both good ethics and good
science — the same discipline that keeps a solution honest keeps it
generalising.

## Contents
- [1. Dataset understanding](#1-dataset-understanding)
- [2. EDA checklist](#2-eda-checklist)
- [3. Validation checklist](#3-validation-checklist)
- [4. Baseline modeling checklist](#4-baseline-modeling-checklist)
- [5. Feature engineering checklist](#5-feature-engineering-checklist)
- [6. Experiment tracking checklist](#6-experiment-tracking-checklist)
- [7. Error analysis checklist](#7-error-analysis-checklist)
- [8. Ensembling checklist](#8-ensembling-checklist)

---

## 1. Dataset understanding

Before any plot or model, learn what you were given. The goal is to be able to
say, in plain words, *what one row is, what you predict, and how you are scored*.

- [ ] Read the competition **Overview**, **Data**, and **Rules** tabs. The Rules
      define what is allowed (external data, team size, submission limits) —
      read them first, every time.
- [ ] Write one sentence: "**One row is ___; I predict ___; I am scored by
      ___.**" If you cannot fill the blanks, you do not understand the task yet.
- [ ] Confirm the **metric** — do not guess it. Probe it with the score of
      `sample_submission` or a constant prediction (see the metric-probe note in
      [tabular.md](tabular.md#validation)); the number reveals the metric's
      nature.
- [ ] Inspect **shapes**: rows/columns of train and test, target column,
      id column, obvious groupings (user, time, session).
- [ ] Check the **target**: type (binary/multiclass/regression), class balance,
      range, skew. Imbalance and skew change both the model and the CV scheme.
- [ ] Map **missingness**: which columns, how much, and whether it is random or
      informative (a column missing only for one class is itself a signal).
- [ ] Note **train vs test differences** in columns and value ranges. If test
      has categories or ranges train never shows, plan for it early.
- [ ] Flag **leakage risks**: ids that encode the target, timestamps that let
      the future leak into the past, duplicate rows across a fold boundary.
- [ ] Write 3–5 **hypotheses** ("feature X probably drives the target because
      …") to test later. Understanding first, optimisation second.

---

## 2. EDA checklist

Exploratory data analysis is how you turn "a table" into "a mental model". Keep
it purposeful — every plot should answer a question, not decorate the notebook.

- [ ] **Univariate:** distribution of each feature (histogram / value counts).
      Spot skew, outliers, constant or near-constant columns, unexpected codes
      (e.g. `-999` as a missing marker).
- [ ] **Target relationship:** how each feature relates to the target (grouped
      means, box plots per class, correlation for numerics). This ranks your
      feature-engineering ideas.
- [ ] **Bivariate / interactions:** pairs that may interact (a ratio, a
      difference, a group aggregate). Note candidates for [feature
      engineering](#5-feature-engineering-checklist).
- [ ] **Categoricals:** cardinality per column; rare levels; whether the same
      entity appears in both train and test (affects encoding and CV grouping).
- [ ] **Time / order:** if there is a timestamp, plot the target over time —
      trends and regime shifts mean a time-based split, not a random one.
- [ ] **Adversarial validation:** train a simple classifier to tell train from
      test. High AUC ⇒ a distribution shift you must design the CV around; near
      0.5 ⇒ train and test look alike and random KFold is safer.
- [ ] **Duplicates & near-duplicates:** exact and fuzzy. Duplicates spanning a
      fold boundary silently leak and inflate CV.
- [ ] Write down what each finding *implies* for validation and features. EDA
      that changes no decision was wasted.

---

## 3. Validation checklist

A trustworthy validation scheme is the single most valuable thing you build. The
CV doctrine (build validation *before* the model) is in
[grandmaster-playbook.md](grandmaster-playbook.md); the tabular specifics
(correct KFold, metric match, leakage) are in
[tabular.md](tabular.md#validation). This checklist is the ordered procedure.

- [ ] **Choose the split before the model**, matched to the data shape:
      - Class imbalance → **StratifiedKFold**.
      - Rows grouped by an entity (user/session/image-source) → **GroupKFold**,
        so no group leaks across folds.
      - Temporal data → a **time-based split** (train past, validate future).
- [ ] **Freeze folds and the seed** so every experiment is comparable. Save the
      fold assignment; reuse it for every model and OOF file.
- [ ] **Use the exact LB metric** in CV — confirmed by the probe, not assumed.
- [ ] **Zero leakage between folds:** fit all preprocessing (scalers, target
      encoders, imputers) *inside* the fold on the training part only.
- [ ] **Record the CV↔LB gap** on your first few submissions. A stable, small
      gap means the CV is trustworthy; a large or inverted gap means the CV is
      broken — fix it before optimising anything else.
- [ ] Prefer a **slightly pessimistic but correlated** CV over an optimistic one
      that diverges from the LB.
- [ ] In simulation/agent competitions, "validation" means a **pool of 3–5
      distinct opponents** plus a past version of yourself — never a single
      opponent (see [simulation.md](simulation.md)).

---

## 4. Baseline modeling checklist

Build two baselines, in this order. The first teaches you the problem; the
second is the score you must beat.

- [ ] **Own simple baseline first.** A plain model (logistic/linear regression,
      or a single LightGBM with defaults) on raw features, scored on your CV.
      Purpose: understand the task and learn the score *floor* — not to win.
- [ ] Confirm the **plumbing** end to end: it trains, predicts, writes a valid
      submission in the expected format, and the CV number is sane.
- [ ] **Reproduce a strong public baseline next**, with **attribution**: note
      the author and check the notebook's licence before reusing code. Run it as
      published; confirm it reproduces.
- [ ] For each step of the public baseline, ask **"why is this here?"** Reading
      to understand beats reading to copy — that understanding is the transfer
      you keep for the next competition.
- [ ] Lock the reproduced score as your **`BEST_KNOWN`** (see the PRESERVE THE
      BEST rule in SKILL.md). Every later idea is measured against it.
- [ ] Keep both baselines runnable — when a fancy idea underperforms, you fall
      back to a known-good starting point instead of debugging in the dark.

---

## 5. Feature engineering checklist

The technique menu (encodings, interactions, group aggregations, and the
strict-out-of-fold rule for target encoding) lives in
[tabular.md](tabular.md#feature-engineering). This checklist is the disciplined
*process* around it.

- [ ] Start from an **EDA hypothesis**, not a random transform. "The ratio of A
      to B should matter because …" — you learn more from a reasoned feature
      than from a scattershot one.
- [ ] Engineer one feature (or one small group) at a time and **measure it on
      CV**. Keep it only if the real metric improves.
- [ ] **Guard against leakage** in every feature: target/count encodings strictly
      out-of-fold; no aggregate that peeks at the validation rows; no feature
      built from the future.
- [ ] Check the feature exists **the same way in test** (same categories, same
      ranges) — a feature you cannot compute at inference is useless.
- [ ] Prefer features with a **plausible mechanism**. A feature that helps CV for
      no explainable reason is a leakage suspect until proven otherwise.
- [ ] Look at **past seasons of the same series** for what worked (see the
      solution troves in [arsenal.md](arsenal.md)); Playground series repeat
      patterns.
- [ ] Keep a short note per feature: idea, CV delta, kept/dropped. This is your
      feature lab notebook and it compounds across competitions.

---

## 6. Experiment tracking checklist

Once variants pass a handful, memory fails and you start repeating yourself. The
tools (Weights & Biases, Neptune, a plain `results.csv`) are listed in
[arsenal.md](arsenal.md#experiment-tracking); this is the discipline to apply
whichever you pick.

- [ ] Give every run a **unique id / slug** and a one-line **hypothesis** ("what
      am I testing and what do I expect?").
- [ ] Log, per run: id, features used, model + key params, **CV score**, **LB
      score** (when submitted), and the **seed**.
- [ ] Keep the log in **one place** (a `results.csv` in a private dataset is
      enough) and append to it the moment a run finishes — not "later".
- [ ] Make runs **reproducible**: fixed seed, frozen folds, recorded library
      versions. A result you cannot reproduce is a result you cannot trust.
- [ ] **Change one thing at a time** where you can, so a CV move is attributable
      to a cause.
- [ ] Review the log before each new idea — the fastest experiment is the one
      you realise you already ran.

---

## 7. Error analysis checklist

Looking at *where and why the model is wrong* is the most under-used learning
lever and often the source of the next real gain. Do it after every meaningful
model.

- [ ] Pull the **validation predictions** (OOF) and compute the per-row error /
      loss.
- [ ] Sort by error and **read the worst cases** individually. What do the
      hardest examples have in common?
- [ ] **Slice the metric** by feature values, class, group, and time. A model
      that is fine overall but poor on one slice points straight at the next
      feature or a CV flaw.
- [ ] Distinguish **hard examples** (genuinely ambiguous) from **systematic
      misses** (a pattern the model never learned) — only the second is fixable
      by you.
- [ ] Check the **confusion pattern** (classification) or **residual pattern**
      (regression). Structured residuals mean a missing feature or wrong loss.
- [ ] Turn each finding into a **concrete next step**: a new feature, a
      re-weighting, a CV fix, or a targeted data cleanup — then measure it.
- [ ] Note whether errors correlate with **train↔test shift** (tie back to the
      adversarial-validation result in EDA).

---

## 8. Ensembling checklist

The depth — Hill Climbing over OOF, multi-level stacking, pseudo-labeling — is in
[tabular.md](tabular.md#hill-climbing-ensemble). This checklist is the ordered
practice and the guardrails.

- [ ] **Diversity first.** An ensemble of copies of one notebook gains almost
      nothing. Combine models of *different nature* (e.g. GBDT + TabPFN + a
      neural net) trained on the **same frozen folds** so their OOF align.
- [ ] Save **OOF predictions** for every model (train predictions by fold + test
      predictions) — the ensemble is built on OOF, not on refit guesses.
- [ ] Start simple: a **rank-average** or mean of the best few OOF, scored on CV.
- [ ] Then let **Hill Climbing** pick the subset and weights over OOF (it drops
      harmful members automatically); apply the chosen weights to the test
      predictions.
- [ ] Only go to **multi-level stacking** once simple blending plateaus — it is
      many more fits for a smaller gain.
- [ ] **Guard `BEST_KNOWN`.** Keep the ensemble as an *additional* submission;
      never discard a proven single until the ensemble beats it on the real
      metric.
- [ ] Watch for **CV↔LB divergence** in the ensemble — an ensemble tuned hard on
      CV can overfit the validation just like a single model.
- [ ] If you reuse someone's public OOF for diversity, **attribute the author**
      and respect the notebook's licence.

---

*Related:* competition-type technique files ([tabular.md](tabular.md),
[deep-learning.md](deep-learning.md), [simulation.md](simulation.md),
[code-and-hackathon.md](code-and-hackathon.md)); cross-cutting craft
([grandmaster-playbook.md](grandmaster-playbook.md)); tools and communities
([arsenal.md](arsenal.md)); public learning sources
([resources.md](resources.md)).
