# Cross-domain doctrine and anti-patterns

Synthesised from the mined winning solutions in `gm-methods.md`. Two halves: the principles that hold ACROSS competition types, and the documented ways strong competitors still lose.

**The most valuable field here is `Inverts when`.** A rule you apply everywhere is a rule you will misapply somewhere. Each principle records the regime where it flips sign, because carrying a habit across that boundary is how experienced competitors lose medals — this account did exactly that by bringing fixed-leaderboard submission habits onto a rating ladder.

## Principles (28)


### 1. Exploration cost is set by the competition's scoring format: near-free under a selectable-submission leaderboard, structurally impossible under a continuously-scored live policy, undefined under a judged single-deliverable format.

**Why.** A fixed leaderboard with a submission cap lets you generate many independent candidates and keep the best, bounded only by compute; a live-policy format scores your CURRENT deployed policy continuously, so trying a variant means replacing what's live. Different cost structures, not degrees of the same cost.

**Evidence.** Playground S5E6 1st (~300 pred. sets, private LB 0.38652); Rock Paper Scissors 1st (online Bayesian bandit, no submission selection); Optiver Trading at the Close 1st (best config = overtime at last update, 9h live window).

**Seen in.** tabular-modern, optimization-combinatorial, time-series, code-efficiency

**Inverts when.** A judge-scored hackathon or single-deliverable format with no leaderboard collapses cost to pure development-time opportunity cost against one unscored submission — a third regime. Separately, a runtime-capped efficiency track inverts the 'always ensemble' prior: Feedback Prize Effectiveness Efficiency-track 1st distilled a 90+-model ensemble into ONE deployed model via pseudo-label distillation.


### 2. Reach for the mature, general-purpose, provably-strong tool first (a decades-tuned solver, exact MIP, AutoML, or an already-pretrained model) and spend custom engineering only on the residual it structurally can't cover.

**Why.** Mature tools encode more accumulated refinement (algorithmic tuning or web-scale pretraining) than a contest timeframe can reproduce; the highest-value use of your own time is adapting or constraining the mature tool, not re-deriving its core competence.

**Evidence.** Traveling Santa 2018 1st (team = LKH/Concorde's own authors); Playground S4E8 1st (AutoGluon baseline + L2 stacker over 72 OOFs); NBME 2022, Feedback Prize Effectiveness 2022, US Patent Phrase Matching 2022 (independently converged on deberta-v3-large backbone).

**Seen in.** optimization-combinatorial, tabular-modern, nlp-transformers, graph-molecular

**Inverts when.** Past the tool's design envelope — scale (Concorde failed past ~5,000 nodes; GA-EAX solved 66,000-node in minutes) or constraint topology (LKH can't respect path-dependent geometric constraints; Santa 2022 1st/2nd rejected forcing it) — the default inverts. Also period/domain-conditional: pre-2021 NLP (Jigsaw 2019, Tweet Sentiment 2020, Google QUEST 2020) relied on genuine cross-architecture diversity before one backbone dominated; even post-DeBERTa-v3, US Patent kept a weaker domain-BERT at real ensemble weight (0.4) for diversity.


### 3. When data is anonymized or synthetically constructed, reverse-engineer the true generative structure (real IDs, chronological order) to unlock standard feature engineering — but distrust it as a direct model input.

**Why.** Anonymization pipelines are rarely adversarially hardened; small residual signals (tick size, timestamp proxies, hashed-field fragments) let you invert the transform. Group-by aggregates and lags are only meaningful once rows are correctly grouped/ordered, which anonymization hides but doesn't destroy.

**Evidence.** IEEE-CIS Fraud 1st/2nd (UID reconstruction, AUC 0.9245→0.9377, public LB 0.9485→0.9617); Optiver Realized Vol 1st (tick-size t-SNE chrono order, 360/600 final features, verified vs yfinance); Ventilator Pressure 1st (PID param grid inversion, 66% exact rows).

**Seen in.** tabular-classic, time-series, graph-molecular, meta-gm-craft

**Inverts when.** Reliability doesn't transfer to test the way it verifies on train: Optiver's own author declined to use recovered order as a direct test-time feature ('no guarantee' outside the verified window); IEEE-CIS's UID team flags that lag-stacking predictions on the reconstructed ID produces leakage-shaped CV inflation with zero LB payoff; Ventilator's inversion 'breaks against any noisier/continuous system' and cost 9 CPU-hours outside kernel limits.


### 4. Encoding known invariances directly into features or architecture (explicit distance/graph features, physics-informed decomposition) is the default, sample-efficient design choice for structured scientific/physical data.

**Why.** An architecture with invariance built in doesn't spend learning capacity or data discovering that invariance from examples — every training example teaches only task-relevant variation.

**Evidence.** CHAMPS Molecular Properties 1st (Bosch/BCAI meta-graph Transformer, hand-engineered rotational/translational invariance in attention).

**Seen in.** graph-molecular

**Inverts when.** Given enough data/augmentation/compute, a structurally-blind model can match it — the bitter lesson inside one competition. CHAMPS 2nd ('Quantum Uncertainty') dropped the graph entirely, used raw point-clouds + augmentation, reached a private-LB score (-3.223) competitive with 1st; but the same team's earlier smaller-scale PointNet version of the idea underperformed (-2.28), confirming the win needed real scale, not just the architectural choice.


### 5. When a target's generating process is fully known and either enumerable or a closed-form control law, direct inversion or exhaustive search beats learned approximation outright, often with zero training.

**Why.** A learned model can only approximate a generating function from noisy examples; if the function is identifiable and its parameters fittable directly, that's reconstruction, not approximation — no gap remains for deep learning to close wherever the assumption holds exactly.

**Evidence.** G2Net Continuous Gravitational Waves 1st (zero-ML sinc-kernel template search, public LB 0.825→0.848, 5 GPU-days); Ventilator Pressure 1st (PID inversion, 66% test rows exact zero-error).

**Seen in.** audio-signal, time-series forecasting, graph-molecular

**Inverts when.** Requires an instrumented/simulated process with a knowable closed form — inapplicable to organic, human-generated targets, and brittle to injected noise (Ventilator needed a second ~9 CPU-hour brute-force pass for the noisy remaining 34%). When the signal family isn't fully enumerable — a transient, variable-shape event — deep learning wins instead: G2Net Gravitational Wave Detection (2021, transient chirps) was won by a raw 1D-CNN that surpassed hand-designed CQT/spectrogram frontends, the opposite paradigm within the same host and physics.


### 6. When several independent, imperfect solutions exist for the same problem, recombine them via structural crossover instead of picking one winner or deepening a single search — but only when they share exploitable structural overlap.

**Why.** Any hard non-convex search converges to a local optimum shaped by its own trajectory; when several such optima genuinely overlap in structure, recombination extracts the best sub-structure from each, reaching a combined solution neither run could reach alone.

**Evidence.** Traveling Santa 2018 2nd (Iterative Partial Transcription, shared-endpoint segment swaps); Santa 2022 2nd (same mechanism, ~74076→74075.706541 in <30min); AI Math Olympiad Progress Prize 2 1st (checkpoint weight-interpolation, maj@16 69.1 vs 62.9/66.8 parents, shorter than TIR parent).

**Seen in.** optimization-combinatorial, code-efficiency

**Inverts when.** Santa 2018's own team found naive ILP-recombination of two runs differing by up to 30,000 edges failed outright — 'too different to reconcile.' Recombination requires solutions already close enough to share salvageable structure, a precondition not guaranteed just because multiple runs exist; absent it, population-based search from the start (GA-EAX) or keeping the single best run is correct instead.


### 7. Leak-free cross-validation is a precondition for every method consuming out-of-fold predictions — except the final retrain, which should use 100% of the data despite that retrain being unverifiable.

**Why.** Leakage in the base CV split is amplified by every method built on top of OOF predictions, since they select/weight on the inflated signal. But purging is a data tax; once no decisions remain to protect, paying it on the shipped model wastes signal, and the heuristics approximating full-data convergence are by construction unverifiable — no held-out set remains at 100%-data scale.

**Evidence.** Google QUEST 1st (leaky pseudo-label CV 0.414→0.445 LB disagreed; leak-free 0.414→0.422 transferred); Jane Street 1st (AE-MLP jointly trained, not pretrained, to avoid encoder leakage); Ubiquant 1st (purged CV for decisions, plain KFold for final fit); IEEE-CIS 2nd (time-gapped CV); Playground S5E6 1st (100-seed full-data retrain, MAP@3 0.376→0.380, private LB 0.38652).

**Seen in.** nlp-transformers, tabular-modern, time-series, meta-gm-craft, tabular-classic

**Inverts when.** CPMP's IEEE-CIS 2nd pushed 'reuse derived structure' one step further (lagging per-entity averaged predictions into an L2 model) and it failed — CV rose but LB dropped 0.01; the safe-reuse/leakage boundary moves per-technique and is only locatable via LB, not reasoning. The final-retrain iteration-count heuristic (e.g. +25% boosting rounds for K=5) is itself explicitly 'an approximation, not a validated number.'


### 8. Whether to trust cross-validation over the public leaderboard, or vice versa, is a fresh empirical diagnosis every competition, not a fixed doctrine.

**Why.** CV and public LB are both noisy proxies for the private objective, and which is less noisy depends on competition-specific facts (train/test similarity, drift, LB sample size) that must be checked directly; picking wrong without a safeguard converts a diagnostic error into a submission error.

**Evidence.** Jigsaw Rate Severity 2022 1st (trusted CV over misleading LB, GA-weighted blend for robustness); APTOS 2019 1st (CV never correlated with LB, relied on LB, limited hyperparameter freedom as safeguard); LANL Earthquake 1st (rebuilt train to match adversarial test distribution, verified via KS-test).

**Seen in.** meta-gm-craft, tabular-classic

**Inverts when.** Trusting a CV-LB relation curve at all (Playground S6E2's pick-just-inside-trustworthy-range method) requires a large, stable public LB as an explicit precondition — its own stated pitfall warns that applying it blindly on an unstable-LB competition like LANL or Jigsaw reproduces the exact mistake the opposite camp warns against. The precondition, not the technique, decides the regime.


### 9. Shape both the training loss and any prediction-combination statistic to mirror the evaluation metric's mathematical structure, rather than defaulting to a generic loss or a plain mean.

**Why.** Gradient descent optimizes exactly the loss surface given, and each combination statistic is the population-level minimizer for one specific loss geometry (mean for squared error, median for absolute error, rank-transform for order-sensitive metrics) — a mismatched default fights the objective instead of serving it.

**Evidence.** M5 Forecasting 1st (LightGBM Tweedie objective, zero-inflated demand); Ventilator Pressure (mean→median ensembling, LB 0.157→0.155; discrete rounding →0.153); Jigsaw Unintended Bias 1st (custom mimic loss, subgroup bias-AUC decomposition); SIIM-ISIC Melanoma 2020 1st (rank-averaged differently-calibrated backbones for AUC).

**Seen in.** time-series, nlp-transformers, tabular-classic, time-series forecasting, medical-imaging

**Inverts when.** For rank-based/non-differentiable metrics (Spearman, exact-match, NDCG) there's no way to port the formula into a gradient loss at all — Jigsaw's own writeup falls back to a smooth proxy. Even in the applicable regime, literal ports are unstable (Tweet Sentiment Extraction needed an added smoothing term; Jigsaw's power-means blow up on small subgroups). For proper scoring rules (log-loss, Brier) rank-averaging is actively wrong; median under RMSE/L2 is wrong since mean is optimal there; and the statistic choice needs roughly 5+ models to matter at all.


### 10. Pair or chain structurally-complementary model families so each supplies exactly what the other structurally cannot — but judge the composite only at the full-ensemble level, never component-by-component.

**Why.** Different model families have well-understood structural blind spots — GBDTs split on observed ranges and can't extrapolate, a global linear model can't fit local nonlinearity — so composing along complementary strengths beats forcing one model to implicitly approximate the other's strength; but the correction is fit specifically to the other's residual/logit, so its value is often invisible in isolation.

**Evidence.** Rossmann Store Sales 1st (Ridge trend extrapolation feeds XGBoost, confirmed by academic review of forecasting winners); Walmart Stormy Weather 1st (local baseline + global L1 residual model beat GBDT/RF/SVM ensembles); Playground S5E5 1st (NN-over-LinearRegression residual stack, CV 0.0608→0.0599; XGB-over-NN stage kept despite no standalone CV gain); Elo Merchant 1st (classifier-gated regressor, 0.015 RMSE gain); set_base_margin warm-start credited across two separate Playground wins.

**Seen in.** time-series, tabular-modern, tabular-classic, time-series forecasting

**Inverts when.** Evaluating the composite component-by-component, as any other candidate, causes you to wrongly discard it — the normal 'drop what looks weak alone' filter must be suspended specifically for residual/warm-start pairs, since their value is sometimes visible only at the full-ensemble level.


### 11. When a decision is cheap to score against validation data, search it directly and exhaustively rather than hand-picking — but this is also the most effective way to overfit your own validation set.

**Why.** Manual/uniform choices are one arbitrary point in a large decision space that OOF-based search improves on directly. But every added degree of freedom in the search, not just the model, consumes the validation set's information content, so a large enough search space relative to sample size guarantees you eventually find and lock in noise.

**Evidence.** Rossmann Store Sales 1st (~125,000 pairwise combinations tested against a single 6-week holdout); Playground S5E12 1st (pure OOF hill-climbing plateaued and diverged from LB despite rising CV, forced switch to Ridge-regularized stacking on top-36 ranked OOFs, CV 0.70860 vs 0.70886 but better generalization).

**Seen in.** tabular-modern, time-series

**Inverts when.** The Rossmann winner frames his own search as trustworthy only because a 6-week holdout could support that many pairs 'without overfitting' — inherently harder to sustain as holdout shrinks or search space grows. S5E12 shows the same search family crossing from primary lever to actively misleading once candidate count grew large relative to OOF size, with no bright-line rule for where that crossover happens — unverifiable in advance.


### 12. Ensembling and model-family diversity is the default way to buy score, because decorrelated errors partially cancel under averaging.

**Why.** Independent model families or feature sets make different mistakes on different rows; averaging removes the mistake-specific variance while keeping shared signal, which is why nearly every top-3 team blends rather than ships a single model.

**Evidence.** Otto Group 1st (33 L1 → 3 L2 → L3 blend); CHAMPS top solutions (8-14 model blends); Jane Street/Santander GBDT+NN blends; academic review of 6 Kaggle forecasting competitions: 'Ensembles won all of the competitions.'

**Seen in.** tabular-classic, tabular-modern, nlp-transformers, time-series forecasting, graph-molecular, medical-imaging, audio-signal, cv-segmentation-detection

**Inverts when.** Under a hard wall-clock/memory ceiling (efficiency track, fixed-runtime code competition), marginal accuracy from more models is worthless once cost exceeds the ceiling — the optimum inverts to compressing a strong ensemble into one model. Feedback Prize Efficiency-Prize winner distilled its full ensemble into one deberta-v3-large (0.557 private LB in 5m40s); LMSYS 1st (solo gold) averaged 5 folds' LoRA weights into one merged adapter instead of ensembling at inference, trading accuracy for constant-factor speedup.


### 13. Adversarial validation is a diagnostic for which features differ between train and test, not a prescription for what to do about it — and single-feature screens can't see value that only exists in combination.

**Why.** A high adversarial-AUC feature is definitely distributionally different, but that says nothing about whether the difference is nuisance drift or genuine target-correlated signal — a separate empirical question. And because the screen evaluates features one (or a few) at a time, it structurally cannot detect value that only appears jointly.

**Evidence.** IEEE-CIS Fraud 1st (time-consistency filter dropped ~5% of columns incl. V322-V339 block, ~0.60 train AUC vs ~0.40 later-month); LANL Earthquake 1st (KS-test p>0.05 gate, ended with only 4 features); Playground S6E5 1st (adversarial AUC flagged 'Driver' as most-different, dropping it treated as a CV/LB-verified bet); Optiver 1st (adversarial validation flagged order_count/total_volume as drifting).

**Seen in.** tabular-classic, meta-gm-craft, tabular-modern, time-series forecasting

**Inverts when.** IEEE-CIS states it directly: 'a feature can fail this single-feature test yet still help in combination... an overly aggressive cutoff can discard useful interaction terms' — the same screen that correctly removes drifted noise has no mechanism to distinguish that from a weak-alone, valuable-in-interaction feature. And even an unambiguous, correctly-flagged fix can be low-value: Optiver's rank-transform of its two flagged features produced only a small LB gain.


### 14. A host's data-construction pipeline is forensically detectable via cheap statistics and exploitable for large score gains — but it's a Kaggle-meta skill with zero transfer to real deployment.

**Why.** A host's data-construction code is not adversarially hardened the way production is, so cheap detectors recover real train/test asymmetries unrelated to the underlying task — but real production has no host-constructed synthetic padding or linked original dataset, so the mechanism doesn't exist outside this setting, and a gain confined to one split is the same signature as overfitting the leaderboard.

**Evidence.** Santander Customer Transaction 1st (per-feature uniqueness LB 0.910→0.921 after separating synthetic filler rows); Playground Series wins (linked 'original' dataset reused as extra rows AND as a target-encoding source, both as separate ensemble members); TGS Salt 1st (train/test seismic-mosaic adjacency, labels copied across seam, public LB 0.876→0.884).

**Seen in.** tabular-classic, tabular-modern, meta-gm-craft, cv-segmentation-detection

**Inverts when.** TGS Salt's own mosaic-adjacency trick left private LB completely unchanged (+0.000) despite the +0.008 public gain — the textbook signature of exploiting an artifact that exists only in the probed split, not a generalizing pattern even within the same competition's private set, let alone real deployment. Treating this as a general modeling principle rather than a leaderboard-specific meta-skill is a category error.


### 15. A cheap post-hoc correction exploiting side information the base model can't see can move the score more than further architecture work, at zero retraining cost — but its benefit hinges on an unverifiable estimate.

**Why.** Row-by-row models can't use cross-row structure or protocol/prevalence bias unless it's injected after the fact; the loss function has no way to see it. But because the correction (a group ID, a true deployment distribution) is often an inferred quantity rather than a given, its correctness can't be checked locally.

**Evidence.** IEEE-CIS Fraud 1st (client-consistency post-processing, +0.001 LB on top of validated UID reconstruction); Feedback Prize Effectiveness 1st (mean-recalibration to train-label mean); RSNA Pneumonia Detection 1st (87.5% box-shrink for annotation-protocol bias, LB 0.222→0.260); Rainforest Connection 9th (log-odds rescaling for prevalence mismatch, private LB 0.926→0.963, ~13 places).

**Seen in.** tabular-classic, nlp-transformers, medical-imaging, audio-signal

**Inverts when.** Flips sign outright when the underlying estimate is wrong — an incorrectly-merged group blends unrelated rows' predictions and actively hurts, with no ground truth to check against locally, so a bad correction confidently miscalibrates with no warning sign. The correction is often discoverable only by probing public-LB deltas — a leaderboard-overfitting risk baked into the same mechanism that makes it powerful; even IEEE-CIS's own 'verified' UID reconstruction was outperformed by the model's own implicit clustering.


### 16. The right balance of pooling versus segmentation across a panel/entity hierarchy is set by measured panel heterogeneity and per-branch data volume, not a fixed default in either direction.

**Why.** Pooling lets a model learn shared structure from far more effective samples than any single entity provides, but the benefit is proportional to how much genuinely transferable structure exists across entities; each additional split multiplies model count and starves sub-models of data, so there's an interior optimum, not a monotonic direction.

**Evidence.** Academic review of 4 Kaggle retail-forecasting winners (global cross-learning + ensembling common to all: Rossmann 1,115-store XGBoost family, Web Traffic ~145k-page RNN family); M5 Forecasting 1st (selected hierarchy-cut depth by watching cross-fold variance rise, not just mean improve); M5 3rd (NN team found a different reconciliation hierarchy level worse).

**Seen in.** time-series, time-series forecasting

**Inverts when.** The same review's smallest, lowest-entropy dataset, Walmart Sales in Stormy Weather 1st, inverts the pooling default outright — one simple regularized global linear model beat GBDT/RF/SVM ensembles, with the review noting 'global always wins' isn't universal for small-N, low-entropy panels. M5's own variance-watching heuristic is the only available signal for the interior optimum on the segmentation side; no bright-line rule locates it in advance.


### 17. For metrics that are discontinuous functions of a continuous score, retraining an identical config with many seeds and averaging denoises disproportionately more than it helps smooth regression/classification losses.

**Why.** A ranking metric depends on a discrete, threshold-sensitive event (which items land in the top k), far more sensitive to small stochastic prediction shifts than a smooth scalar loss; averaging many independent noisy estimates of the same probability shrinks exactly that boundary-crossing noise.

**Evidence.** Playground S5E6 1st (100 averaged 5-fold XGBoost reruns, 500 fits, MAP@3 0.376→0.380); AI Math Olympiad Progress Prize 2 1st (12-way self-consistency majority vote under a hard time ceiling); Web Traffic Time Series Forecasting 1st (3-seed × 10-checkpoint, 30-model ensemble + ASGD weight averaging).

**Seen in.** tabular-modern, time-series forecasting

**Inverts when.** Explicitly 'a compute-multiplication strategy, not a modeling insight' — averaging 100 seeds of a mediocre configuration won't out-rank one strong model, so once base-model quality rather than noise is the bottleneck, more repetition stops helping. It also inverts under a hard wall-clock/memory ceiling: AIMO's own pipeline deliberately cuts self-consistency sampling short via early-stopping-on-consensus, trading away noise-reduction because runtime, not boundary noise, was the binding constraint.


### 18. A trained, non-linear meta-model captures real value from individually-weak, diverse base models that a linear blend would zero out — but only when a non-linear combiner sits on top.

**Why.** A non-linear meta-model can learn per-instance 'which base model to trust,' recovering signal from a weak-but-decorrelated learner's error pattern even when its raw score never reveals it; a linear combiner can only globally reweight, so a consistently weak model just gets pushed toward zero or adds noise.

**Evidence.** Playground S5E4 1st (TabPFN kept at CV 13.2, tied-weakest of 12 families vs GBDT's 11.8, inside a winning 75-model stack, private LB 11.44); Otto Group Product Classification 1st (kept Naive Bayes/Sofia/various-k KNN despite weak solo scores); Feedback Prize ELL/Effectiveness 1st (manufactured decorrelated inputs via varied pooling heads and task reformulation, gated on ensemble not solo score).

**Seen in.** tabular-modern, tabular-classic, nlp-transformers

**Inverts when.** Inside a purely linear hill-climbing/Ridge blend, the identical weak-but-diverse model is 'more likely to be zero-weighted or add noise instead,' per the same source for both cases — whether to include a weak model inverts entirely on which combiner sits above it, not on the model's own quality.


### 19. Adding auxiliary prediction heads on cheap correlated-but-unscored metadata regularizes a shared backbone toward meaningful representations at low marginal cost — but isn't safe to apply blindly.

**Why.** A model trained on a single scored objective takes the shortest path to minimizing it, which can exploit spurious shortcuts; forcing the shared representation to also predict correlated signals removes shortcut-only representations and keeps ones that generalize, especially valuable when labeled data is scarce.

**Evidence.** Jigsaw Unintended Bias 1st (auxiliary tasks listed as technique #3 of six); RSNA Breast Cancer 2023 2nd (auxiliary EQL-loss heads for BIRADS/density/view/invasive-status at 0.1x weight, listed under 'what worked').

**Seen in.** nlp-transformers, medical-imaging, graph-molecular

**Inverts when.** A historically strong signal can actively hurt once folded into a larger fusion without checking — CAFA 5 Protein Function 1st found their own Net-KNN component, one of their strongest per prior published work, hurt performance when blindly included; a different Breast Cancer team found pseudo-labeling an absent auxiliary field backfired. The same category of lever helps in one configuration and hurts in a structurally similar one.


### 20. Beam search, or plain greedy construction, is the correct cheap default for discrete sequential-construction problems with small per-step branching, and should be tried before escalating to heavier machinery — group theory, exact MIP, population-based search.

**Why.** Keeping only the top-k partial states by a cheap heuristic score is inexpensive to implement and near-optimal whenever local goodness correlates reasonably with eventual global goodness, which covers a large fraction of discrete construction problems without deeper structural analysis.

**Evidence.** Santa 2023 1st (deprioritized the 'wreath' puzzle family because a short solution could be found via simple beam search); ARC Challenge 1st (DAG-deduplicated depth-3/4 enumeration, a beam-adjacent brute-force default tried before any learned component).

**Seen in.** optimization-combinatorial

**Inverts when.** On deceptive, path-dependent landscapes where a good long-range state looks locally bad, a plain score-only beam permanently discards the seed of the eventual best solution with no completeness guarantee. Santa 2022's two independent top teams both found a naive beam collapses diversity in the dimension that determines long-range feasibility, needing a coarsened DP-feasibility table or invariant-partitioned bucketing before it worked on their harder constraint family.


### 21. How many stacking levels, and how non-linear the top level, is worth building scales with how many genuinely diverse base models you actually have.

**Why.** Each added stacking level needs enough independent training signal (diverse, decorrelated OOF columns) to learn real structure rather than noise. Large teams naturally produce enough base-model diversity to feed a deep stack safely; a small pool gives it too few effectively-independent inputs, so it mostly overfits the OOF set.

**Evidence.** Home Credit Default Risk 1st (3-level stack, 90+ L1 models from a 6-person team feeding a non-linear L2, stated trigger: 'large teams pooling 50+ genuinely diverse models; fewer/similar models mostly add overfitting risk'); Otto Group 1st (33 L1 → 3 non-linear L2 models for row-level regime-switching structure).

**Seen in.** tabular-classic, tabular-modern

**Inverts when.** There is no universal 'stack N levels' rule — only a threshold set by a model pool's REAL, correlation-checked diversity, not its nominal count. A solo competitor mechanically copying a 90-model team's 3-level architecture onto 8 models inverts the intended benefit into pure overfitting risk.


### 22. In adversarial or open-set detection tasks, investing in training-data source diversity dominates investing in model or architecture sophistication — the inverse of the usual priority.

**Why.** No amount of architecture or hyperparameter tuning helps against a generator or attack style your training data never represented at all; coverage of the input space is the binding constraint in an adversarial setting, whereas a fixed-distribution task's input space is already well-sampled by train, so further gains come from fitting it better, not covering more of it.

**Evidence.** LLM - Detect AI Generated Text 1st (verbatim: 'modelling strategies themselves had a lesser impact... as compared to the datamix'; multiple single models scored 0.970+ once the datamix spanned 4 generator-source categories and 7 augmentation types).

**Seen in.** nlp-transformers

**Inverts when.** The same source states the inversion explicitly: 'for competitions with a fixed, well-specified label distribution, architecture and pipeline tricks matter proportionally far more' — over-indexing on data diversification for e.g. a fixed patient-notes NER task is a category error. Whether data diversity or architecture dominates flips on whether the test distribution is adversarially hidden or fixed and knowable — a property of the competition, not the domain.


### 23. Continuing masked-language-model pretraining on unlabeled in-domain text — legitimately including the competition's own unlabeled test text — before fine-tuning closes a vocabulary/style gap cheaply.

**Why.** General-purpose pretraining corpora under-represent domain-specific vocabulary and style; a short continued-MLM phase shifts subword embeddings and attention patterns toward that vocabulary before the labeled objective ever sees it, for a fraction of full-pretraining's cost.

**Evidence.** NBME 1st (MLM continuation on patient_notes.csv excluding train rows, +0.002 CV); NBME 4th (identical MLM step across all 5 backbone/head variants).

**Seen in.** nlp-transformers

**Inverts when.** Only helps when there's meaningfully more in-domain unlabeled text available than the labeled train set alone provides; when the domain is narrow enough that train already covers its vocabulary and style, the extra phase is wasted compute. Jigsaw Multilingual 1st explicitly tried the identical lever and lists it under 'what didn't work,' confirming the failure mode occurs even within the same NLP-transformer domain, not just in principle.


### 24. Freezing a backbone's lower layers and widening the task-specific head is the right lever specifically when the task is shallow or the pretrained representation already matches it closely.

**Why.** Fine-tuning risks catastrophic forgetting and overfitting on small competition datasets; when the task only needs shallow lexical/semantic matching the pretrained representation already encodes, spending capacity on head width instead of backbone adaptation avoids that risk while adding capacity where it's useful.

**Evidence.** US Patent Phrase to Phrase Matching 1st (froze BERT's embedding layer entirely, widened the head instead, reasoning the target was 'simple short words similarity' needing no deep fine-tuning).

**Seen in.** nlp-transformers

**Inverts when.** Caps performance well below a fully fine-tuned model whenever the task needs deep semantic or reasoning adaptation the frozen layers don't already encode — long-document argument evaluation, nuanced writing-quality judgment (CommonLit, Feedback Prize) — where competing top solutions instead invested in full backbone fine-tuning, layer-wise LR decay, and reinitializing top layers. The right choice is set by task depth, not a general preference for parameter-efficiency.


### 25. Whether to invest scarce effort in feature engineering or in model-family diversity is decided by triage at the start of a competition, not by habit — and over-investing in the wrong one, even while local CV keeps improving, actively costs placement.

**Why.** Feature engineering pays off when data has exploitable relational/combinatorial/temporal structure and enough rows that new features don't overfit; on small, noisy, or heavily synthetic data the same effort mostly manufactures spurious signal, so cross-validated model-family diversity captures more real signal per unit effort instead.

**Evidence.** Playground S4E12 1st (combinatorial target/count encoding across ~145,000 candidate column combinations, credited as 'the secret sauce'); Playground S5E3 2nd, same author (single RAPIDS SVC with NO feature engineering matched 2nd place alone, private LB 0.90610; a 3-model no-FE blend would have scored 1st at 0.90728 had it shipped).

**Seen in.** tabular-modern

**Inverts when.** The S5E3 case is the inversion of S4E12 within the same author's own record: adding 3 more feature-engineered models after local CV rose (0.900-0.901) DROPPED private LB (0.90599-0.90604) and cost 1st place — directly contradicting 'more feature/model investment always helps if CV improves.' The correct choice flips on data properties diagnosed fresh each time, not the last competition's winning formula.


### 26. When a hard constraint is expensive to enforce in a solver's main loop but empirically 'almost satisfied' by a relaxed objective, decompose into a relaxed-core solve plus a separate, cheap repair pass.

**Why.** Enforcing an expensive path-dependent constraint on every candidate move inside the main search multiplies per-iteration cost across the whole search; isolating it into a post-hoc repair step that runs once on an already-near-optimal relaxed solution is far cheaper when the gap between relaxed-optimal and fully feasible is genuinely small.

**Evidence.** Santa 2022 1st (soft-penalized GA-EAX core + DP-table-pruned beam-search repair) and 2nd (soft-penalized custom-LKH core + randomized-backtracking repair) independently converged on the same two-stage architecture, within 0.0001% of each other (74075.70654 vs 74075.706541).

**Seen in.** optimization-combinatorial

**Inverts when.** Both teams spent significant separate analytical effort deriving exact necessary feasibility conditions before trusting the constraint really was 'almost free' — the decomposition only pays off if that empirical assumption holds, with no way to know this a priori without roughly the same domain analysis the shortcut is meant to save. If wrong, the relaxed-core solutions are genuinely unrepairable and the whole run is wasted.


### 27. Merging genuinely independently-built solutions decorrelates errors far more effectively than varying hyperparameters within a single shared pipeline.

**Why.** Ensembling's benefit scales with how uncorrelated component errors are; independent design choices produce genuinely different blind spots, whereas hyperparameter variation on one pipeline mostly perturbs the same blind spots by degree.

**Evidence.** ARC Challenge 2nd (merged a width-3 beam-search solver with a teammate's independently-designed solver, near-zero solved-task overlap); IEEE-CIS Fraud (CPMP's late merger under 10 days left, independently-sourced pipeline, lifted the team's blend).

**Seen in.** optimization-combinatorial, meta-gm-craft, tabular-classic

**Inverts when.** Depends on independence being real, not assumed — real independence is something teams can architect for but frequently cannot fully guarantee; ARC's 2nd place explicitly attributes their near-zero overlap to luck. Merging a second copy of an approach derived from the same public kernel or shared method library adds team size and compute without adding signal, a failure mode IEEE-CIS's own late-merger writeup flags directly.


### 28. Under a hard, shared compute ceiling across many sub-tasks or stages, allocate budget adaptively — but a cascade's ceiling is set entirely by its cheapest, earliest stage, which must be tuned for recall, not its own metric.

**Why.** Task difficulty is heterogeneous and evaluation cost grows steeply between stages, so spending the expensive tail only on cheap-filter survivors multiplies throughput; but a candidate lost at the first filter is permanently unrecoverable, so stage 1's job (maximize coverage) is structurally different from every later stage's (maximize precision), and optimizing it by its own metric silently caps the whole pipeline.

**Evidence.** Eedi Mining Misconceptions 1st (3-stage cascade retrieval→14B→32B→72B, +0.023 private LB, retrieval chosen by highest recall not MAP); RSNA Cervical Spine Fracture 1st (cheap segmentation before expensive classifiers, full ensemble fit in 7.5h, ceiling set by segmentation quality); AI Math Olympiad Progress Prize 2 1st (350s+210s shared-buffer time-banking, 560s cap, early-stop on 4-of-5 agreement, 5h/50-question ceiling).

**Seen in.** code-efficiency, optimization-combinatorial, llm-era, medical-imaging

**Inverts when.** Inverts the instinct that every stage should be tuned to its own best metric (a recall-blind stage-1 choice looks locally optimal while discarding unrecoverable cases) and the instinct behind flat ensembling that more diverse cheap models always help (a weak early filter dilutes recall and caps the ceiling regardless of downstream strength). Early-stopping on consensus fails against a confident-but-wrong systematic bias shared across sampled trajectories, since the rule can't distinguish convergence-because-right from convergence-because-biased.


---

## Anti-patterns (19)

Each entry pairs the failure with the CHEAP CHECK that catches it early. Prefer running the check to trusting that you are too careful to need it.


### 1. Treating a small, high-variance public LB split as ground truth

**What happens.** The public leaderboard scores only a fraction of the test set, often on a metric that is itself unstable at that sample size (an ordinal/kappa metric with few positives, a hierarchically-weighted metric like WRMSSE, or an outlier-sensitive metric like R²). Competitors read 3rd-4th-decimal differences on that sliver as real signal, when the gap is well inside the metric's own sampling noise; the private board, scored on the full test set (sometimes even a different time period), then reorders the field by hundreds of places — not because anyone did anything wrong, but because the public number was never precise enough to support the decisions being made on it.

**Where it bit.** Mercedes-Benz Greener Manufacturing (2017) scored on R², a metric extremely sensitive to outliers, and produced a public-to-private reordering large enough that a dedicated public Kaggle dataset exists specifically documenting the shakeup. M5 Forecasting Accuracy (2020) trained/validated mostly at lower series levels while the actual WRMSSE metric weights heavily toward higher hierarchical aggregations; one publicly discussed result moved from public 0.48734 to private 0.62408 while still only placing 190th of 5,558 — i.e. nearly the whole field moved similarly, not just outliers. PetFinder.my Adoption Speed (2019), scored on quadratic weighted kappa, is documented with teams moving from around 13th on the public board to roughly 30th on the private board.

**Cheap check.** Before trusting any public-LB delta, estimate its noise: how many rows/positives is the public split actually scored on, and how much does the metric swing under bootstrap resampling of your own OOF predictions at that same sample size? If a leaderboard-visible gain is smaller than that resampling spread, it is noise, not signal — this is especially severe for ordinal/kappa metrics and hierarchically-weighted metrics.

**Fix.** Size every LB-based decision against the public split's real sample size and the metric's known instability rather than its face value; for hierarchical or ordinal metrics, implement and validate against the exact metric formula locally instead of trusting a training-grain proxy.


### 2. Picking both final submissions by correlated public-LB rank, with no CV-consistency check

**What happens.** Competitors treat their two allowed final submissions as "my two highest public-LB scores." Both picks are usually correlated (same recipe family, tuned the same way), so when the leaderboard reshuffles, both fail together — even though a genuinely better submission (by CV) sits unselected in their own submission history. No diversification against a shake-up ever happens because the selection criterion never included CV agreement in the first place.

**Where it bit.** bestfitting (profiled in 2018 as the world's #1-ranked Kaggle competitor, with wins including Planet: Understanding the Amazon from Space and Cdiscount Image Classification) states the discipline explicitly: always lock one slot to a conservative, fully-understood weighted-average ensemble and at most one to a calculated risk, and never select any submission — however high its public LB — that can't be explained; the same account credits this discipline for surviving 'a wicked leaderboard shake-up' while staying top-5 in Two Sigma Financial Modeling (2016-17), Kaggle's first code competition. Kawamata's 1st-place Playground Series S6E2 (2026) solution independently mapped several real submissions' CV against public LB across the competition and deliberately passed over his numerically highest-ever CV (0.955865) once that curve showed it no longer tracked LB, picking a lower-CV submission instead. The cost of skipping this is visible across many shake-ups: Mercedes-Benz Greener Manufacturing (2017, R² metric, dedicated public dataset documenting the leaderboard shakeup), PetFinder.my Adoption Speed (2019, teams moving from roughly 13th public to 30th private on quadratic weighted kappa), and Cassava Leaf Disease Classification (2020, competitors reporting swings of 400+ places between public and private).

**Cheap check.** Before the deadline, check whether your two selected finals are highly correlated with each other and were both chosen primarily by public rank. Separately, plot CV against public LB across your actual submission history over the course of the competition — a point where the relationship visibly bends or reverses is the signal that further CV-only or LB-only gains are no longer trustworthy.

**Fix.** Reserve one slot for the submission with the most trustworthy CV score (ideally from the region where your CV↔LB curve still agreed) and the other for a genuinely different recipe, not a correlated variant of the same one; never submit a blend you can't explain regardless of its LB rank, and spend some submission budget explicitly mapping the CV-LB curve rather than only chasing the top number on either axis.


### 3. Search or optimization pressure against a fixed validation signal overfits the signal itself

**What happens.** A validation signal — OOF predictions or a holdout set — has only so much capacity to distinguish real generalization from noise. Spending that capacity across many evaluations, whatever form the search takes, eventually fits noise in that specific signal rather than genuine improvement: a continuous blend-weight optimizer with many free parameters (per-model, further split into per-range or per-target sub-weights) fit directly on OOF predictions; hundreds or thousands of candidate configurations, feature subsets, or blend pairs scored against one fixed, modestly-sized holdout; or a greedy hill-climbing ensemble-selection process that keeps adding whichever candidate most improves blended OOF CV score with no penalty for how many candidates were tried. In every case each individual model may be trained perfectly honestly — the selection process itself is what overfits, and the result looks excellent on both CV and even public LB while being measurably worse on private, because no new genuine signal entered the system, only fold-specific noise the search learned to exploit.

**Where it bit.** Allstate Claims Severity 7th place (2016, Gilberto 'Giba' Titericz): a globally-optimized blend (scipy.optimize.minimize/Nelder-Mead against OOF predictions) 'worked very well with CV and public LB,' but after adding a further segmentation of the blend-weight search across prediction ranges, the segment-wise version 'overfitted a little Private LB' relative to the simpler global blend. Rossmann Store Sales 1st place (2015, Gert Jacobusse) ran over 500 randomly feature-subsetted XGBoost models and systematically searched roughly 500×250 candidate pairings against a single validation holdout that was itself only six weeks of data, later stating he was 'surprised' the selected weights held up rather than having overfit that holdout — an outcome he frames as fortunate, not guaranteed. Kaggle Playground Series S5E12 (2025, Chris Deotte, 1st place) reports directly that in that competition 'every further [hill-climbing] addition raised CV but lowered public LB.' The companion Playground S5E3 (2025, 2nd place) case is described as a directly evidenced instance of CV-driven over-ensembling backfiring: three extra models raised CV from 0.898 to 0.900-0.901 but diluted the private LB score.

**Cheap check.** Count the effective free parameters or distinct configurations actually evaluated against your validation signal (OOF rows/folds or holdout) over the course of the competition; if that count is in the hundreds or thousands against a holdout of only weeks or a few thousand rows, treat the final selection as at meaningful risk of holdout-overfitting regardless of how good any individual candidate looked in isolation. If an elaborate segmented/negative-weight blend's edge over a simple global-weight blend disappears or reverses on a genuinely held-out slice, that's the overfitting signature. For hill-climbing specifically: log the CV delta AND an independent check (a held-out slice never used in the search, or public LB) at every step — a step that raises CV while the independent check stays flat or drops is the tell, especially once the candidate pool is in the hundreds.

**Fix.** Prefer the fewest search parameters or candidates that materially help. Where possible, split the search itself across an outer/inner holdout (search on one, confirm on a second untouched one), or explicitly budget how many candidate evaluations the holdout is allowed to absorb before trusting a selection made against it. Stabilize automated ensemble search (e.g. Ridge-weighted hill climbing rather than raw greedy averaging), cap ensemble size, and require agreement on an independently-held check before accepting any addition — treat any optimized blend or selection as a model that itself needs its own validation split, not something safe to fit directly on the same OOF pool used to select what went into it.


### 4. Judging a technique from a single seed/fold run

**What happens.** A feature, architecture tweak, or hyperparameter change is kept or discarded based on one run's CV/LB delta. For high-variance metrics (top-k ranking metrics like MAP@k, small-positive-count metrics, small datasets) the run-to-run noise from just the random seed or fold split can be comparable to or larger than the observed 'improvement,' so the keep/drop decision is effectively being made by chance while looking like a real ablation.

**Where it bit.** Kaggle Playground Series S5E6 (2025, Chris Deotte, 1st place) cites a public-notebook demonstration that averaging 100 separately-seeded 5-fold XGBoost runs (500 model fits total) lifted MAP@3 from an average of 0.376 per single fold-run to 0.380 combined — meaning any single seed's read on 'is this feature good' already carries noise of a magnitude comparable to many real feature-engineering gains reported elsewhere in the same competition family.

**Cheap check.** Before crediting a change, check whether its reported gain exceeds the seed-to-seed/fold-to-fold variance obtained by simply rerunning the identical unchanged configuration multiple times. If the 'improvement' sits inside that noise band, it isn't established.

**Fix.** For high-variance metrics or small datasets, default to multi-seed (and/or multi-fold) averaging as the baseline comparison unit for any keep/drop decision, not a single run.


### 5. CV folds that don't reproduce the real train→test time gap

**What happens.** A time-ordered dataset is split with a plain walk-forward or contiguous scheme (train through month N, validate on month N+1) when the actual test set sits a genuine gap after training data ends. Recency-window features are fresher in a contiguous CV split than they can ever be against the real test set, so CV systematically overstates generalization — gap-induced feature staleness/drift never shows up in validation at all.

**Where it bit.** IEEE-CIS Fraud Detection 2nd place (2019, CPMP with team) built an explicit month-indexed expanding-window scheme with a skipped buffer specifically 'to mimic the fact that there is a significant time gap between train and test,' stating a naive contiguous split 'systematically overstates generalization.' The same purge/embargo logic anchors Jane Street Market Prediction 1st place (2021, 31-group purge gap) and Ubiquant Market Prediction 1st place (2022) — whose own team deliberately downgraded to plain grouped KFold for their FINAL production fit only, after confirming purge/embargo cost too much usable training data, showing even the fix needs a conscious trade-off rather than blanket application. Zillow Prize's two-round structure (2017-2019), which evaluated the final round against real home sales up to a year after model freeze, is a structural example of the same underlying trap at competition-design scale.

**Cheap check.** Compare the time distance between your CV's training cutoff and its validation window start against the actual (or best-estimated) gap between the real training data's end and the real test period's start. A CV gap of zero when the true gap isn't zero is the signature.

**Fix.** Build expanding-window or purged/embargoed CV folds with an explicit skipped buffer matching the true production gap for feature-engineering and model-selection decisions, while consciously deciding whether the final production fit can afford to relax that discipline given the training-data cost.


### 6. CV split at the row level when the real dependency structure is at the group level (batch, entity, or duplicate)

**What happens.** Rows share a natural grouping unit — a real-world entity (patient, customer, essay, question, driver) whose identity correlates with the label independent of the intended features, an exact/near-duplicate row or image, or a technical collection batch (microscopy plate, scanner session, acquisition run) carrying systematic pixel/signal statistics unrelated to the true label. Plain row-level K-Fold lets members of the same group land in both a training fold and its paired validation fold, so the model can partly memorize entity-specific patterns or key off batch-specific artifacts rather than learn the transferable signal. This costs nothing in ordinary CV, where groups typically recur across folds, so local CV looks deceptively strong — right up until deployment against genuinely unseen entities or batches, against which there is no leftover duplicate or artifact to exploit and no identity to have memorized.

**Where it bit.** Recursion Cellular Image Classification (2019) was explicitly structured around the batch-shortcut version of this trap: images were generated in 51 discrete experimental batches, with competitors given only 33 for training and the remaining 18 held out entirely for test — a design whose stated purpose was to force models to separate biological signal from experimental/batch noise. NBME - Score Clinical Patient Notes 1st place (2022) reports the switch 'from 10 folds to GroupStratifiedKFolds [by patient note]... has been a huge [improvement],' implying the ungrouped version was materially misleading. Google QUEST Q&A Labeling 1st place (2020) lists GroupKFold by question/title as literally their first baseline-improving trick. SIIM-ISIC Melanoma Classification (2020): the sticky thread 'True duplicates in this dataset' (65 votes) and the widely-adopted community fix 'Triple Stratified Leak-Free KFold CV' (384 votes) were built specifically because naive K-fold leaked across patients/duplicates, and documented reports around this dataset note that cross-split duplicate images meant 'using a GroupKFold wasn't sufficient' on its own once exact duplicates straddled train and test; Chris Deotte's own competition writeup on this dataset is titled '21st Public - 53rd Private - Trust Your CV,' directly naming the CV-vs-LB gap this data structure produced.

**Cheap check.** For every dataset, explicitly check for any column identifying a real-world entity spanning multiple rows — not just an obvious "patient_id" but derived proxies like device/IP/session — and verify the grouping key is fully contained within one fold, not merely sampled across every fold. Separately, hash rows/images to check for exact or near-duplicates, including any that could straddle the host's own public/private or train/test boundary. Whenever training data was collected in discrete sessions/batches/plates/scanners, check whether your CV folds ever hold out an ENTIRE batch, not just a random sample of rows from every batch — if every batch appears in every fold, your CV cannot detect batch-shortcut learning at all.

**Fix.** Always split by the correct leakage unit (GroupKFold or a multilabel-stratified group variant) rather than by row, and confirm it is leak-free by checking that no entity or duplicate hash appears in more than one fold before trusting any CV-based decision; when pooling in any external dataset, deduplicate across the entire combined pool by content hash, not just by ID field. For batch-structured data specifically, additionally build at least one CV variant that holds out whole batches/sessions to test for shortcut learning, and prefer per-batch normalization, strong augmentation, or domain-adversarial training that discourages the model from keying off batch identity.


### 7. Target/count encoding computed without strict out-of-fold isolation

**What happens.** A categorical column is replaced by the mean/count of the target within that category (or a KNN-based local target mean) computed once over the whole training set rather than fold-by-fold. The feature then partially encodes the label directly; local CV looks excellent because validation rows still see target information smuggled in through their category's global encoding, and the model degrades sharply once deployed on genuinely unseen categories or once the leak is removed.

**Where it bit.** Home Credit Default Risk 1st place (2018, Bojan Tunguz + team)'s KNN local-target-mean feature is documented as needing computation 'strictly out-of-fold (train-fold-only neighbor targets) or it leaks the label into near-duplicate rows.' Kaggle Playground Series S4E12 1st place (2024, Chris Deotte) states 'every TE/CE computation must be nested-fold or it leaks target info directly into features.' Kaggle Playground Series S5E2 1st place (2025, Deotte) explicitly flags target-derived groupby features as 'a textbook leakage vector without nested/out-of-fold logic.'

**Cheap check.** Recompute the same feature two ways — once leaking (fit on all rows including the validation fold) and once strictly out-of-fold — and compare the CV delta. A large CV gain that mostly disappears under strict OOF computation is the signature; a CV gain with no matching public-LB gain is a secondary tell.

**Fix.** Always compute target/count encodings inside the same CV loop used for the model itself (fit on train-fold only, apply to validation-fold), never on the full dataset before splitting.


### 8. Transductive leakage: fitting a scaler, transform, or representation on pooled train+test before the CV split exists

**What happens.** A rank transform, standard scaler, PCA, clustering, autoencoder, or embedding model is fit once on the concatenated train+test dataframe (or otherwise outside the fold loop) before any CV split exists, and only afterward is a supervised model trained with K-fold CV on top of it. No label is involved, so the leak is easy to miss — but every row's transformed value or learned representation has already been shaped by statistics that include validation-fold and test rows, inflating apparent CV generalization because the transform itself has 'seen' the distribution it will later be evaluated against.

**Where it bit.** Porto Seguro's Safe Driver Prediction 1st place (2017, Michael Jahrer): the RankGauss normalization central to the winning NN ensemble is documented with the explicit caveat to 'fit the rank mapping on train only and apply it consistently to test to avoid transductive leakage'; the same solution's denoising-autoencoder representation, trained unsupervised on combined train+test, is flagged as the boundary case — legitimate only when done before any label is touched, and the exact mechanism to audit whenever preprocessing precedes a CV split. Jane Street Market Prediction 1st place (2021) built the autoencoder and downstream MLP as one graph, retrained fully from scratch inside every fold, specifically because 'pre-training the AE once before the CV split... leaks validation-fold info through the encoder and inflates CV' — describing exactly what the public kernels of the era were doing; the resulting single-model AE-MLP scored 6022.202 on the private leaderboard and would independently have placed 1st with no ensembling.

**Cheap check.** Grep your feature-engineering code for any `fit()`, `.mean()`, `.rank()`, or similar statistic computed on a dataframe that includes both train and validation (or train and test) rows before a split — if preprocessing runs before your CV-split code, assume this leak until proven otherwise. Equivalently, check whether any unsupervised preprocessing step (autoencoder, PCA fit, clustering, embedding) was fit on rows that later appear in a validation fold.

**Fix.** Fit every scaler, rank transform, or dimensionality reduction on train-fold data only, then apply (transform, not refit) to validation/test; wrap preprocessing inside the same fold loop as the model, not as a one-time step beforehand. Treat unsupervised pretraining as part of the model, not part of preprocessing — refit it from scratch inside every fold on train-fold rows only, accepting the extra compute cost.


### 9. Pseudo-labeling with a model that saw the fold it's labeling

**What happens.** A team generates pseudo-labels for extra or external data using their full trained ensemble (or any model fit using a given validation fold), then adds those pseudo-labels back into training data touching that same fold. Local CV on that fold rises because the fold's own model effectively re-labeled data using its own partially-memorized judgment, but the private leaderboard doesn't confirm the gain because no new information actually entered the system.

**Where it bit.** Google QUEST Q&A Labeling 1st place (2020, team Bibimorph) diagnosed and quantified this directly: leak-free, fold-consistent pseudo-labeling moved CV 0.414→0.422, while the leakier version (pseudo-labels from an ensemble that included models which had seen the fold) moved the same starting CV to 0.414→0.445 — a number 'the leaderboard did not agree with.' Their own review calls this 'the single most common failure mode reported' in pseudo-labeling across competition writeups.

**Cheap check.** For every pseudo-labeling round, check which fold(s) each pseudo-label-generating model was trained on versus which fold's training data it's being added to — any overlap is the leak. A CV jump from a PL round noticeably larger than the matching public-LB jump is the empirical tell.

**Fix.** Generate K separate pseudo-label sets, one per fold, each produced only by models that never saw that fold — fold-consistent, leak-free pseudo-labeling — even though it costs K times the inference work.


### 10. Fold-mismatched multi-level stacking

**What happens.** A Level-1 pool of models is trained on inconsistent CV folds (different seeds, different splitting logic, or early-stopping that implicitly leaks validation-fold information beyond just the stopping decision). The Level-2 meta-model, trained on the resulting out-of-fold predictions, learns to exploit these small inconsistencies as if they were real signal — inflating CV further while adding nothing, or actively hurting, on the private leaderboard.

**Where it bit.** Otto Group Product Classification 1st place (2015, Titericz & Semenov, 33 L1 models → 3 L2 meta-models → weighted L3 blend) states 'every L1 model must share identical, leak-free CV folds... or L2 learns to exploit leakage instead of learning genuine complementary signal.' Playground Series S6E2 1st place (2026, Kawamata) independently confirms 'fold-matching is easy to violate silently — even Ridge stacking overfits if OOF-generation folds and meta-model folds diverge.'

**Cheap check.** Audit that every Level-1 model and the Level-2 meta-model share the exact same fold-assignment array (same seed, same split object), not merely 'K-fold with K=5' reimplemented per model or notebook. Confirm no model's early-stopping used information beyond just the stopping round.

**Fix.** Instantiate one CV-split object at the very start of the competition and reuse it by reference in every training script; never regenerate folds per model.


### 11. Extending ID reconstruction into causally-impossible lag-stacked features

**What happens.** After legitimately reconstructing a hidden entity ID from anonymized columns to compute simple aggregates (counts, means), a team pushes further and feeds the model's own predictions back through time via that same synthetic ID (a lagged-prediction feature). Because the reconstructed ID is imperfect and the 'lag' partly reflects information only available after the fact within the training/CV setup, this produces a large CV improvement that cannot be realized at real inference time, where a future row's true label isn't available yet.

**Where it bit.** IEEE-CIS Fraud Detection (2019) — both the 1st place team (Konstantin Yakovlev, Chris Deotte, and team) and the 2nd place team (CPMP/Jean-François Puget with team) independently converged on reconstructing anonymized entity IDs for aggregation features, and both independently stopped short of lag-stacking predictions on those IDs; the documented reasoning states that further extension 'is the exact failure mode: leakage-shaped CV inflation with no LB payoff.'

**Cheap check.** For any candidate feature, ask whether it could be computed in a genuine forward-only production/scoring setting with no access to information not yet available at scoring time — including the model's own future outputs. If not, treat any CV gain it produces as suspect regardless of size.

**Fix.** Use reconstructed IDs for simple, causally-available aggregates only (counts, historical means up to the row being scored); do not feed a model's own predictions back through synthetic entity links across time.


### 12. Trusting raw anonymized/ID-like features whose values differ between train and test

**What happens.** Anonymized or hashed categorical columns (card IDs, device IDs) commonly contain many values that appear only in train or only in test, simply because IDs are assigned over time or per-population. A GBDT given the raw value learns spurious per-ID structure from train that cannot transfer to test's mostly-unseen values, producing a model that looks strong in CV (where train-only IDs still recur across folds) and degrades once scored on the real test set.

**Where it bit.** IEEE-CIS Fraud Detection 2nd place (2019, CPMP/Jean-François Puget with team) diagnosed this directly: 'We see that lots of card1 values only appear in test. If we use it directly it will lead to major drop' — the fix (frequency-encoding instead of the raw ID) took his solo, feature-selected model to 0.942 public LB before UID-based features were even added.

**Cheap check.** For every candidate feature, especially anonymized/hashed/ID-shaped columns, plot its value-frequency distribution in train overlaid against test, or run adversarial validation across all features at once rather than eyeballing a handful. A feature where a large share of test's mass sits on values rare-or-absent in train is a red flag regardless of how much it helps CV.

**Fix.** Replace raw high-cardinality/ID-like categoricals with frequency encodings, counts, or other transforms whose train/test distributions actually overlap; formalize the screening with adversarial validation once there are more than a handful of suspect columns — but confirm any drop actually helps CV/LB rather than dropping reflexively, since a drifting feature isn't automatically useless.


### 13. Optimizing a proxy loss instead of the real hierarchical/ordinal/weighted competition metric

**What happens.** Teams train against a convenient standard loss (plain BCE, RMSE, accuracy) and only check the actual competition metric at evaluation time. When the real metric has structure the proxy doesn't respect — an ordinal weighting (quadratic weighted kappa), a hierarchical aggregation with different weights than the training grain (WRMSSE), or a decomposed fairness/subgroup formula — the proxy-optimized model systematically underperforms what the metric actually rewards, and any threshold/rounding tuned against the proxy overfits the wrong decision surface.

**Where it bit.** M5 Forecasting Accuracy (2020) trained largely at lower series levels while WRMSSE evaluation weights heavily toward higher hierarchical aggregations, a structural mismatch widely cited as a major driver of that competition's public/private divergence. PetFinder.my Adoption Speed (2019), scored on quadratic weighted kappa, saw threshold/rounding choices tuned against local CV fail to hold on the private board, part of the documented public-to-private movement. Jigsaw Unintended Bias in Toxicity Classification 1st place (2019, ods.ai) instead implemented the metric's exact subgroup/BPSN/BNSP decomposition as the training loss itself, specifically to avoid this mismatch — that this was worth building custom loss code for is itself evidence of how much a generic proxy loss leaves on the table.

**Cheap check.** Before trusting any CV number, implement the exact competition metric (not an approximation) locally and confirm it reproduces known public-LB scores from your own submission history. Check explicitly whether the metric decomposes hierarchically or ordinally in a way your training loss doesn't mirror.

**Fix.** Where feasible, build a loss/objective that structurally mirrors the real metric's formula; always validate any threshold/rounding search on genuinely held-out folds distinct from the data used to fit those thresholds.


### 14. Host-side data-generation or split artifact lets a shortcut reconstruct the label with little real modeling

**What happens.** Something about how the host assembled or split the data — file metadata, row ordering, duplicated historical rows recurring as shifted columns, or (most commonly) a test set built by randomly holding out rows from an otherwise grouped or time-ordered dataset instead of holding out whole groups or future periods — lets a trivial, content-blind feature or a simple groupby aggregation reconstruct or closely approximate the label. Competitors who find it first gain a massive, largely non-modeling edge; the competition effectively collapses into a race to discover and copy the shortcut rather than a genuine modeling contest, and a change that exploits the artifact can look like a real modeling win on the leaderboard while being nothing of the sort.

**Where it bit.** Draper Satellite Image Chronology (2016) had a leak where raw image file size alone, with no image content used, reached a public LB score around 0.30. Santander Value Prediction Challenge (2018) had a documented leak where roughly 16% of test rows (7,897 of 49,342) could be assigned their exact target value with full confidence, because a customer's historical values recurred as shifted columns elsewhere in the row structure. Predicting Red Hat Business Value (2016): the thread '~0.987 Kernel now available - seems like leakage' (59 votes) showed that a single groupby-mean-by-date-and-group_1 trick reached ~0.987 AUC — close to the eventual winning score — because the host's random-sample-from-time-series split methodology let each date/group's individual outcome leak; it was voluntarily published so the competition would be 'more about actual prediction improvement, and not clever tricks based on imperfect train/test split methodology,' with a companion thread arguing 'Kaggle should seriously think if they formally allow probing + hand labeling.' TGS Salt Identification Challenge 1st place (2018) documents the cautionary flip side directly: a mosaic-tile-adjacency exploit moved their public LB by +0.008 but their private LB by +0.000, and their own writeup flags that exact public-only divergence as 'diagnostic of leak exploitation... not a modeling win worth trusting.'

**Cheap check.** Sanity-check your own model's score against the dumbest possible content-blind baseline — file metadata, row index, a suspicious ID, or a groupby/aggregation on visible categorical+date columns alone. If a trivial baseline reaches a near-ceiling or suspiciously high score, the competition's real signal is the generation/split artifact, not your features. Separately, treat any change that moves public LB but not a genuinely held-out/CV estimate (or vice versa) as diagnostic of exploiting something that won't hold, not a win to build further strategy on.

**Fix.** Verify any suspiciously strong feature's contribution against a held-out split with no possible exposure to the artifact before trusting it. When a leak surfaces, treat leak-derived features as inputs the model can learn to weight rather than a hand-coded override, so your submission degrades gracefully if the host patches the split or filters the target before final scoring — and escalate it to the host rather than keeping it private. Check explicitly whether the private test was generated by the same process, since some leaks persist to private and most evaporate or get patched.


### 15. Leaderboard probing as a label/answer extraction oracle

**What happens.** On metrics or answer formats that let a single submission's returned score be inverted into ground-truth information — either algebraically (a per-feature-decomposable metric like AUC over independent, standardized input variables) or by enumeration (a small, discrete, linked answer space where correct answers repeat across public/private groups) — a competitor can extract real label information through the scoring function itself rather than through modeling. With enough targeted submissions (one or a few per candidate variable/group), a strong linear model or a full answer key can be reconstructed almost entirely from leaderboard feedback rather than from a small labeled training set, especially when the legitimate training set is too small to compete on its own.

**Where it bit.** Don't Overfit II (2019, 250 training rows, ~19,750 test rows, AUC metric — deliberately built to teach this): Chris Deotte's thread 'The Mathematics of AUC/LB-Probing' (47 votes) derives the exact inversion formula (a_k = 1/sqrt(2-2*AUC) - 1 for AUC near 0.5-0.7) and cites a public kernel using it that scored 0.890 (his own 2nd place); the competition's 1st-place writeup is framed directly around this: 'Careful leaderboard probing was the key,' and a 4th-place team explicitly branded their approach '[no LB probing]' to distinguish it from the norm. Draper Satellite Image Chronology (2016, 274 image-ordering sets, 60-day competition window): the forum thread 'Leak in the dataset' describes manually solving several sets by hand ('manually solving 4-5 sets a day is more than possible, it's inviting') and then using linked/overlapping answer-groups plus the public LB score itself to binary-search or confirm guesses across the rest — serious community discussion of early closure followed ('Should Draper close the competition?'), and the eventual 1st-place writeup was titled 'How to win the competition if you know nothing about image processing.'

**Cheap check.** Compare your submission budget (and the information recoverable per submission) against the entropy of what would need to be recovered — the size of the hidden label set, or the number of distinguishable answers/orderings/permutations per group. If (available submissions) × (information per submission) approaches that entropy, or if a human could plausibly hand-solve or brute-force a meaningful fraction of the answer space within the submission limit, the public leaderboard is not a trustworthy skill signal for anyone — check forums for public kernels already extracting per-feature LB scores as coefficients.

**Fix.** When this regime is recognized, treat leaderboard rank as compromised and prioritize a validation scheme that owes nothing to submission feedback. As a host, cap submissions, avoid metrics that linearly decompose across a small/independent feature or example set a modest submission budget can fully probe, and shrink the format itself (larger/less-overlapping answer space, tighter submission caps, held-out-only scoring) before or during launch.


### 16. Manually-solved per-row special cases substituted into predictions without validating their scope

**What happens.** A competitor obtains high-confidence values for a SUBSET of rows or entities — by re-identifying anonymized real-world entities via external web search, by matching rows through a discovered data-generation leak, or by reverse-engineering which specific rows are extreme outliers through repeated public-LB score deltas per submitted tweak — and substitutes those values directly into predictions, or tunes how aggressively to trust them, especially against the public LB. The failure is treating that subset as uniformly reliable and representative: some 'confirmed' matches turn out poisoned, filtered, or out-of-range; true outlier/entity membership differs between the public and private test splits; or the solved subset is too small a fraction of the total test set to justify the leaderboard position it produced. Because the special-casing was tuned against (or trusted because of) the public split specifically, the position it bought inverts hard once private is revealed.

**Where it bit.** ASHRAE - Great Energy Predictor III (2019): the thread 'Sites, buildings identified by internet search' (106 votes) identified real universities/buildings (site 0 = University of Central Florida, etc.) via public building-energy PDFs; the team that led the PUBLIC leaderboard finished 497th PRIVATE, named directly in the top writeup '[497th place shake down solution] PUBLIC LB 1st place solution' (104 votes) — whose author used the leak only conservatively for out-of-fold validation on confirmed sites and stated 'no other methods to exploit leakage,' implying the public #1 team over-exploited it. Santander Value Prediction Challenge (2018): ianlini's thread 'Dropped from 2nd to 3566th' (73 votes) found 50 highly-confident leaked rows, trusted them wholesale, and fell from a projected top-3 finish to private LB 1.62 (rank 3566) — called out in-thread as likely 'the biggest LB drop... I've seen them in 2000s before but not in 3000s.' Giba (eventual co-winner) explained the mechanism directly in-thread: the host's target field was filtered to a fixed numeric range (30000-40000000), and out-of-range leaked matches were poisoned/unscored; removing only the zero-valued leaked rows recovered the score to 0.62, versus 0.53 for his leak-free models. Elo Merchant Category Recommendation (2019): raddar's thread 'Shakeup incoming!' (60 votes) documents that 'having a single override can make a 0.003-0.004 score difference... current public LB gold position... are within 3 overrides range,' and that the resulting shakeup dropped the public leaderboard leader 1,412 places and the public 2nd-place team down to 2,460th.

**Cheap check.** Validate every leaked/matched/re-identified value against an independent plausibility signal (does it fall inside the observed target range? does removing a subset of matches improve or hurt held-out CV?) before trusting it fully — a leak or match discovery should raise, not lower, your scrutiny. If any part of a score improvement depends on external, non-provided sources or matches tied to specific rows/entities, check what fraction of the total test set that actually covers — a signal touching only a handful of identifiable rows cannot support a leaderboard position that assumes it generalizes. Separately, check whether a public-LB jump came from a broad feature/model change or from a small number of manually-flagged special-case rows; if removing any single override changes your score by more than the medal-cutoff gap, your rank is noise, not signal.

**Fix.** Treat any manually-solved, leaked, or re-identified value as one noisy, confidence-gated blend component — never a wholesale substitute for the modeled prediction — and always keep a leak-free model as the fallback backbone, blending rather than overriding. Where an outlier or special-case subpopulation is large enough to matter, model it as its own gated classifier trained and validated with proper CV rather than hand-tuning individual rows against the public score; trust local CV stability, not per-row LB deltas. Report re-identification leaks to the host immediately rather than quietly exploiting them, and cap any leak-derived component's influence in the final blend.


### 17. Code-competition resource ceilings silently invalidating an otherwise-good pipeline

**What happens.** In kernel/notebook-only code competitions, a hard wall-clock, memory, or no-internet ceiling applies to the actual (often larger or differently-shaped) hidden test set, not to whatever smaller sample was used during local development. A pipeline that runs fine locally can time out, run out of memory, or fail to load a needed package silently during the real scored run, producing a failed or truncated submission despite strong offline metrics.

**Where it bit.** Mercari Price Suggestion Challenge (2018), Kaggle's first kernels-only competition, enforced a 60-minute runtime ceiling on 4 vCPUs with no GPU and a 16GB memory cap for the scored run against the full test set, constraints substantial enough to rule out heavier modeling approaches outright and require batch-wise rather than full in-memory inference. Later code competitions generalized the same trap: Feedback Prize - Predicting Effective Arguments' Efficiency track (2022) scored explicitly against a 32,400-second (9-hour) ceiling, and ARC Prize 2024's winning pipeline had to fit within a shared 12-hour budget across roughly 100 tasks, with the writeup noting that a few slow-converging tasks can silently starve the rest of the budget if not capped per-task.

**Cheap check.** Before relying on any code-competition score, confirm your pipeline's measured runtime/memory on a dataset sized like the ACTUAL hidden test set, not your local dev sample, including cold-start costs (model loading, package install from an offline-staged wheel) — and check that offline-staged package wheels actually match the scoring environment's CUDA/Python/glibc versions, since those can silently drift between when the dataset was built and when the notebook reruns.

**Fix.** Profile end-to-end against a full-size synthetic or held-out set early, build in a hard per-item/per-stage time budget with graceful degradation rather than hoping the ceiling is never hit, and re-verify offline dependency wheels against the current scoring environment shortly before the deadline.


### 18. Organized private-sharing / medal-selling fraud rings

**What happens.** Paid coaching operations, advertised openly on third-party marketplaces, hand customers ready-made models or submission files for kernel-only or CSV-submission competitions respectively, guaranteeing a purchased medal — a direct violation of no-private-sharing rules that clutters the leaderboard with fraudulent entries and can put implausibly novice-looking accounts in gold/silver range, sometimes via networks of dozens of accounts run by one operator across multiple competitions.

**Where it bit.** SIIM-ISIC Melanoma Classification, 2020. Thread 'Evidence regarding private sharing' (343 votes, verified via Kaggle API): documented a Taobao storefront selling a guaranteed 1% finish (silver medal) for roughly $600, screenshots of an operation boasting '38 (!!) silver medals within one competition,' and named Kaggle Masters whose accounts appeared in the advertisements while independently sitting at public-LB rank 18; the same operator's name was tied to '20 different accounts in the deepfake competition' (Deepfake Detection Challenge). The thread's author, a Grandmaster, added a first-person cautionary tale: 'I might have blood on my hands... I teamed up with someone that seemed very trust-worthy... based on previous competition results & current LB standing. After teaming up, this person went completely silent... We also suspect the profile picture used to be fake.'

**Cheap check.** Treat a prospective merge partner's displayed rank/medal history as necessary but not sufficient evidence of real skill; a cluster of low-history accounts sitting anomalously high near medal cutoffs, or a partner who goes silent right after merging, are both flaggable patterns.

**Fix.** Verify a merge candidate's actual technical contribution (their code, their approach, genuine forum engagement) before merging rather than merging on LB position alone; report suspicious clustering to the host/Kaggle for cross-checking against the Meta Kaggle dataset before prizes are finalized.


### 19. Exfiltrated test labels disguised as external data

**What happens.** A team obtains true private-test answers through illegitimate means (e.g., scraping a host's public-facing website) and hides that information inside a legitimately-permitted external dataset — encoding, obfuscating, and hashing the answers into an ID field disguised as unrelated content — then decodes them at inference time under deliberately unreadable, deeply nested processing code, using only a subset of the recovered answers to keep the final score plausible enough to avoid suspicion.

**Where it bit.** PetFinder.my Adoption Prediction, 2019. Official host announcement 'PetFinder.my Contest: 1st Place Winner Disqualified' (321 votes, verified via Kaggle API): the winning 'Bestpetting' team, including a Kaggle Grandmaster, 'fraudulently obtained adoption speed answers for the private test data (possibly by scraping our website)... encoded, obfuscated and hashed into an ID field that was disguised as part of their external "cute-cats-and-dogs-from-pixabaycom" dataset... processing codes were meticulously hidden and obfuscated under many nested layers of functions and codes.' The scheme was caught by a fellow top finisher (Benjamin Minixhofer) after the competition closed; the Grandmaster was permanently banned and prize money was later refunded.

**Cheap check.** When a solution's external-data justification doesn't obviously explain the size of its score jump, or its data-processing code is unusually obfuscated relative to its stated purpose, treat that as a red flag worth independent review — exactly what caught this case.

**Fix.** Keep external-data provenance and processing fully transparent and reproducible as a competitor; as a host, restrict or manually audit external-dataset claims that could plausibly encode host-website content, and budget time after the deadline for forensic code review of top finishers before finalizing prizes.

