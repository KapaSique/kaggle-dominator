# Cross-domain doctrine and anti-patterns

Synthesised from the mined winning solutions in `gm-methods.md`. Two halves: the principles that hold ACROSS competition types, and the documented ways strong competitors still lose.

**The most valuable field here is `Inverts when`.** A rule you apply everywhere is a rule you will misapply somewhere. Each principle records the regime where it flips sign, because carrying a habit across that boundary is how experienced competitors lose medals — this account did exactly that by bringing fixed-leaderboard submission habits onto a rating ladder.

## Principles (44)


### 1. The cost of exploring many candidate solutions is set by the competition's scoring format, not by your compute budget — the identical strategy (train many variants, keep only the best) is nearly free under a fixed leaderboard with selectable final submissions, structurally impossible under a continuously-scored live policy, and undefined under a single-deliverable judged format with no interim numeric signal.

**Why.** A fixed numeric leaderboard with a submission cap lets you generate many independent candidates and simply keep whichever scored best, so parallel search is bounded only by wall-clock/compute cost. A rating-ladder or live-policy format instead scores your CURRENT deployed policy continuously, so 'trying a variant' means replacing what's live — a bad update directly displaces a good one rather than sitting unselected alongside it. These are different cost structures, not different degrees of the same cost.

**Evidence.** Playground Series wins routinely retrain 100+ seeded variants and average/select at negligible marginal risk (S5E6: ~300 total prediction sets feeding the final blend, private LB 0.38652, 1st place). By contrast the Rock Paper Scissors 1st place runs one continuously-scored live agent whose action each round is chosen by an online Bayesian bandit over a fixed opponent pool — there is no 'keep the best of many submissions,' only online adaptation of a single active policy. Optiver Trading at the Close 1st place ran periodic online retraining inside a live 9-hour scored window and reports their best configuration 'is overtime at last update' — the exploration itself consumed the scored budget.

**Seen in.** tabular-modern, optimization-combinatorial, time-series, code-efficiency

**Inverts when.** A judge-scored hackathon or single-deliverable competition with no leaderboard inverts both regimes at once: with no interim numeric signal to search against, neither 'free parallel search' nor 'displacement cost' applies — cost collapses to pure development-time opportunity cost against one unscored-until-judged submission, a third regime. Separately, a runtime-capped code-competition efficiency track inverts the near-universal 'always ensemble' prior directly: Feedback Prize Effectiveness's Efficiency-track 1st place distilled a 90+-model-equivalent ensemble into ONE deployed model via pseudo-label distillation, because the format made deploying the real ensemble infeasible regardless of its offline accuracy edge.


### 2. Reach for the mature, general-purpose, provably-strong method first -- a decades-tuned solver (LKH, Concorde), an exact MIP formulation, or an AutoML baseline (AutoGluon) -- and spend custom engineering only on the residual the general method structurally can't cover.

**Why.** These tools encode enormous accumulated algorithmic knowledge that would be prohibitively expensive to rediscover inside a competition's time budget; the marginal, competition-specific value-add is almost always in the narrow gap the general tool wasn't built to handle, not in beating the tool at its own core job.

**Evidence.** Traveling Santa 2018's actual 1st-place team was LKH's and Concorde's own authors; Playground S4E8 1st place used AutoGluon both as a turnkey baseline and as an automated non-linear Level-2 stacker over 72 hand-built OOFs.

**Seen in.** optimization-combinatorial, tabular-modern

**Inverts when.** Past the general method's design envelope -- either in raw scale (Concorde was impractical past ~5,000 nodes while GA-EAX solved a 66,000-node instance in minutes) or in constraint topology (LKH's move operators can't cleanly respect a path-dependent geometric constraint like robot-arm reachability) -- the 'start with the mature solver' default becomes the wrong first move. Santa 2022's 1st- and 2nd-place teams both explicitly tried and rejected forcing LKH to respect their constraint directly, building a soft-penalty population-based search plus a repair pass instead; Santa's Workshop Tour 2019's 1st place went the opposite direction (an exact, LP-bound-pruned MIP) because that competition's coupling constraint WAS cleanly linearizable, where an unpruned direct MIP had stalled 24 hours at <=70,913 before LP-relaxation pruning reached a feasible <=70,134 in about 70 minutes.


### 3. When data is anonymized, shuffled, or synthetically constructed, look for artifacts that let you reverse-engineer the true generative structure (real entity IDs, true chronological order, the physical/control-system formula) — this often dominates purely statistical feature engineering built on top of the anonymized view, but the recovered structure is frequently too brittle to trust as a direct model input.

**Why.** Anonymization pipelines built for a competition are rarely adversarially hardened; small residual signals (tick size revealing true price scale, timestamp proxies revealing entity identity, digit rounding revealing physical parameters) let you invert the transform and recover ground-truth structure a purely statistical model would otherwise approximate indirectly.

**Evidence.** IEEE-CIS Fraud Detection 1st place reconstructed hidden client/card UIDs from anonymized fields, raising local validation AUC 0.9245→0.9377 and public LB 0.9485→0.9617 on an otherwise-identical feature set. Optiver Realized Volatility 1st place recovered true chronological time-id order via tick-size-derived price reconstruction plus t-SNE, verified against real yfinance history for one anchor stock, then built 360 of ~600 final features from it. Ventilator Pressure Prediction 1st place brute-force-inverted the generating PID controller's exact 20x6 discrete parameter grid, perfectly predicting 66% of test rows outright.

**Seen in.** tabular-classic, time-series, graph-molecular

**Inverts when.** The recovered structure is often too unreliable to use directly even by its own discoverers — Optiver's 1st place used the reconstructed order only for CV and neighbor-grouping, explicitly declining to feed it as a literal test-time feature because 'I don't think we can use future information in a real Optiver scenario' and reconstruction quality wasn't guaranteed outside the verified window. The Ventilator PID inversion likewise 'breaks against any noisier/continuous system' and cost over 9 CPU-hours outside Kaggle's kernel infrastructure — real but brittle, the opposite of a generalizable technique.


### 4. Encoding known invariances and structure directly into features or architecture (explicit distance/graph features enforcing rotational and translational invariance, physics-informed decomposition) is the default, sample-efficient design choice for structured scientific/physical data.

**Why.** An architecture with invariance built in doesn't spend learning capacity or training data discovering that invariance from examples -- every training example teaches it strictly about task-relevant variation, not about symmetries it should have known for free.

**Evidence.** CHAMPS Predicting Molecular Properties 1st place (Bosch/BCAI) built a meta-graph Transformer with explicit distance-decay attention over atom-pairs, hand-engineering rotational/translational invariance directly into the attention mechanism.

**Seen in.** graph-molecular

**Inverts when.** Given enough data, augmentation, and compute, a structurally-blind model that never encodes the invariance at all can match or exceed the structure-aware design -- direct evidence of the bitter lesson inside a single competition. CHAMPS's 2nd place ('Quantum Uncertainty') dropped the molecular graph entirely, feeding raw (x,y,z, atom-type) point clouds into a plain Transformer with no positional encoding and no built-in invariance, relying purely on rotation/translation augmentation and ensemble scale to reach a final private-LB ensemble score (-3.223) fully competitive with the explicitly structure-encoding 1st-place design -- the same team's own earlier, smaller-scale PointNet-style version of the identical idea had badly underperformed (-2.28 LB), confirming the win required real data/compute scale, not just the architectural choice.


### 5. When a target signal's full generative or statistical model is exactly known and its parameter space is small enough to enumerate (a periodic waveform's frequency/phase family, a matched-filter template bank), a classical closed-form or exhaustive-search solution is optimal and beats deep learning outright, with zero training needed.

**Why.** A matched filter, or equivalent exhaustive template search, is the information-theoretically correct detector for a fully-specified signal shape in known noise; a learned model can only approximate that optimum from finite noisy examples, so once the exact detector is computationally tractable, there's no approximation gap left for a neural network to close.

**Evidence.** G2Net Detecting Continuous Gravitational Waves 1st place (Jun Koda) won with zero machine learning -- an exhaustive 360x241-candidate sinc-kernel-refined template search over the known Doppler-shift physics, moving public LB 0.825->0.848 purely from search refinement, at a cost of 5 GPU-days for the base search.

**Seen in.** audio-signal

**Inverts when.** The instant the signal family isn't fully specified or enumerable -- a transient, variable-shape event rather than a clean periodic template -- the classical approach loses its optimality guarantee and deep learning wins instead, evidenced by a different G2Net competition on the same underlying physics: G2Net Gravitational Wave Detection (2021, transient binary-merger chirps) was won with a raw 1D-CNN over the waveform, where hand-designed time-frequency frontends (CQT, then improved spectrograms) were each explicitly surpassed by letting the network learn the representation directly -- the opposite paradigm winning under the opposite signal-specification regime, within the same host and physical domain.


### 6. Before writing bespoke code or training a new model from scratch, benchmark whatever mature, heavily-optimized general-purpose tool already exists for the shape of your problem — a decades-tuned classical solver or an already-pretrained large model — and spend your own engineering only on what it doesn't already know.

**Why.** Mature general tools encode more accumulated refinement (years of algorithmic tuning, or web-scale pretraining) than a contest timeframe can reproduce from scratch; the highest-value use of your own time is adapting, constraining, or feeding the mature tool for your specific problem, not re-deriving its core competence.

**Evidence.** Traveling Santa 2018's actual 1st-place team WAS the authors of LKH and Concorde themselves, running their own tools with light custom scripting around them. NBME 2022, Feedback Prize Effectiveness 2022, and US Patent Phrase Matching 2022 all independently converged on deberta-v3-large as backbone, spending engineering budget on pooling heads, adversarial weight perturbation, and discriminative learning rates instead of architecture search.

**Seen in.** optimization-combinatorial, nlp-transformers, graph-molecular

**Inverts when.** This dominance is period- and domain-conditional, not universal. Pre-2021 NLP competitions (Jigsaw 2019, Tweet Sentiment Extraction 2020, Google QUEST 2020) relied on genuine cross-architecture diversity as a first-class ensembling lever precisely because no single backbone dominated yet; and even within the DeBERTa-v3 era, a domain-pretrained BERT stayed in US Patent Phrase Matching's winning ensemble at real weight (0.4) 'due to better diversity' despite lower solo CV (8451 vs 8627) than DeBERTa-v3. The principle inverts precisely in the window, or domain corner, before one tool's dominance is established — not always knowable without testing both.


### 7. When several independent, imperfect search runs or trained models exist for the same underlying problem, recombine them via structural crossover (tour-merging, genetic edge-assembly, checkpoint weight-interpolation) instead of picking one winner or deepening a single search — but only when the independent solutions share exploitable structural overlap.

**Why.** Any hard non-convex search (TSP local search, SGD training) converges to a local optimum shaped by its own trajectory; when several such optima genuinely overlap in structure, recombination extracts the best sub-structure from each, reaching a combined solution neither run could reach alone, at far less cost than continuing to search from one point.

**Evidence.** Traveling Santa 2018 2nd place's Iterative Partial Transcription merged independently-optimized tours by swapping in better-scoring shared-endpoint segments; Santa 2022 2nd place used the same mechanism on a bank of independently-accumulated tours to go from a 'liftable' ~74076 to the essentially-optimal 74075.706541 in under 30 minutes. AI Mathematical Olympiad Progress Prize 2 1st place linearly interpolated a long-chain-of-thought checkpoint with a tool-integrated-reasoning checkpoint of the same base model; the merge beat both parents on accuracy (maj@16 69.1 vs 62.9 / 66.8) while running shorter than the TIR parent (12489 vs 15834 average tokens).

**Seen in.** optimization-combinatorial, code-efficiency

**Inverts when.** The Santa 2018 team's own account gives the exact failure mode: a naive ILP-based recombination of two runs differing by up to 30,000 edges failed outright because they were 'too different to reconcile.' Recombination requires the independent solutions to already be close enough to share salvageable structure — a precondition that isn't guaranteed just because multiple independent runs exist; absent it, the correct move inverts to population-based search from the start (GA-EAX) or simply keeping the single best run.


### 8. Purge information leakage at every pipeline stage where a model or statistic could see data ahead of when it's allowed to — except at the very last step, where deliberately retraining on the CV-held-out folds for the final shipped model is correct despite losing the ability to validate that exact retrain.

**Why.** Leakage silently inflates whatever internal metric drives decisions, and because the inflation is invisible in the number itself it survives until an honestly-held-out check fails to confirm it. But purging is itself a data tax that shrinks every fold's training set; paying it forever, including on the model actually shipped, wastes real signal once no further decisions remain to protect.

**Evidence.** Google QUEST 1st place's own diagnosis: fold-inconsistent pseudo-labeling inflated CV 0.414→0.445 while 'the leaderboard did not agree with' that number; fold-consistent pseudo-labeling gave a smaller but real 0.414→0.422 that did transfer. Jane Street Market Prediction 1st place retrained a full autoencoder+MLP end-to-end inside every CV fold specifically because pretraining once beforehand let the encoder see validation-fold data. Ubiquant Market Prediction 1st place used strict purged/grouped time-series CV for every feature and hyperparameter decision, then explicitly relaxed to plain KFold for the FINAL production fit once no further decisions remained.

**Seen in.** nlp-transformers, tabular-modern, time-series

**Inverts when.** CPMP's IEEE-CIS Fraud Detection team (2nd place) pushed the same 'reuse derived structure' idea one step further — lagging per-entity averaged PREDICTIONS into a second-level model — and it failed outright: 'CV skyrocketed, but LB dropped by 0.01.' The inversion isn't whether to purge but WHERE the safe-reuse/leakage boundary sits; it moved on this specific extension, and the team could only locate it by testing against LB, not by reasoning about it in advance.


### 9. Whether to trust cross-validation over the public leaderboard (or vice versa) is a fresh empirical diagnosis every competition, not a fixed doctrine — the same grandmaster correctly makes opposite calls in different competitions, each paired with a compensating safeguard.

**Why.** CV and public LB are both noisy proxies for the private objective, and which is LESS noisy depends on competition-specific facts (train/test similarity, temporal or entity drift, public LB sample size) that must be checked directly; picking the wrong one without a compensating safeguard converts a diagnostic error into a submission error.

**Evidence.** Guanshuo Xu, solo 1st place, Jigsaw Rate Severity of Toxic Comments 2022: 'Public LB looks misleading so I focused on the validation performance only,' paired with linear/GA-weighted blends chosen specifically for robustness. Same competitor, solo 1st place, APTOS 2019 Blindness Detection: CV never correlated with LB under any scheme tried, so he 'solely relied on public LB,' paired with a compensating safeguard of limiting hyperparameter degrees of freedom. LANL Earthquake Prediction 1st place (The Zoo) went further still, rebuilding train itself to match test's feature distribution because public LB was actively adversarial, verified via a KS-test on OOF-vs-test prediction distributions.

**Seen in.** meta-gm-craft, tabular-classic

**Inverts when.** Trusting a CV-LB relation curve at all (as in Playground S6E2's method of picking a submission from just inside the last-trustworthy CV range rather than the maximum) requires a large, stable public LB as an explicit precondition — its own stated pitfall warns that applying it blindly on an unstable-LB competition like LANL or Jigsaw 'reproduces the exact mistake' the opposite camp warns against. The precondition, not the technique, decides which regime you are in.


### 10. Shape the training loss (and any downstream aggregation statistic) to mirror the actual evaluation metric rather than defaulting to a generic loss — but a literal, unsmoothed port of the metric formula is frequently numerically unstable at its own edge cases and needs deliberate stabilization to train at all.

**Why.** Gradient descent optimizes exactly the loss surface it's given, so a mismatched standard loss spends capacity on distinctions the real metric doesn't reward. But metrics themselves are frequently non-smooth near edge cases (small subgroup counts, values near zero), so a literal metric-as-loss port can fail to converge without an explicit smoothing term added on top.

**Evidence.** M5 Forecasting 1st place trained LightGBM directly with a Tweedie objective, not a two-stage hurdle model, for zero-inflated intermittent demand. Ventilator Pressure Prediction: switching fold-ensembling from mean to median (matching the MAE metric) moved public LB 0.157→0.155, and rounding to the nearest of 950 valid discrete values moved it to 0.153. Jigsaw Unintended Bias 1st place implemented a 'custom mimic loss' structurally replicating the competition's own subgroup bias-AUC decomposition.

**Seen in.** time-series, nlp-transformers, tabular-classic

**Inverts when.** The same class of literal metric port is explicitly flagged as unstable near its own boundary — generalized power-means blow up with small subgroup counts, Jaccard/IoU-style losses are non-smooth at extremes — so Tweet Sentiment Extraction's 1st-place team needed an added smoothing term before their metric-shaped loss would train stably at all. A naively 'more metric-faithful' loss can be strictly worse than a well-tuned standard loss until stabilized, inverting the expectation that closer-to-the-metric is automatically better.


### 11. Pair a simple model with a complex one so each supplies exactly what the other is structurally unable to (linear extrapolation for a GBDT that cannot extrapolate, a warm-start prior, a residual correction) — but the composite's value can be invisible when the simple/complex pairing is judged component-by-component instead of at the ensemble level.

**Why.** Different model families have well-understood structural blind spots — GBDTs split on observed ranges and cannot extrapolate; a global linear model can't fit local nonlinear interactions — so composing two cheap, well-understood models along their complementary strengths is more reliable than forcing one expensive model to implicitly approximate the other's strength.

**Evidence.** Rossmann Store Sales 1st place fed a per-entity Ridge trend extrapolation into XGBoost as a feature specifically because 'GBDT cannot extrapolate trends,' independently confirmed by a later academic review of Kaggle forecasting winners. Walmart Sales in Stormy Weather 1st place fit a flexible local per-entity baseline curve, then one global L1 linear model over the pooled residuals for calendar/holiday interactions, beating GBDT/RF/SVM ensembles outright. The set_base_margin warm-start trick (boosting from a linear model's own logits) is credited across two separate Playground Series wins by the same author as 'an overlooked XGBoost trick.'

**Seen in.** time-series, tabular-modern, tabular-classic

**Inverts when.** A residual-boosted or warm-started component can score worse than its own base model in standalone validation and still be the correct model to keep in the final ensemble — evaluating it only by solo CV, as any other candidate, causes you to wrongly discard it. The composite's value is sometimes only visible at the full-ensemble level, so the normal 'does this component look good alone' filter must be suspended specifically for residual/warm-start pairs.


### 12. When a decision (blend weights, feature subset, hyperparameters) is cheap to score against your validation data, search it directly and exhaustively rather than hand-picking — but this is simultaneously the most effective way to overfit your own validation set, since you are now optimizing an entire search procedure against one finite sample.

**Why.** Manual/uniform choices are one arbitrary point in a large decision space that OOF-based search can improve on directly and cheaply. But every added degree of freedom in the SEARCH, not just the model, consumes some of the validation set's information content, so a large enough search space relative to sample size guarantees you eventually find and lock in noise.

**Evidence.** Rossmann Store Sales 1st place tested roughly 125,000 pairwise model combinations against a single 6-week holdout to build a 10+-model ensemble. Playground S5E12 1st place saw pure OOF hill-climbing plateau and start diverging from LB even as its own CV kept rising, forcing a switch to Ridge-regularized stacking on ranked OOF predictions of only the top-36 models — a result scoring slightly lower on CV (0.70860 vs 0.70886) but generalizing better.

**Seen in.** tabular-modern, time-series

**Inverts when.** The Rossmann winner frames his own 125,000-combination search as trustworthy only because his 6-week holdout could support selecting among that many pairs 'without overfitting' — a claim inherently harder to sustain the smaller the holdout or the larger the search space. The S5E12 case shows the same family of search crossing from primary lever to actively misleading once candidate count grew large relative to OOF sample size, with no bright-line rule for where that crossover happens — the technique's own success condition is unverifiable in advance.


### 13. Ensembling / model-family diversity is the default way to buy score, because decorrelated errors partially cancel under averaging.

**Why.** Independent model families or feature sets make different mistakes on different rows; averaging (or a learned blend) removes the mistake-specific variance while keeping the shared signal, which is why nearly every top-3 team in the corpus blends rather than ships a single model.

**Evidence.** Otto Group 1st place (33 L1 models -> 3 L2 models -> L3 blend); CHAMPS top solutions all blend 8-14 models; Jane Street/Santander GBDT+NN blends; Bojer & Meldgaard's academic review of 6 Kaggle forecasting competitions: 'Ensembles won all of the competitions.'

**Seen in.** tabular-classic, tabular-modern, nlp-transformers, time-series forecasting, graph-molecular, medical-imaging, audio-signal, cv-segmentation-detection

**Inverts when.** Under a hard wall-clock/memory ceiling (an efficiency-scored track or a code competition with a fixed runtime budget), marginal accuracy from more models is worthless once total cost exceeds the ceiling, so the optimal strategy inverts to compressing a strong ensemble into one model. Team Hydrogen's Feedback Prize Efficiency-Prize winner distilled its full winning ensemble's soft pseudo-labels into one deberta-v3-large (0.557 private LB, top-3-accuracy-equivalent, in 5m40s), and LMSYS's 1st-place solo-gold winner averaged 5 folds' LoRA adapter weights into one merged adapter instead of ensembling checkpoints at inference, trading a little accuracy for a large constant-factor speedup.


### 14. Screen every candidate feature for train/test distributional stability (adversarial-validation AUC, KS-test, early-vs-late temporal single-feature AUC) before trusting it, regardless of raw importance — but single-feature screens structurally cannot see value that only exists in combination, so aggressive cutting discards real interaction terms.

**Why.** A feature can be a powerful summary of past structure while encoding a pattern that has already drifted; because global importance metrics don't distinguish 'genuinely predictive' from 'predictive only in a since-shifted regime,' a stability check is a necessary complementary signal — but by construction it evaluates features one (or a few) at a time, so it cannot detect value that only appears jointly.

**Evidence.** IEEE-CIS Fraud Detection 1st place's time-consistency filter (train on an early month, score on the last month) dropped roughly 5% of candidate columns, including the entire V322-V339 block, which scored ~0.60 single-feature training AUC but only ~0.40 on later-month data. LANL Earthquake Prediction 1st place gated every feature at a KS-test p>0.05 between train and test, ending with only 4 features in the winning model.

**Seen in.** tabular-classic, meta-gm-craft

**Inverts when.** The IEEE-CIS source states the inversion directly: 'a feature can fail this single-feature test yet still help in combination with others... an overly aggressive cutoff can discard useful interaction terms.' The same screening step that correctly removes genuinely-drifted noise features has no mechanism to distinguish that from a feature that's individually weak but a valuable interaction term — strict single-feature gating is safe for the former and actively harmful for the latter, and the check cannot tell which case it is looking at.


### 15. Competition datasets are frequently the output of a specific, knowable HOST construction pipeline (padding with synthetic rows, sampling a smaller 'original' real dataset) — detecting and exploiting that pipeline is a legitimate, high-value lever, but it is entirely a Kaggle-competition-meta skill with no analogue in real production deployment.

**Why.** A host's data-construction code is not adversarially hardened the way a production system defending against gaming would be, so cheap detectors (per-value uniqueness statistics, adversarial-validation AUC on source flags, frequency-signature checks) recover real, exploitable train/test asymmetries that have nothing to do with the underlying prediction problem itself.

**Evidence.** Santander Customer Transaction Prediction 1st place built per-feature uniqueness categories that alone reached LB 0.910, rising to 0.921 once real vs. host-injected synthetic test rows were correctly separated (via a different competitor's published detector) before computing the joint train+test uniqueness statistic. Multiple Playground Series wins reuse the linked pre-synthetic 'original' dataset two structurally different ways — extra training rows AND a source for target-encoded columns merged into the synthetic data — as separate, non-redundant ensemble members.

**Seen in.** tabular-classic, tabular-modern

**Inverts when.** This entire family has essentially zero transfer value once the competition framing is removed — a real production model has no linked 'original' dataset merged with synthetic padding by a known host process, so treating this as a general modeling principle rather than a leaderboard-specific meta-skill is a category error. It inverts completely outside the Kaggle-competition domain, since none of the underlying mechanism applies once you leave that specific institutional setting.


### 16. A final post-hoc pass that exploits side information the base model structurally can't see (same-entity grouping, calibration to the known label mean) adds a small, reliable improvement at zero retraining cost — but its sign depends entirely on the correctness of an upstream reconstruction that is usually unverifiable against ground truth.

**Why.** Row-by-row base models have no way to use cross-row structural information unless it's explicitly injected; a grouping-and-averaging or recalibration pass injects it for free after the fact. But because the 'group' is often an inferred quantity rather than a given, the correction's benefit is conditional on that inference being right, with no way to check it directly.

**Evidence.** IEEE-CIS Fraud Detection 1st place's client-consistency post-processing (replacing each row's prediction with its reconstructed-client group average) added +0.001 LB in a competition decided in the 4th decimal, on top of an already-validated UID reconstruction. Feedback Prize Effectiveness 1st place's mean-recalibration of predictions to the train-label mean exploited the fact that log-loss is only calibration-optimal when predicted and true means match.

**Seen in.** tabular-classic, nlp-transformers

**Inverts when.** The same family flips sign outright when the grouping is wrong — averaging over an incorrectly-merged 'client' blends unrelated rows' predictions together, actively hurting rather than helping, and because there is no ground-truth client ID to check against in an anonymized dataset, you cannot verify in advance which regime you're in. Even the source's own 'verified' UID reconstruction was outperformed by the model's own implicit identity-clustering, meaning the correction was built on an admittedly imperfect proxy to begin with.


### 17. Default to training one model that cross-learns across an entire panel/entity hierarchy rather than fitting separate local models per entity — but this default inverts specifically on small, low-heterogeneity panels, where a single simpler regularized global model wins outright and further ensembling actively hurts.

**Why.** Pooling data across entities lets a model learn shared structure (seasonality, promotion response) from far more effective samples than any single entity's own history provides. But the benefit is proportional to how much genuinely shared, transferable structure exists across entities; on a small or homogeneous panel that transfer has little left to contribute over a well-regularized model fit directly.

**Evidence.** An academic review of four Kaggle retail-forecasting winners (Rossmann 2015, Web Traffic 2017, Favorita 2018, Walmart Store Sales 2014) found global cross-learning plus ensembling common to all of them — one XGBoost family spanning 1,115 Rossmann stores, one RNN family spanning roughly 145,000 Wikipedia pages. M5 Forecasting 1st place explicitly selected hierarchy-cut depth by watching cross-fold variance rise, not just mean CV improve, because deeper cuts fragment the cross-learning benefit.

**Seen in.** time-series

**Inverts when.** The same review's smallest, lowest-entropy dataset, Walmart Sales in Stormy Weather (2015), inverts the pattern outright — its 1st place used one simple regularized global linear model on residuals, beating GBDT/RF/SVM ensembles, with the review noting explicitly that 'global always wins' isn't universal for small-N, low-entropy panels. The correct amount of pooling is a dial set by measured panel heterogeneity, with documented failure modes on both the over-pooled and over-segmented ends.


### 18. For metrics that are discontinuous functions of a continuous score (top-k ranking metrics, majority-vote consensus), reducing per-run noise via brute repetition has outsized value because a small perturbation can flip which side of a hard boundary an item lands on — a benefit that mostly vanishes for smooth metrics where one well-cross-validated run already averages out that noise internally.

**Why.** Near a decision boundary a metric behaves like a step function of an underlying continuous quantity; independent noisy estimates of that quantity, averaged, concentrate probability mass away from the boundary and reduce the chance of landing on the wrong side of it — a mechanism with no analogue once the metric is already smooth in the underlying score.

**Evidence.** A cited public notebook for Playground Series S5E6 showed 100 averaged 5-fold XGBoost reruns (500 total fits) lifted MAP@3 from a 0.376 per-fold average to 0.380 combined, contributing to the eventual 1st place. AI Mathematical Olympiad Progress Prize 2 winners used 12-way parallel self-consistency sampling with majority-vote consensus as a core accuracy lever under a hard time ceiling.

**Seen in.** tabular-modern, code-efficiency

**Inverts when.** The source calls this explicitly 'a compute-multiplication strategy, not a modeling insight' — averaging 100 seeds of a mediocre configuration won't out-rank one strong model, so once base-model quality rather than noise is the bottleneck, more repetition stops helping. It also directly inverts against a hard wall-clock budget: AIMO's own pipeline deliberately CUTS self-consistency sampling short via early-stopping-on-consensus to save time, trading away exactly the noise-reduction benefit because runtime, not ranking-boundary noise, was the binding constraint there.


### 19. Non-linear ensembling captures value from individually-weak, structurally-diverse models that a linear blend would zero out — model diversity is worth more than another copy of your best model, but only when a trained (not averaged) combiner sits on top.

**Why.** A non-linear level-2/3 meta-model can learn per-instance 'which base model to trust,' recovering signal from a weak-but-decorrelated learner's errors; a linear/weighted-average blend can only globally reweight, so a consistently weak model just gets down-weighted toward zero or adds noise instead.

**Evidence.** TabPFN was kept at CV 13.2 (tied-weakest of 12 model families) inside Playground S5E4's winning 75-model, private-LB-11.44 stack. Otto Group Product Classification 1st place (2015) explicitly kept Naive Bayes, Sofia, and various-k KNN despite weak solo scores, stating plainly they 'learn not to discard low performance algorithms, since it have enough predictive power to improve performance in a 2nd level training.' Feedback Prize ELL/Effectiveness 1st places manufactured decorrelated inputs by varying pooling heads AND reformulating the same prediction task (ratio targets, auxiliary dominant-feature prediction, dominant-feature-dropped specialization), gating every addition on ensemble CV/LB improvement rather than solo score.

**Seen in.** tabular-modern, tabular-classic, nlp-transformers

**Inverts when.** Inside a purely linear hill-climbing/Ridge ensemble, a consistently weak model is 'more likely to be zero-weighted or add noise instead' per the same source material — the identical weak-but-diverse model that's a net positive feeding a GBM/NN stacker is dead weight in a linear blend. Whether to include it inverts entirely on which combiner sits above it, not on the model's own quality.


### 20. Adding auxiliary prediction heads trained on cheap, correlated-but-unscored metadata (discourse type, diagnosis subtype, BIRADS density) regularizes a shared backbone toward semantically meaningful representations, at low marginal cost since the heads share the expensive forward pass.

**Why.** A model trained on a single scored objective takes the shortest path to minimizing that loss, which can exploit spurious shortcuts; forcing the shared representation to also predict correlated signals removes shortcut-only representations and keeps ones that generalize, especially valuable when the primary labeled set is small.

**Evidence.** Jigsaw Unintended Bias 1st place lists auxiliary tasks as technique #3 of six; RSNA Breast Cancer 2023 2nd place's auxiliary EQL-loss heads for BIRADS/density/view/invasive-status at 0.1x weight was listed explicitly under 'what worked.'

**Seen in.** nlp-transformers, medical-imaging, graph-molecular

**Inverts when.** Blindly adding any available correlated signal is not safe by default -- it must be validated per-component, since a historically strong signal can actively hurt once folded into a larger fusion without checking. CAFA 5 Protein Function Prediction's 1st place found their own Net-KNN component, historically one of their strongest per prior published work, hurt overall performance specifically when blindly included, and a different Breast Cancer team found pseudo-labeling an absent auxiliary metadata field backfired -- the same category of lever that helps in one configuration hurts in a structurally similar one.


### 21. Beam search, or plain greedy construction, is the correct cheap default for discrete sequential-construction problems with small per-step branching, and should be tried before escalating to heavier machinery -- group theory, exact MIP, population-based search.

**Why.** Keeping only the top-k partial states by a cheap heuristic score is inexpensive to implement and often near-optimal whenever local goodness correlates reasonably well with eventual global goodness, which covers a large fraction of discrete construction problems without deeper structural analysis.

**Evidence.** Santa 2023's 1st place explicitly deprioritized the entire 'wreath' puzzle family because 'a very short solution could be found using simple beam search'; ARC Challenge 1st place's DAG-deduplicated depth-3/4 enumeration is a beam-adjacent brute-force default tried before any learned component.

**Seen in.** optimization-combinatorial

**Inverts when.** Beam search's core weakness is its greedy locality -- on deceptive, path-dependent landscapes where a good long-range state looks locally bad by the cheap heuristic, a plain score-only beam permanently discards the seed of the eventual best solution with no completeness guarantee to fall back on. Santa 2022's two independent top teams both found a naive beam collapses diversity in precisely the dimension that determines long-range feasibility, and had to augment it with a coarsened DP-feasibility table or invariant-partitioned bucketing before it worked on their harder constraint family.


### 22. When the leaderboard metric has a decomposable, differentiable structure that differs from a generic proxy loss (BCE/MSE), implementing the metric's own formula directly as the training loss captures gradient signal a proxy loss would smooth away.

**Why.** A generic loss optimizes a different, if correlated, objective than the one being scored; wherever the metric's structure is knowable and smooth (subgroup masks, correlation formulas, weighted terms), training directly against it removes that objective mismatch instead of hoping the proxy's optimum lines up with the real one.

**Evidence.** Jigsaw Unintended Bias 1st place implemented the exact subgroup/BPSN/BNSP bias-AUC decomposition as a custom loss; Tweedie-objective GBDTs for M5's zero-inflated demand skip a manual two-stage hurdle model entirely.

**Seen in.** nlp-transformers, time-series forecasting

**Inverts when.** For metrics that are rank-based or fundamentally non-differentiable (Spearman correlation, exact-match, unsmoothed NDCG), there is no way to port the metric's formula directly into a gradient-based loss -- the principle simply doesn't apply, and teams fall back to a smooth proxy loss instead, a limitation the Jigsaw writeup itself flags. Even within the applicable regime, a literal 1:1 port can be numerically unstable near the metric's own edge cases (Jigsaw's generalized power-mean terms blow up on small subgroups), so the mirrored loss usually still needs deliberate smoothing, not a verbatim translation.


### 23. How many stacking levels (and how non-linear the top level) is worth building scales with how many genuinely diverse base models you actually have — a 3-level stack with a non-linear L2/L3 is correct for a large team pooling 90+ models and actively counterproductive for a solo competitor with a handful.

**Why.** Each added stacking level is itself a model that needs enough independent training signal (diverse, decorrelated OOF columns) to learn real structure rather than noise. Large teams naturally produce enough base-model diversity to feed a deep stack safely; a small pool gives a deep stack too few effectively-independent inputs, so it mostly overfits the OOF set instead.

**Evidence.** Home Credit Default Risk 1st place's 3-level stack (90+ L1 models from a 6-person team feeding a non-linear L2 of NN/ExtraTrees/hill-climbing) is captioned directly in its own trigger condition: 'Large teams pooling many genuinely diverse base models (50+); with fewer/similar models a 3-level stack mostly adds overfitting risk for negligible gain.' Otto Group 1st place used 33 L1 models feeding 3 non-linear L2 models specifically because a dominant-but-sometimes-missing feature created genuine regime-switching structure only a trained L2 combiner could exploit per row.

**Seen in.** tabular-classic, tabular-modern

**Inverts when.** The mined evidence states its own inverting threshold explicitly rather than leaving it implicit — there is no universal 'stack N levels' rule, only a threshold set by a model pool's REAL (correlation-checked) diversity, not its nominal count. A solo competitor mechanically copying a 90-model team's 3-level architecture onto 8 models inverts the intended benefit into pure overfitting risk.


### 24. Under a hard, shared compute/time ceiling across many sub-tasks of uneven difficulty, allocate the budget adaptively (cascades, cross-task time-banking, early-stopping on consensus) rather than spending uniformly — but every cascade's accuracy ceiling is set by its cheapest, earliest stage, since a candidate lost there can never be recovered downstream.

**Why.** Task difficulty is heterogeneous and evaluation cost grows steeply between stages, so spending the expensive tail of the budget only on candidates that survive cheap filtering multiplies effective throughput — at the structural cost that early-stage recall becomes a hard, invisible cap on the whole pipeline's final accuracy.

**Evidence.** Eedi Mining Misconceptions 1st place's 3-stage rerank cascade (broad retrieval to 14B to 32B to 72B listwise) improved private LB by a combined +0.023 by adding narrowing stages ahead of its most expensive model. RSNA Cervical Spine Fracture 1st place segmented vertebrae cheaply on a small masked subset before running expensive classifiers only on cropped regions, fitting a full ensemble inside 7.5 hours. AI Mathematical Olympiad Progress Prize 2 1st place combined per-question time-banking with consensus early-stopping to fit a strict 5-hour/50-question ceiling.

**Seen in.** code-efficiency, optimization-combinatorial

**Inverts when.** This directly inverts the instinct behind ensembling weak-but-diverse models (see the non-linear stacking principle): adding a weak early-stage filter dilutes recall and CAPS the ceiling regardless of how strong later stages are, whereas the same 'add more diverse cheap models' move helps a flat ensemble. Cascades and flat ensembles reward opposite instincts about where diversity or weak models are safe to add.


### 25. When timestamps, entity IDs, or other structure have been anonymized away, reconstructing that hidden structure from surviving artifacts (tick sizes, hashed-field fragments, missingness patterns) unlocks an entire class of otherwise-inaccessible standard feature engineering -- aggregates, cross-sectional pooling, trend extrapolation.

**Why.** Group-by aggregates, lag features, and cross-sectional comparisons are only meaningful once rows are correctly grouped by real-world entity or ordered by real-world time; anonymization hides these keys but not the underlying structure, which just requires one extra reconstruction step to reach.

**Evidence.** Optiver 1st place recovered true chronological order from tick-size-implied unrounded prices via 1-D t-SNE, verified against real 2020-2021 market history; IEEE-CIS 1st and 2nd place independently reconstructed client/UID identity from fragmented fields, raising local AUC from 0.9245 to 0.9377.

**Seen in.** tabular-classic, time-series forecasting, meta-gm-craft

**Inverts when.** The reconstruction's reliability isn't guaranteed to transfer to the test set the way it's verified on train, so using it directly as a raw predictive feature -- rather than as scaffolding for other features or CV splits -- is a documented, self-identified failure mode. Optiver's own author declined to use the recovered order as a direct test-set feature ('no guarantee' it holds there), and IEEE-CIS's UID team explicitly flags that extending the trick to lag-stacked predictions on the reconstructed ID produces leakage-shaped CV inflation with zero LB payoff.


### 26. In adversarial or open-set detection tasks, where the private test set is deliberately drawn from undisclosed generators or attacks, investing in training-data source diversity dominates investing in model or architecture sophistication — the inverse of the usual priority ordering for competitions with a fixed, well-specified label distribution.

**Why.** No amount of architecture or hyperparameter tuning helps against a generator or attack style your training data never represented at all; coverage of the input space is the binding constraint in an adversarial setting, whereas in a fixed-distribution task the input space is already well-sampled by train and further gains come from fitting it better, not covering more of it.

**Evidence.** LLM - Detect AI Generated Text 1st place, verbatim: 'modelling strategies themselves had a lesser impact on the overall performance as compared to the datamix,' with multiple single models scoring 0.970+ once the datamix spanned 4 generator-source categories and 7 augmentation types.

**Seen in.** nlp-transformers

**Inverts when.** The same source states the inversion explicitly: 'for competitions with a fixed, well-specified label distribution, architecture and pipeline tricks matter proportionally far more, and over-indexing on just diversify the data for e.g. a fixed patient-notes NER task is a category error.' Whether data-source diversity or architecture sophistication dominates flips entirely on whether the test distribution is adversarially hidden or fixed and knowable — a property of the competition, not a general truth about the domain.


### 27. A host's data-generation or preprocessing artifact -- synthetic rows built from independently-resampled marginals, or a secret crop-adjacency structure between train and test images -- can be forensically detected and exploited for one of the largest single score gains available in a competition.

**Why.** Any automated data-generation or splitting process leaves statistical fingerprints (implausible value uniqueness/repetition patterns, tileable adjacency) unrelated to the underlying task but fully legible to a model or hand-built detector once someone looks, and they require no domain understanding of the actual task to find.

**Evidence.** Santander Customer Transaction 1st place's train/test row-uniqueness features moved LB from 0.910 to 0.914 purely from detecting host-synthesized filler rows; TGS Salt 1st place found train tiles sharing a train/test seismic-mosaic adjacency and copied labels across the seam.

**Seen in.** tabular-classic, tabular-modern, meta-gm-craft, cv-segmentation-detection

**Inverts when.** The same technique inverts from winning modeling insight into a diagnostic red flag for pure leaderboard overfitting once you check whether the gain survives out of the exploited partition. TGS Salt's own mosaic-adjacency trick moved public LB from 0.876 to 0.884 (+0.008) while leaving private LB completely unchanged (+0.000) -- the textbook signature of exploiting an artifact that exists only in the specific split being probed, not a pattern generalizing even within the same competition's private set, let alone to real deployment.


### 28. The statistic used to combine multiple predictions for the same row must match the geometry of the scoring metric -- arithmetic mean for squared-error metrics, median for absolute-error metrics, rank/percentile averaging (not raw-probability averaging) when combining differently-calibrated models under an order-sensitive metric like AUC.

**Why.** Each statistic is the population-level minimizer or invariance-preserver for a specific loss geometry -- mean minimizes squared error, median minimizes absolute error, rank transforms remove scale/calibration disagreement a ranking metric doesn't care about -- so using the wrong one fights the objective instead of serving it.

**Evidence.** A public Ventilator Pressure Prediction notebook moved LB 0.157->0.155 purely by switching fold-ensembling from mean to median under the competition's MAE metric; SIIM-ISIC Melanoma 2020's 1st place rank-averaged predictions across differently-calibrated backbones specifically because the metric was AUC.

**Seen in.** time-series forecasting, medical-imaging

**Inverts when.** For metrics that are proper scoring rules rewarding calibrated probability magnitude (log-loss, Brier score), rank-averaging is actively wrong -- it discards exactly the magnitude information the metric measures, per the SIIM-ISIC writeup's own caveat. Median-ensembling under an RMSE/L2 metric is likewise actively wrong since mean is metric-optimal there, and the whole lever needs enough models (roughly 5+) to have a meaningful cluster -- with 2-3 models the 'wrong' statistic barely differs from the 'right' one.


### 29. Continuing masked-language-model pretraining on unlabeled in-domain text -- which can legitimately include the competition's own unlabeled test text, since no labels are used -- before supervised fine-tuning closes the gap between a generic pretraining corpus's vocabulary/style and the task's, cheaply.

**Why.** General-purpose pretraining corpora under-represent domain-specific vocabulary, sentence structure, and stylistic conventions; a short continued-MLM phase on in-domain text shifts subword embeddings and attention patterns toward that vocabulary/style before the labeled objective ever sees it, for a fraction of full-pretraining's cost.

**Evidence.** NBME 1st place's MLM continuation on patient_notes.csv (excluding train.csv rows) added +0.002 CV; NBME 4th place applied the identical MLM step across all 5 of its backbone/head variants before fine-tuning.

**Seen in.** nlp-transformers

**Inverts when.** This only helps when there's meaningfully more in-domain unlabeled text available than the labeled train set alone provides; when the domain is narrow enough that train already covers its vocabulary and style, the extra MLM phase is wasted compute with nothing new to teach the model. Jigsaw Multilingual's 1st-place team explicitly tried the identical lever ('further MLM pretraining ... using task data') and lists it under 'what didn't work,' confirming the failure mode occurs even within the same general NLP-transformer domain, not just in principle.


### 30. Freezing the backbone's lower/embedding layers and putting adaptation capacity into a widened task-specific head is the right lever specifically when the task is shallow (short-text/lexical similarity) or the backbone's pretrained representation already matches the task closely.

**Why.** Fine-tuning risks catastrophic forgetting and overfitting, especially on small competition datasets; when the task only needs shallow lexical/semantic matching the pretrained representation already encodes, spending capacity on head width rather than backbone adaptation avoids that risk while adding capacity exactly where it's useful.

**Evidence.** US Patent Phrase to Phrase Matching 1st place froze BERT's embedding layer entirely and widened the head instead, explicitly reasoning that the target was 'simple short words similarity' not requiring deep fine-tuning.

**Seen in.** nlp-transformers

**Inverts when.** The same lever caps performance well below a fully fine-tuned model whenever the task actually needs deep semantic or reasoning adaptation the frozen layers don't already encode -- long-document argument evaluation, nuanced writing-quality judgment (CommonLit, Feedback Prize) -- where competing top solutions instead invested in full backbone fine-tuning, layer-wise learning-rate decay, and reinitializing top layers rather than freezing anything. The right choice is set by task depth, not by a general preference for parameter-efficiency.


### 31. Whether to invest scarce effort in feature engineering or in model-family diversity is decided by triage at the start of a competition, not by habit — and over-investing in the wrong one, even when local CV keeps improving, actively costs placement.

**Why.** Feature engineering pays off when the data has exploitable relational/combinatorial/temporal structure and enough rows that new features don't overfit; on small, noisy, or heavily synthetic data the same effort mostly manufactures spurious signal, so cross-validated model-family diversity captures more real signal per unit effort instead.

**Evidence.** Chris Deotte's Playground S4E12 1st place credited combinatorial target/count encoding across roughly 145,000 candidate column combinations as 'the secret sauce.' The same author's Playground S5E3 2nd place instead used a single RAPIDS SVC with NO feature engineering to score equivalent to 2nd place alone (private LB 0.90610), and a 3-model no-FE blend would have scored 1st (0.90728) had it shipped as-is.

**Seen in.** tabular-modern

**Inverts when.** The S5E3 case is itself the inversion of the S4E12 case within the same author's own record: adding 3 more feature-engineered models after local CV rose (0.900-0.901) DROPPED private LB (0.90599-0.90604) and cost 1st place, directly contradicting 'more feature/model investment always helps if CV improves.' The correct choice flips on data properties that must be diagnosed fresh, not assumed from the last competition's winning formula.


### 32. When a hard constraint is expensive to enforce inside a solver's main search loop but empirically 'almost satisfied' by solutions that optimize a relaxed version of the objective, decompose into a fast relaxed-core solve plus a separate, cheap dedicated repair pass — but the decomposition's applicability can only be confirmed after paying most of the analytical cost it's meant to save.

**Why.** Enforcing an expensive path-dependent constraint on every candidate move inside the main search multiplies per-iteration cost across the whole search; isolating it into a post-hoc repair step that runs once on an already-near-optimal relaxed solution is far cheaper when the gap between relaxed-optimal and fully feasible is genuinely small.

**Evidence.** Santa 2022's 1st place (soft-penalized GA-EAX core, DP-table-pruned beam-search repair) and 2nd place (soft-penalized custom-LKH core, randomized-backtracking repair) independently converged on this exact two-stage architecture, landing within 0.0001% of each other (74075.70654 vs 74075.706541) using different core solvers and different repair algorithms.

**Seen in.** optimization-combinatorial

**Inverts when.** Both teams spent significant SEPARATE analytical effort deriving exact necessary feasibility conditions before trusting that the constraint really was 'almost free' — the decomposition only pays off if that empirical assumption holds, and there is no way to know this a priori without roughly the same domain analysis the shortcut is meant to save. If the assumption is wrong, the relaxed-core solutions are genuinely unrepairable and the whole run is wasted.


### 33. A cheap, post-hoc correction to model output for a known systematic difference between training signal and deployment reality -- a labeling-protocol bias (ground truth built as the intersection of multiple readers, systematically smaller than any single-annotator-trained model predicts) or a train/test prevalence mismatch -- can move the score more than further architecture work, because it fixes a bias the model's own loss function has no way to see.

**Why.** The model is trained to be correct relative to its own training distribution's protocol and label frequencies; if the true scored target was generated under a different protocol or class balance, no representational capacity closes that gap, because the gap isn't a modeling error, it's an unmodeled input to the scoring function.

**Evidence.** RSNA Pneumonia Detection 1st place's uniform 87.5% box-shrink corrected for the intersection-vs-single-read annotation protocol, moving public LB 0.222->0.260 in one step; Rainforest Connection Species Audio Detection's independent 9th-place log-odds rescaling for train/test class-prevalence mismatch moved private LB 0.926->0.963 (~13 places) from that one step alone.

**Seen in.** medical-imaging, audio-signal, tabular-classic

**Inverts when.** Because there's usually no ground truth available locally to check the correction against, a wrong estimate of the true target distribution doesn't fail visibly -- it confidently miscalibrates output in the wrong direction with no local warning sign. The RSNA correction factor itself could only be discovered by directly probing public-LB score deltas, a real leaderboard-overfitting risk baked into the very mechanism that makes the lever powerful.


### 34. When a competition target is produced by an instrumented, simulated, or otherwise controlled physical process rather than organic behavior, reverse-engineering that generating function yields exact features or predictions that structurally dominate any learned approximation wherever the process is truly deterministic.

**Why.** A learned model has to approximate the generating function from noisy, indirect examples; if the function itself can be identified and its parameters fit directly, that's reconstruction, not approximation, with zero error wherever the assumption holds exactly.

**Evidence.** Ventilator Pressure Prediction 1st place exactly recovered PID-controller-generated pressure for 66% of test timesteps with zero error by identifying and inverting the controller's own formula; G2Net Detecting Continuous Gravitational Waves 1st place used pure physics-based matched-filter template search with zero machine learning.

**Seen in.** time-series forecasting, graph-molecular, audio-signal

**Inverts when.** This lever requires an instrumented or simulated data-generating process with a knowable closed form; it's inapplicable by construction to organic, human-generated targets, where no such generating function exists to recover. Even within its applicable regime it's brittle to injected noise -- Ventilator's winning solution needed a second, ~9-hour brute-force matching pass specifically to handle organizer-added exploratory noise that broke the clean inversion for the remaining 34% of cases.


### 35. Splitting one global model into several models along a natural entity hierarchy (per-store, per-store-category, per-store-department) lets each sub-model fit locally homogeneous structure a single global model has to average away.

**Why.** A single model's splits/weights compromise across every sub-population in the training set; when different hierarchy branches genuinely differ in how features relate to target, segmenting removes that cross-branch compromise cost, up to the point where each branch still has enough data to fit reliably.

**Evidence.** M5 Forecasting Accuracy 1st place trained separate models at multiple hierarchy cuts (by store, by store x category, by store x department) and selected the cut using both mean and standard deviation of validation score.

**Seen in.** time-series forecasting

**Inverts when.** Past an optimum depth, the lever inverts -- every additional split multiplies model count and starves each sub-model of data, showing up specifically as rising CV variance even while CV mean is still improving. M5's own winner picked segmentation depth by watching that variance signal rather than mean alone, precisely because mean-only selection would keep pushing past the point where segmentation helps; M5's 3rd-place NN team separately found predicting at a different hierarchy level for reconciliation was outright worse.


### 36. Under one hard aggregate compute/time ceiling shared across many independent sub-tasks of varying difficulty, dynamically reallocating budget -- banking an easy item's unused time for a shared pool, stopping generation early once samples agree -- captures most of an oracle allocation's benefit without knowing task difficulty in advance.

**Why.** A fixed uniform per-item budget wastes compute on easy items that finish early and starves hard items that need more; letting unused time flow to where it's still needed, and stopping early once redundant sampling has converged, approximates knowing the difficulty distribution ahead of time using only observed completion signals.

**Evidence.** AIMO Progress Prize 2's 1st place ran a 350-second base budget per question with up to 210 extra seconds drawn from a shared buffer (560s hard ceiling), plus early-stopping once 4 of the first 5 samples agreed, under a 5-hour/50-question ceiling.

**Seen in.** code-efficiency, llm-era

**Inverts when.** Early-stopping on sample agreement implicitly bets that consensus equals correctness; on any problem with a confident-but-wrong failure mode -- a systematic bias shared across most sampled trajectories, not independent random error -- this exact mechanism locks in the wrong answer faster and with more apparent confidence than a slower, complete search would have, since the stopping rule can't distinguish convergence-because-right from convergence-because-the-sampling-process-is-biased.


### 37. Retraining an identical model configuration many times with different seeds and averaging raw predictions denoises ranking-sensitive metrics (MAP@k, NDCG, top-k selection) disproportionately more than it moves smooth regression/classification losses, because probability noise near a decision boundary flips top-k membership even when it barely moves the underlying loss.

**Why.** A ranking metric's value depends on a discrete, threshold-sensitive event (which items land in the top k), far more sensitive to small stochastic prediction shifts than a smooth scalar loss is; averaging many independent noisy estimates of the same probability shrinks exactly that boundary-crossing noise.

**Evidence.** A cited Playground S5E6 public notebook showed 100 averaged 5-fold XGBoost runs (500 model fits total) lifted MAP@3 from an average of 0.376 per single fold to 0.380 combined; Web Traffic Time Series Forecasting 1st place's 3-seed x 10-checkpoint (30-model) ensemble plus ASGD weight averaging was explicitly credited with the winning leaderboard-SMAPE gain.

**Seen in.** tabular-modern, time-series forecasting

**Inverts when.** This is pure compute-multiplication, not a modeling insight, and needs GPU-scale budgets to be affordable; under a hard wall-clock or memory ceiling (an efficiency-scored track, or a code competition with a fixed runtime cap), retraining dozens to hundreds of redundant seeds is simply unaffordable, forcing the same denoising goal to be pursued through cheaper means -- larger single models, cascades, or accepting the noise -- instead.


### 38. Once your CV/OOF procedure has finished selecting hyperparameters and model composition, retrain the final shipped model(s) on 100% of the data to recover what K-fold necessarily withholds — accepting that this exact final retrain is the one pipeline step whose correctness cannot itself be checked against held-out data.

**Why.** K-fold trades away 1/K of training data for an honest performance estimate; once that estimate has done its job, continuing to withhold data from the shipped model leaves signal on the table for no further decision-quality benefit. But the heuristics used to approximate full-data convergence (scaled iteration counts, LR schedules) are, by construction, unverifiable, since no held-out set remains at 100%-data scale to confirm them.

**Evidence.** Chris Deotte states this is applied 'in all my Kaggle competitions' and quantifies it directly in Playground S5E6: averaging 100 full-data-retrained, multi-seeded 5-fold XGBoost reruns raised MAP@3 0.376→0.380, contributing to the final private LB 0.38652, 1st place.

**Seen in.** tabular-modern

**Inverts when.** The author's own iteration-count heuristic (for example +25% boosting rounds for K=5) is explicitly labeled 'an approximation, not a validated number' for exactly this reason — the technique that best uses all your data is also the one technique whose own correctness you cannot empirically verify before submitting, inverting the normal expectation that a good technique is one you can validate before relying on it.


### 39. Leak-free, structurally-matched cross-validation (purged, embargoed, grouped, or gapped to mirror the real train-to-test boundary) is a precondition for every downstream method that consumes out-of-fold predictions -- stacking, hill-climbing, pseudo-labeling, feature selection.

**Why.** Leakage in the base CV split doesn't stay contained -- it gets amplified by every method built on top of the OOF predictions, since those methods select or weight based on exactly the inflated signal the leak created.

**Evidence.** Jane Street 1st place's jointly-trained (not pre-trained) AE-MLP fixed encoder leakage into validation folds; IEEE-CIS 2nd place's time-gapped CV; Optiver's and Ubiquant's purged-group time-series CV; Google QUEST's leak-free multi-round pseudo-labeling (leak-free CV 0.414->0.422 vs. a leaky 0.414->0.445 the LB didn't agree with).

**Seen in.** meta-gm-craft, tabular-classic, tabular-modern, time-series forecasting

**Inverts when.** Once model/feature/ensemble-weight decisions are already locked in and only the final production fit remains, the same purging/embargo that protected those decisions becomes pure waste, discarding real training rows near every fold boundary with no further protective benefit. Ubiquant's 1st place used PurgedGroupTimeSeriesSplit for feature and hyperparameter decisions but explicitly downgraded to plain grouped KFold for the final training fit to recover that discarded data.


### 40. Merging two or more genuinely independently-built solutions (different features, algorithms, or hand-written heuristics, not just different seeds of one pipeline) decorrelates errors far more effectively than varying hyperparameters within a single shared pipeline.

**Why.** Ensembling's benefit scales with how uncorrelated component errors are; independent design choices produce genuinely different blind spots, whereas hyperparameter variation on one pipeline mostly perturbs the same blind spots by degree.

**Evidence.** ARC Challenge 2nd place merged a width-3 beam-search solver with a teammate's independently-designed solver, each covering different, only partially-overlapping task subsets; CPMP's late merger into a new IEEE-CIS Fraud Detection team under 10 days left, contributing an independently-sourced pipeline, lifted the team's blend.

**Seen in.** optimization-combinatorial, meta-gm-craft, tabular-classic

**Inverts when.** This depends on independence being real, not assumed, and real independence is something teams can architect for but frequently cannot fully guarantee -- the ARC 2nd-place team explicitly attributes their near-zero solved-task overlap to luck. Merging a second copy of an approach derived from the same public kernel or shared method library adds team size and compute without adding signal, a failure mode IEEE-CIS's own late-merger writeup flags directly.


### 41. Chaining structurally-complementary model families -- a model that captures what another structurally cannot, such as a linear model's extrapolation ability feeding a GBDT's split-based fit, or a simple model's logits warm-starting a boosted model -- exploits each family's specific blind spot rather than just averaging general-purpose strength.

**Why.** GBDTs cannot extrapolate past observed feature ranges because they split on observed values; a linear/Ridge component captures exactly the monotonic trend a GBDT can't, so composing them fixes a structural gap rather than the random-noise gap plain ensembling addresses.

**Evidence.** Rossmann 1st place's Ridge-regression trend correction layered on GBDT; Playground S5E5 1st place's 'NN over LinearRegression' residual stack (standalone CV 0.0608->0.0599); Elo Merchant 1st place's classifier-gated regressor split for a point-mass-contaminated target (0.015 RMSE improvement over one direct regression).

**Seen in.** time-series forecasting, tabular-classic, tabular-modern

**Inverts when.** Because the second model is fit specifically to the first model's residual or logit, it must be judged by its marginal contribution to the combined pipeline, not by its own standalone score -- the usual instinct to drop any component that 'looks weak in isolation' is directly wrong here. Playground S5E5's own 'XGB over NN' residual stage didn't improve its own standalone CV yet was kept because it improved the final ensemble.


### 42. A multi-stage cascade (cheap broad retrieval into progressively larger/pricier rerankers) bounds the cost of the most expensive model by shrinking the candidate pool it runs on, but every later stage's ceiling is set by what the first stage keeps, so the first stage must be optimized for recall specifically, not for its own natural ranking/accuracy metric.

**Why.** A later stage can only refine what survived every earlier filter; a true answer dropped at stage 1 is permanently unrecoverable no matter how good stage 4 is, so stage 1's job is structurally different from every later stage's (maximize coverage vs. maximize precision/ordering), and optimizing it the same way silently caps the pipeline's ceiling.

**Evidence.** Eedi 1st place's 4-stage funnel (retrieval -> 14B ranker -> 32B ranker -> 72B listwise ranker) explicitly chose its retrieval embedding model by highest recall rather than highest MAP; RSNA Cervical Spine 1st place's segmentation-then-classification cascade has its accuracy ceiling set entirely by the small-subset-trained segmentation stage's localization quality.

**Seen in.** llm-era, medical-imaging, code-efficiency

**Inverts when.** This inverts the usual instinct that each component should be tuned to its own best metric -- applied uniformly to a cascade's first stage, that instinct silently caps the whole pipeline, since a stage-1 model chosen for its own best ranking metric (not recall) will look locally optimal while quietly discarding cases no downstream stage can ever recover.


### 43. Adversarial validation (training a classifier to distinguish two data sources) is a diagnostic for which features differ between them, not a prescription for what to do about it.

**Why.** A high adversarial-AUC feature is definitely distributionally different between the two sources, but that says nothing about whether the difference is nuisance drift or genuine signal correlated with the target -- a separate, empirical question.

**Evidence.** Playground S6E5 1st place used a fellow competitor's per-variable adversarial AUC to flag 'Driver' as most-different, then treated dropping it as a CV/LB-verified bet, not an automatic action; Optiver's 1st place used adversarial validation to flag order_count and total_volume as drifting.

**Seen in.** tabular-classic, tabular-modern, time-series forecasting

**Inverts when.** Even when the 'obvious' fix is applied to a correctly-flagged feature, the payoff can be minimal rather than large -- Optiver's own rank-transform of its two adversarially-flagged features produced only a small LB improvement despite the diagnostic being unambiguous, showing flagged-and-fixed is not the same as flagged-and-valuable; the fix must still be validated on its own merits every time.


### 44. In a large stacking pool with a non-linear Level-2 meta-model, deliberately including model families with the weakest standalone CV is worthwhile purely because their errors are less correlated with the strong models, giving the meta-learner independent signal to extract.

**Why.** A non-linear stacker can learn arbitrary interactions between base-model predictions, including conditional corrections, a pattern a weak model's raw score never reveals but its error pattern still contains.

**Evidence.** Playground S5E4 1st place kept TabPFN (CV 13.2, tied for weakest of 12 model families against GBDT's 11.8) in the winning 75-model stack; the same competition's problem-reframing technique exists purely to manufacture more decorrelated, not necessarily individually-strong, Level-1 inputs.

**Seen in.** tabular-modern

**Inverts when.** This only pays off net of a genuinely non-linear meta-model. In a linear hill-climbing or Ridge-weighted blend, the same weak model is far more likely to be assigned a near-zero weight or actively drag the blend down, because a linear combiner has no way to condition its trust in a component on context the way a non-linear stacker can -- the S5E4 writeup states this distinction explicitly.


---

## Anti-patterns (28)

Each entry pairs the failure with the CHEAP CHECK that catches it early. Prefer running the check to trusting that you are too careful to need it.


### 1. Picking both final submissions by public-LB rank, with no CV-consistency check

**What happens.** Competitors treat their two allowed final submissions as "my two highest public-LB scores." Both picks are usually correlated (same recipe family, tuned the same way), so when the leaderboard reshuffles, both fail together — even though a genuinely better submission (by CV) sits unselected in their own submission history. No diversification against a shake-up ever happens because the selection criterion never included CV agreement in the first place.

**Where it bit.** bestfitting (profiled in 2018 as the world's #1-ranked Kaggle competitor, with wins including Planet: Understanding the Amazon from Space and Cdiscount Image Classification) states the discipline explicitly: always lock one slot to a conservative, fully-understood weighted-average ensemble and at most one to a calculated risk, and never select any submission — however high its public LB — that can't be explained; the same account credits this discipline for surviving 'a wicked leaderboard shake-up' while staying top-5 in Two Sigma Financial Modeling (2016-17), Kaggle's first code competition. Kawamata's 1st-place Playground Series S6E2 (2026) solution independently mapped several real submissions' CV against public LB across the competition and deliberately passed over his numerically highest-ever CV (0.955865) once that curve showed it no longer tracked LB, picking a lower-CV submission instead. The cost of skipping this is visible across many shake-ups: Mercedes-Benz Greener Manufacturing (2017, R² metric, dedicated public dataset documenting the leaderboard shakeup), PetFinder.my Adoption Speed (2019, teams moving from roughly 13th public to 30th private on quadratic weighted kappa), and Cassava Leaf Disease Classification (2020, competitors reporting swings of 400+ places between public and private).

**Cheap check.** Before the deadline, check whether your two selected finals are highly correlated with each other and were both chosen primarily by public rank. Separately, plot CV against public LB across your actual submission history over the course of the competition — a point where the relationship visibly bends or reverses is the signal that further CV-only or LB-only gains are no longer trustworthy.

**Fix.** Reserve one slot for the submission with the most trustworthy CV score (ideally from the region where your CV↔LB curve still agreed) and the other for a genuinely different recipe, not a correlated variant of the same one; never submit a blend you can't explain regardless of its LB rank, and spend some submission budget explicitly mapping the CV-LB curve rather than only chasing the top number on either axis.


### 2. Treating a small, high-variance public LB split as ground truth

**What happens.** The public leaderboard scores only a fraction of the test set, often on a metric that is itself unstable at that sample size (an ordinal/kappa metric with few positives, a hierarchically-weighted metric like WRMSSE, or an outlier-sensitive metric like R²). Competitors read 3rd-4th-decimal differences on that sliver as real signal, when the gap is well inside the metric's own sampling noise; the private board, scored on the full test set (sometimes even a different time period), then reorders the field by hundreds of places — not because anyone did anything wrong, but because the public number was never precise enough to support the decisions being made on it.

**Where it bit.** Mercedes-Benz Greener Manufacturing (2017) scored on R², a metric extremely sensitive to outliers, and produced a public-to-private reordering large enough that a dedicated public Kaggle dataset exists specifically documenting the shakeup. M5 Forecasting Accuracy (2020) trained/validated mostly at lower series levels while the actual WRMSSE metric weights heavily toward higher hierarchical aggregations; one publicly discussed result moved from public 0.48734 to private 0.62408 while still only placing 190th of 5,558 — i.e. nearly the whole field moved similarly, not just outliers. PetFinder.my Adoption Speed (2019), scored on quadratic weighted kappa, is documented with teams moving from around 13th on the public board to roughly 30th on the private board.

**Cheap check.** Before trusting any public-LB delta, estimate its noise: how many rows/positives is the public split actually scored on, and how much does the metric swing under bootstrap resampling of your own OOF predictions at that same sample size? If a leaderboard-visible gain is smaller than that resampling spread, it is noise, not signal — this is especially severe for ordinal/kappa metrics and hierarchically-weighted metrics.

**Fix.** Size every LB-based decision against the public split's real sample size and the metric's known instability rather than its face value; for hierarchical or ordinal metrics, implement and validate against the exact metric formula locally instead of trusting a training-grain proxy.


### 3. Chasing a host-side data-generation artifact instead of, or beyond checking, real signal

**What happens.** Something about how the host assembled the data — file metadata, row ordering, duplicated historical rows, a synthetic-row generation quirk — lets a feature reconstruct or closely approximate the label with little genuine modeling. Competitors who find it first gain a massive edge; competitors who don't are effectively competing in a harder, different competition than the leaderboard implies, and a change that exploits the artifact can look like a real modeling win while being nothing of the sort.

**Where it bit.** Draper Satellite Image Chronology (2016) had a leak where raw image file size alone, with no image content used, reached a public LB score around 0.30. Santander Value Prediction Challenge (2018) had a documented leak where roughly 16% of test rows (7,897 of 49,342) could be assigned their exact target value with full confidence, because a customer's historical values recurred as shifted columns elsewhere in the row structure — a property widely reported to have frustrated teams since it rewarded finding the trick over building a model. TGS Salt Identification Challenge 1st place (2018) documents the cautionary flip side directly: a mosaic-tile-adjacency exploit moved their public LB by +0.008 but their private LB by +0.000, and their own writeup flags that exact public-only divergence as 'diagnostic of leak exploitation... not a modeling win worth trusting.'

**Cheap check.** If a trivially simple, content-blind baseline (file metadata, row index, a single suspicious ID) scores far above what the stated task should allow, suspect a generation artifact. Whenever a change moves public LB but not a genuinely held-out/CV estimate (or vice versa), treat that divergence itself as diagnostic of exploiting something that won't hold, not a win to build further strategy on.

**Fix.** Verify any suspiciously strong feature's contribution against a held-out split with no possible exposure to the artifact before trusting it; when a leak is found, check explicitly whether the private test was generated by the same process (some leaks persist to private, most evaporate or get patched) rather than assuming either way.


### 4. Code-competition resource ceilings silently invalidating an otherwise-good pipeline

**What happens.** In kernel/notebook-only code competitions, a hard wall-clock, memory, or no-internet ceiling applies to the actual (often larger or differently-shaped) hidden test set, not to whatever smaller sample was used during local development. A pipeline that runs fine locally can time out, run out of memory, or fail to load a needed package silently during the real scored run, producing a failed or truncated submission despite strong offline metrics.

**Where it bit.** Mercari Price Suggestion Challenge (2018), Kaggle's first kernels-only competition, enforced a 60-minute runtime ceiling on 4 vCPUs with no GPU and a 16GB memory cap for the scored run against the full test set, constraints substantial enough to rule out heavier modeling approaches outright and require batch-wise rather than full in-memory inference. Later code competitions generalized the same trap: Feedback Prize - Predicting Effective Arguments' Efficiency track (2022) scored explicitly against a 32,400-second (9-hour) ceiling, and ARC Prize 2024's winning pipeline had to fit within a shared 12-hour budget across roughly 100 tasks, with the writeup noting that a few slow-converging tasks can silently starve the rest of the budget if not capped per-task.

**Cheap check.** Before relying on any code-competition score, confirm your pipeline's measured runtime/memory on a dataset sized like the ACTUAL hidden test set, not your local dev sample, including cold-start costs (model loading, package install from an offline-staged wheel) — and check that offline-staged package wheels actually match the scoring environment's CUDA/Python/glibc versions, since those can silently drift between when the dataset was built and when the notebook reruns.

**Fix.** Profile end-to-end against a full-size synthetic or held-out set early, build in a hard per-item/per-stage time budget with graceful degradation rather than hoping the ceiling is never hit, and re-verify offline dependency wheels against the current scoring environment shortly before the deadline.


### 5. Organized private-sharing / medal-selling fraud rings

**What happens.** Paid coaching operations, advertised openly on third-party marketplaces, hand customers ready-made models or submission files for kernel-only or CSV-submission competitions respectively, guaranteeing a purchased medal - a direct violation of no-private-sharing rules that clutters the leaderboard with fraudulent entries and can put implausibly novice-looking accounts in gold/silver range, sometimes via networks of dozens of accounts run by one operator across multiple competitions.

**Where it bit.** SIIM-ISIC Melanoma Classification, 2020. Thread "Evidence regarding private sharing" (343 votes, verified via Kaggle API): documented a Taobao storefront selling a guaranteed 1% finish (silver medal) for roughly $600, screenshots of an operation boasting "38 (!!) silver medals within one competition," and named Kaggle Masters whose accounts appeared in the advertisements while independently sitting at public-LB rank 18; the same operator's name was tied to "20 different accounts in the deepfake competition" (Deepfake Detection Challenge). The thread's author, a Grandmaster, added a first-person cautionary tale: "I might have blood on my hands... I teamed up with someone that seemed very trust-worthy... based on previous competition results & current LB standing. After teaming up, this person went completely silent... We also suspect the profile picture used to be fake."

**Cheap check.** Treat a prospective merge partner's displayed rank/medal history as necessary but not sufficient evidence of real skill; a cluster of low-history accounts sitting anomalously high near medal cutoffs, or a partner who goes silent right after merging, are both flaggable patterns.

**Fix.** Verify a merge candidate's actual technical contribution (their code, their approach, genuine forum engagement) before merging rather than merging on LB position alone; report suspicious clustering to the host/Kaggle for cross-checking against the Meta Kaggle dataset before prizes are finalized.


### 6. CV folds that don't reproduce the real train→test time gap

**What happens.** A time-ordered dataset is split with a plain walk-forward or contiguous scheme (train through month N, validate on month N+1) when the actual test set sits a genuine gap after training data ends. Recency-window features are fresher in a contiguous CV split than they can ever be against the real test set, so CV systematically overstates generalization — gap-induced feature staleness/drift never shows up in validation at all.

**Where it bit.** IEEE-CIS Fraud Detection 2nd place (2019, CPMP with team) built an explicit month-indexed expanding-window scheme with a skipped buffer specifically 'to mimic the fact that there is a significant time gap between train and test,' stating a naive contiguous split 'systematically overstates generalization.' The same purge/embargo logic anchors Jane Street Market Prediction 1st place (2021, 31-group purge gap) and Ubiquant Market Prediction 1st place (2022) — whose own team deliberately downgraded to plain grouped KFold for their FINAL production fit only, after confirming purge/embargo cost too much usable training data, showing even the fix needs a conscious trade-off rather than blanket application. Zillow Prize's two-round structure (2017-2019), which evaluated the final round against real home sales up to a year after model freeze, is a structural example of the same underlying trap at competition-design scale.

**Cheap check.** Compare the time distance between your CV's training cutoff and its validation window start against the actual (or best-estimated) gap between the real training data's end and the real test period's start. A CV gap of zero when the true gap isn't zero is the signature.

**Fix.** Build expanding-window or purged/embargoed CV folds with an explicit skipped buffer matching the true production gap for feature-engineering and model-selection decisions, while consciously deciding whether the final production fit can afford to relax that discipline given the training-data cost.


### 7. Optimizing a proxy loss instead of the real hierarchical/ordinal/weighted competition metric

**What happens.** Teams train against a convenient standard loss (plain BCE, RMSE, accuracy) and only check the actual competition metric at evaluation time. When the real metric has structure the proxy doesn't respect — an ordinal weighting (quadratic weighted kappa), a hierarchical aggregation with different weights than the training grain (WRMSSE), or a decomposed fairness/subgroup formula — the proxy-optimized model systematically underperforms what the metric actually rewards, and any threshold/rounding tuned against the proxy overfits the wrong decision surface.

**Where it bit.** M5 Forecasting Accuracy (2020) trained largely at lower series levels while WRMSSE evaluation weights heavily toward higher hierarchical aggregations, a structural mismatch widely cited as a major driver of that competition's public/private divergence. PetFinder.my Adoption Speed (2019), scored on quadratic weighted kappa, saw threshold/rounding choices tuned against local CV fail to hold on the private board, part of the documented public-to-private movement. Jigsaw Unintended Bias in Toxicity Classification 1st place (2019, ods.ai) instead implemented the metric's exact subgroup/BPSN/BNSP decomposition as the training loss itself, specifically to avoid this mismatch — that this was worth building custom loss code for is itself evidence of how much a generic proxy loss leaves on the table.

**Cheap check.** Before trusting any CV number, implement the exact competition metric (not an approximation) locally and confirm it reproduces known public-LB scores from your own submission history. Check explicitly whether the metric decomposes hierarchically or ordinally in a way your training loss doesn't mirror.

**Fix.** Where feasible, build a loss/objective that structurally mirrors the real metric's formula; always validate any threshold/rounding search on genuinely held-out folds distinct from the data used to fit those thresholds.


### 8. Batch/session/plate-conditional shortcut learning against an entirely-unseen-batch test set

**What happens.** When training data is collected in discrete technical batches (microscopy plates, scanner sessions, acquisition runs) that carry systematic pixel/signal statistics unrelated to the true label, a model can partly key off batch-specific artifacts rather than the underlying signal — a shortcut that costs nothing in ordinary K-fold CV, where batches typically recur across folds, but collapses once evaluated on test batches the model has literally never seen, because there is no batch-specific artifact left to exploit there.

**Where it bit.** Recursion Cellular Image Classification (2019) was explicitly structured around this trap: images were generated in 51 discrete experimental batches, with competitors given only 33 for training and the remaining 18 batches held out entirely for test — a design whose stated purpose was to force models to separate biological signal from experimental/batch noise. The same family of failure (a held-out grouping unit that ordinary row-level CV doesn't respect) is what motivates GroupKFold-by-patient in the medical-imaging competitions above. Note: this entry documents the competition's structural premise and the general mechanism rather than a single named team's postmortem quantifying a specific rank collapse from it — treat the risk as well-established, not a one-off anecdote.

**Cheap check.** Whenever training data was collected in discrete sessions/batches/plates/scanners, check whether your CV folds ever hold out an ENTIRE batch, not just a random sample of rows from every batch. If every batch appears in every fold, your CV cannot detect batch-shortcut learning at all.

**Fix.** Build at least one CV variant that holds out whole batches/sessions to specifically test for this failure mode, and prefer per-batch normalization, strong augmentation, or domain-adversarial training that discourages the model from keying off batch identity.


### 9. Group/entity leakage across folds — and across the host's own public/private split

**What happens.** Multiple rows share a natural entity key (patient, customer, essay, question, driver) whose identity correlates with the label independent of the features you intend to learn from. Plain row-level K-Fold lets the same entity appear in both a training fold and its paired validation fold, so the model partly memorizes entity-specific patterns rather than the general relationship; CV looks trustworthy right up until deployment against entities the model has genuinely never seen.

**Where it bit.** NBME - Score Clinical Patient Notes 1st place (2022) reports the switch 'from 10 folds to GroupStratifiedKFolds [by patient note]... has been a huge [improvement],' implying the ungrouped version was materially misleading. Google QUEST Q&A Labeling 1st place (2020) lists GroupKFold by question/title as literally their first baseline-improving trick. In medical-imaging competitions generally (the RSNA family, SIIM-ISIC melanoma), the same patient can supply multiple images; documented reports around the ISIC skin-lesion data note that cross-split duplicate images meant 'using a GroupKFold wasn't sufficient' on its own once exact duplicates straddled train and test, showing even the standard fix needs a data-integrity audit behind it.

**Cheap check.** For every dataset, explicitly check for any column identifying a real-world entity spanning multiple rows — not just an obvious \"patient_id\" but derived proxies like device/IP/session. Verify the grouping key is fully contained within one fold, and separately audit for exact/near-duplicate rows that could straddle the host's own public/private or train/test boundary.

**Fix.** Always split by the correct leakage unit (GroupKFold or multilabel-stratified group fold) rather than by row; when pooling in any external dataset, deduplicate across the entire combined pool by content hash, not just by ID field, before trusting any split.


### 10. Re-identifying anonymized host data and exploiting it unevenly

**What happens.** When a host anonymizes real-world entities (buildings, sites, individuals) but leaves enough structural detail (square footage, year built, time zone, weather correlation) for participants to re-identify the true source via public web search, some entities become fully solvable while most of the dataset remains genuinely modeled. Tuning how aggressively to substitute the re-identified true values - especially against the public LB - produces a leaderboard position untethered from actual model quality on the non-leaked majority, so the ranking inverts hard once the private split is revealed.

**Where it bit.** ASHRAE - Great Energy Predictor III, 2019. Thread "Sites, buildings identified by internet search" (106 votes) identified real universities/buildings (site 0 = University of Central Florida, etc.) via public building-energy PDFs; a participant remarked "This became an Internet search competition." The resulting shakeup is named directly in a top writeup titled "[497th place shake down solution] PUBLIC LB 1st place solution" (104 votes) - the team that led the PUBLIC leaderboard finished 497th PRIVATE. That 497th-place author used the leak only conservatively for validation on confirmed sites and stated "no other methods to exploit leakage," implying the public #1 team over-exploited it.

**Cheap check.** If any part of a score improvement depends on external, non-provided sources matched to specific host rows/entities, check what fraction of the total test set that actually covers - a leak touching only a handful of identifiable entities cannot support a leaderboard position that assumes it generalizes.

**Fix.** Report re-identification leaks to the host immediately rather than quietly exploiting them; if used at all, restrict it to out-of-fold validation of your model rather than a direct prediction substitute, and cap its influence in any final blend.


### 11. Exfiltrated test labels disguised as external data

**What happens.** A team obtains true private-test answers through illegitimate means (e.g., scraping a host's public-facing website) and hides that information inside a legitimately-permitted external dataset - encoding, obfuscating, and hashing the answers into an ID field disguised as unrelated content - then decodes them at inference time under deliberately unreadable, deeply nested processing code, using only a subset of the recovered answers to keep the final score plausible enough to avoid suspicion.

**Where it bit.** PetFinder.my Adoption Prediction, 2019. Official host announcement "PetFinder.my Contest: 1st Place Winner Disqualified" (321 votes, verified via Kaggle API): the winning "Bestpetting" team, including a Kaggle Grandmaster, "fraudulently obtained adoption speed answers for the private test data (possibly by scraping our website)... encoded, obfuscated and hashed into an ID field that was disguised as part of their external 'cute-cats-and-dogs-from-pixabaycom' dataset... processing codes were meticulously hidden and obfuscated under many nested layers of functions and codes." The scheme was caught by a fellow top finisher (Benjamin Minixhofer) after the competition closed; the Grandmaster was permanently banned and prize money was later refunded.

**Cheap check.** When a solution's external-data justification doesn't obviously explain the size of its score jump, or its data-processing code is unusually obfuscated relative to its stated purpose, treat that as a red flag worth independent review - exactly what caught this case.

**Fix.** Keep external-data provenance and processing fully transparent and reproducible as a competitor; as a host, restrict or manually audit external-dataset claims that could plausibly encode host-website content, and budget time after the deadline for forensic code review of top finishers before finalizing prizes.


### 12. Near-total leak from a naive train/test split methodology

**What happens.** When a host builds a test set by randomly holding out rows from an otherwise grouped or time-ordered dataset instead of holding out entire groups or future periods, a trivial groupby aggregation (e.g., mean outcome by categorical group and date) can reconstruct almost the entire test target, reaching near-ceiling metric scores with no real predictive modeling. Once published, the competition collapses into a race to copy the leak kernel rather than a modeling contest.

**Where it bit.** Predicting Red Hat Business Value, 2016. Thread "~0.987 Kernel now available - seems like leakage" (59 votes): "random-sample-from-time-series approach of this competition gives a considerable data leakage... give each date and group_1 individual outcome score based on training, and assign those scores... to test" - a single groupby trick reached ~0.987 AUC, close to the eventual winning score, and was voluntarily published specifically "as we believe this competition should be more about actual prediction improvement, and not clever tricks based on imperfect train/test split methodology." A companion thread argued "Kaggle should seriously think if they formally allow probing + hand labeling."

**Cheap check.** Sanity-check your own model's score against a trivial groupby/aggregation baseline on visible categorical+date columns alone - if a near-ceiling score is reachable with almost no modeling, the competition's real signal is the split methodology, not your features.

**Fix.** As soon as a near-total leak surfaces, treat leak-derived features as inputs the model can learn to weight rather than a hand-coded override, so your submission degrades gracefully if the host patches the split or filters the target before final scoring; escalate the leak to the host rather than keeping it private.


### 13. Leaderboard probing as a mathematical label-extraction oracle

**What happens.** On metrics that decompose per-feature or per-example (like AUC over independent, standardized input variables), the score returned by a single-variable submission can be algebraically inverted to recover that variable's true linear coefficient almost exactly. With enough submissions - one or a few per candidate variable - a competitor can reconstruct a strong linear classifier almost entirely from leaderboard feedback rather than from a small labeled training set, especially when the training set is too small for legitimate modeling to compete.

**Where it bit.** Don't Overfit II, 2019 (250 training rows, ~19,750 test rows, AUC metric - deliberately built to teach this). Chris Deotte's thread "The Mathematics of AUC/LB-Probing" (47 votes) derives the exact formula (a_k = 1/sqrt(2-2*AUC) - 1 for AUC near 0.5-0.7) and cites a public kernel using it that scored 0.890 (his own 2nd place). The competition's 1st-place writeup is framed directly around this: "Careful leaderboard probing was the key." A 4th-place team explicitly branded their approach "[no LB probing]" to distinguish it from the norm.

**Cheap check.** Compare submission budget to test-set size and the metric's decomposability; if (available submissions) x (information per submission) approaches the entropy of the hidden labels, the public score is not a trustworthy skill signal for anyone - check whether public kernels are already extracting per-feature LB scores as coefficients.

**Fix.** When this regime is recognized, treat leaderboard rank as compromised and prioritize a validation scheme that owes nothing to submission feedback; as a host, cap submissions and avoid metrics that linearly decompose across a small, independent feature or example set a modest submission budget can fully probe.


### 14. Small discrete answer space turns the leaderboard into a solve-by-hand oracle

**What happens.** When the true answer is drawn from a small, enumerable set per group (e.g., ordering a handful of images chronologically) and overlapping or linked answer-groups exist between the public and private test portions, a competitor can manually deduce the correct answer for one group, infer it holds for every linked group, and use the public leaderboard score itself to binary-search or confirm guesses - turning the task into manual puzzle-solving that needs no real model.

**Where it bit.** Draper Satellite Image Chronology, 2016. Forum thread "Leak in the dataset": "There are only 274 sets and 60 days of competition, manually solving 4-5 sets a day is more than possible, it's inviting... if you have 10-20 sets that are linked together then you can assume that at least one set will be in the public LB fold, so you can easily check the correct sequence by checking the public LB score." The community seriously discussed early closure ("Should Draper close the competition?"), and the eventual 1st-place writeup is titled "How to win the competition if you know nothing about image processing."

**Cheap check.** Compare the total number of distinguishable answers in the competition (permutations, orderings, discrete classes per group) against the available submission budget - if a human could plausibly hand-solve or brute-force-probe a meaningful fraction of it within the submission limit, the leaderboard is not a fair skill measure.

**Fix.** As a host, shrink the fix to the format itself (larger/less-overlapping answer space, tighter submission caps, held-out-only scoring) before or during launch; as a participant, recognize when a strategy is exploiting the answer space rather than the data, and discount any resulting rank accordingly.


### 15. Trusting a discovered leak without stress-testing its scope

**What happens.** A competitor finds a genuine row-matching pattern in "anonymized" test data that appears to reveal target values directly, and swaps their model to trust the leaked values wholesale. Hosts sometimes filter or clip the target range, or the leak mechanism silently injects poisoned/zero values for a subset of matches indistinguishable from genuine ones without careful validation — so blindly trusting every "leaked" row can be dramatically worse than not using the leak at all.

**Where it bit.** Santander Value Prediction Challenge, 2018. ianlini's thread "Dropped from 2nd to 3566th" (73 votes): found 50 highly-confident leaked rows, trusted them wholesale, and fell from a projected top-3 finish to private LB 1.62 (rank 3566) - called out in-thread as likely "the biggest LB drop... I've seen them in 2000s before but not in 3000s" (referencing Mercedes-Benz Greener Manufacturing's earlier ~2000-place public-LB-leader collapse). Removing only the zero-valued leaked rows recovered the score to 0.62; his leak-free models scored 0.53. Giba (eventual co-winner) explained the mechanism directly in-thread: the host's target field was filtered to a fixed numeric range (30000-40000000), and out-of-range leaked matches were poisoned/unscored.

**Cheap check.** Validate every leaked/matched value against an independent signal (does it fall inside the plausible/observed target range? does removing a subset of matches improve or hurt held-out CV?) before trusting it fully; a leak discovery should raise, not lower, your scrutiny.

**Fix.** Treat leak-derived values as one noisy, confidence-gated blend component, not a wholesale substitute for the modeled prediction; always keep a leak-free model as a fallback and blend rather than override.


### 16. Point-mass outlier chasing via public-LB "override" tuning

**What happens.** When the target has a rare, extreme, near-constant outlier subpopulation (a sentinel value), competitors reverse-engineer which specific entities are outliers by submitting classifier-threshold variants and reading the public-LB score delta per tweak. Each single hand-picked "override" can move the public score by more than the gap between medal cutoffs, so public rank ends up dominated by a handful of manually-tuned guesses about a handful of rows rather than genuine model quality. Because the true outlier membership differs between the public and private test splits, those overrides don't transfer, and public-LB leaders collapse on reveal.

**Where it bit.** Elo Merchant Category Recommendation, 2019. raddar's Kaggle forum thread "Shakeup incoming!" (60 votes, verified via Kaggle API): "having a single override can make a 0.003-0.004 score difference... current public LB gold position... are within 3 overrides range." Confirmed in the same thread: "The shakeup was so huge that the 1st place PL[public leaderboard leader] finished just behind me, down 1412 places," and "the 2nd place team ended down at 2460."

**Cheap check.** Before trusting a public-LB jump, check whether it came from a broad feature/model change or from a small number of manually-flagged special-case rows; if removing any single override changes your score by more than the medal-cutoff gap, your rank is noise, not signal.

**Fix.** Model the outlier subpopulation as its own gated classifier trained and validated with proper CV (classify "is this the sentinel value," then route to a separate regressor) rather than hand-tuning individual IDs against the public score; trust local CV stability, not per-row LB deltas.


### 17. Trusting raw anonymized/ID-like features whose values differ between train and test

**What happens.** Anonymized or hashed categorical columns (card IDs, device IDs) commonly contain many values that appear only in train or only in test, simply because IDs are assigned over time or per-population. A GBDT given the raw value learns spurious per-ID structure from train that cannot transfer to test's mostly-unseen values, producing a model that looks strong in CV (where train-only IDs still recur across folds) and degrades once scored on the real test set.

**Where it bit.** IEEE-CIS Fraud Detection 2nd place (2019, CPMP/Jean-François Puget with team) diagnosed this directly: 'We see that lots of card1 values only appear in test. If we use it directly it will lead to major drop' — the fix (frequency-encoding instead of the raw ID) took his solo, feature-selected model to 0.942 public LB before UID-based features were even added.

**Cheap check.** For every candidate feature, especially anonymized/hashed/ID-shaped columns, plot its value-frequency distribution in train overlaid against test, or run adversarial validation across all features at once rather than eyeballing a handful. A feature where a large share of test's mass sits on values rare-or-absent in train is a red flag regardless of how much it helps CV.

**Fix.** Replace raw high-cardinality/ID-like categoricals with frequency encodings, counts, or other transforms whose train/test distributions actually overlap; formalize the screening with adversarial validation once there are more than a handful of suspect columns — but confirm any drop actually helps CV/LB rather than dropping reflexively, since a drifting feature isn't automatically useless.


### 18. Target/count encoding computed without strict out-of-fold isolation

**What happens.** A categorical column is replaced by the mean/count of the target within that category (or a KNN-based local target mean) computed once over the whole training set rather than fold-by-fold. The feature then partially encodes the label directly; local CV looks excellent because validation rows still see target information smuggled in through their category's global encoding, and the model degrades sharply once deployed on genuinely unseen categories or once the leak is removed.

**Where it bit.** Home Credit Default Risk 1st place (2018, Bojan Tunguz + team)'s KNN local-target-mean feature is documented as needing computation 'strictly out-of-fold (train-fold-only neighbor targets) or it leaks the label into near-duplicate rows.' Kaggle Playground Series S4E12 1st place (2024, Chris Deotte) states 'every TE/CE computation must be nested-fold or it leaks target info directly into features.' Kaggle Playground Series S5E2 1st place (2025, Deotte) explicitly flags target-derived groupby features as 'a textbook leakage vector without nested/out-of-fold logic.'

**Cheap check.** Recompute the same feature two ways — once leaking (fit on all rows including the validation fold) and once strictly out-of-fold — and compare the CV delta. A large CV gain that mostly disappears under strict OOF computation is the signature; a CV gain with no matching public-LB gain is a secondary tell.

**Fix.** Always compute target/count encodings inside the same CV loop used for the model itself (fit on train-fold only, apply to validation-fold), never on the full dataset before splitting.


### 19. Over-parameterized blend-weight optimization overfitting the blend itself

**What happens.** Ensemble weights are optimized directly against OOF predictions using a numerical optimizer with many free parameters — per-model weights, further split into per-prediction-range or per-target sub-weights. Each extra degree of freedom is effectively one more model fit to the same limited OOF sample; the resulting blend can look excellent on both CV and public LB and still be measurably worse than a plainer, globally-weighted blend once scored on private LB, because the optimizer quietly fit fold-specific noise rather than genuine complementary error patterns.

**Where it bit.** Allstate Claims Severity 7th place (2016, Gilberto 'Giba' Titericz) reports this from his own pipeline: a globally-optimized blend (scipy.optimize.minimize/Nelder-Mead against OOF predictions) 'worked very well with CV and public LB,' but after adding a further segmentation of the blend-weight search across prediction ranges, the segment-wise version 'overfitted a little Private LB' relative to the simpler global blend.

**Cheap check.** Count the effective free parameters in your blend optimizer against the number of independent OOF rows/folds backing it. If an elaborate segmented/negative-weight blend's edge over a simple global-weight blend disappears or reverses on a genuinely held-out slice, that's the overfitting signature.

**Fix.** Prefer the fewest blend parameters that materially help; treat any optimized blend as a model that itself needs its own validation split, not something safe to fit directly on the same OOF pool used to select which base models to include.


### 20. Transductive leakage from fitting scalers/normalizers on train+test combined

**What happens.** A rank transform, standard scaler, PCA, or other preprocessing step is fit on the concatenated train+test dataframe before any CV split exists, so every row's transformed value is influenced by statistics that include test rows (and, inside CV, other folds). No label is involved, so the leak is easy to miss, but it still inflates apparent generalization because the transform itself has 'seen' the distribution it will later be evaluated against.

**Where it bit.** Porto Seguro's Safe Driver Prediction 1st place (2017, Michael Jahrer)'s RankGauss normalization technique — central to the winning NN ensemble — is documented with the explicit caveat: 'fit the rank mapping on train only and apply it consistently to test to avoid transductive leakage.' The same competition's denoising-autoencoder representation, trained unsupervised on combined train+test, illustrates the boundary case: legitimate when done before any label is touched, but the exact mechanism to audit whenever preprocessing precedes a CV split.

**Cheap check.** Grep your feature-engineering code for any `fit()`, `.mean()`, `.rank()`, or similar statistic computed on a dataframe that includes both train and validation (or train and test) rows before a split. If preprocessing runs before your CV-split code, assume this leak until proven otherwise.

**Fix.** Fit every scaler, rank transform, or dimensionality reduction on train-fold data only, then apply (transform, not refit) to validation/test; wrap preprocessing inside the same fold loop as the model, not as a one-time step beforehand.


### 21. Extending ID reconstruction into causally-impossible lag-stacked features

**What happens.** After legitimately reconstructing a hidden entity ID from anonymized columns to compute simple aggregates (counts, means), a team pushes further and feeds the model's own predictions back through time via that same synthetic ID (a lagged-prediction feature). Because the reconstructed ID is imperfect and the 'lag' partly reflects information only available after the fact within the training/CV setup, this produces a large CV improvement that cannot be realized at real inference time, where a future row's true label isn't available yet.

**Where it bit.** IEEE-CIS Fraud Detection (2019) — both the 1st place team (Konstantin Yakovlev, Chris Deotte, and team) and the 2nd place team (CPMP/Jean-François Puget with team) independently converged on reconstructing anonymized entity IDs for aggregation features, and both independently stopped short of lag-stacking predictions on those IDs; the documented reasoning states that further extension 'is the exact failure mode: leakage-shaped CV inflation with no LB payoff.'

**Cheap check.** For any candidate feature, ask whether it could be computed in a genuine forward-only production/scoring setting with no access to information not yet available at scoring time — including the model's own future outputs. If not, treat any CV gain it produces as suspect regardless of size.

**Fix.** Use reconstructed IDs for simple, causally-available aggregates only (counts, historical means up to the row being scored); do not feed a model's own predictions back through synthetic entity links across time.


### 22. One-off searches against a small holdout treated as free (selection overfitting)

**What happens.** A team searches over a very large number of candidate configurations, feature subsets, or blend pairs against one fixed, modestly-sized holdout, then trusts the winning configuration as if the holdout were unlimited. Even though each individual candidate model was trained honestly, the selection process itself has many effective degrees of freedom and can overfit the holdout's specific noise, especially when the holdout period is short or unrepresentative.

**Where it bit.** Rossmann Store Sales 1st place (2015, Gert Jacobusse) ran over 500 randomly feature-subsetted XGBoost models and systematically searched roughly 500×250 candidate pairings against a single validation holdout that was itself only six weeks of data, later stating he was 'surprised' the selected weights held up rather than having overfit that holdout — an outcome he frames as fortunate, not guaranteed.

**Cheap check.** Count how many distinct configurations/combinations were actually scored against your holdout over the course of a competition; if it's in the hundreds or thousands against a holdout of only weeks or a few thousand rows, treat the final selection as at meaningful risk of holdout-overfitting regardless of how good any individual candidate looked in isolation.

**Fix.** Where possible, split the search itself across an outer/inner holdout (search on one, confirm on a second untouched one), or explicitly budget how many candidate evaluations the holdout is allowed to absorb before trusting a selection made against it.


### 23. Pretraining a shared representation before the CV split exists

**What happens.** An autoencoder, embedding model, or other unsupervised representation is pretrained once on the full train+test pool, and only afterward is a supervised head trained with K-fold CV on top of the frozen representation. Because the encoder's weights were shaped by seeing every validation fold's raw features during pretraining, the downstream CV score is inflated relative to genuine held-out performance — invisible because no label was used in pretraining, only the rows' features.

**Where it bit.** Jane Street Market Prediction 1st place (2021) built the autoencoder and downstream MLP as one graph, retrained fully from scratch inside every fold, specifically because 'pre-training the AE once before the CV split... leaks validation-fold info through the encoder and inflates CV' — describing exactly what the public kernels of the era were doing. The single-model AE-MLP built this way scored 6022.202 on the private leaderboard and would independently have placed 1st with no ensembling.

**Cheap check.** Check whether any unsupervised preprocessing step (autoencoder, PCA fit, clustering, embedding) was fit on rows that later appear in a validation fold. If your feature-engineering script runs before your CV-split script, you likely have this leak regardless of whether target information was involved.

**Fix.** Treat unsupervised pretraining as part of the model, not part of preprocessing — refit it from scratch inside every fold on train-fold rows only, accepting the extra compute cost.


### 24. Pseudo-labeling with a model that saw the fold it's labeling

**What happens.** A team generates pseudo-labels for extra or external data using their full trained ensemble (or any model fit using a given validation fold), then adds those pseudo-labels back into training data touching that same fold. Local CV on that fold rises because the fold's own model effectively re-labeled data using its own partially-memorized judgment, but the private leaderboard doesn't confirm the gain because no new information actually entered the system.

**Where it bit.** Google QUEST Q&A Labeling 1st place (2020, team Bibimorph) diagnosed and quantified this directly: leak-free, fold-consistent pseudo-labeling moved CV 0.414→0.422, while the leakier version (pseudo-labels from an ensemble that included models which had seen the fold) moved the same starting CV to 0.414→0.445 — a number 'the leaderboard did not agree with.' Their own review calls this 'the single most common failure mode reported' in pseudo-labeling across competition writeups.

**Cheap check.** For every pseudo-labeling round, check which fold(s) each pseudo-label-generating model was trained on versus which fold's training data it's being added to — any overlap is the leak. A CV jump from a PL round noticeably larger than the matching public-LB jump is the empirical tell.

**Fix.** Generate K separate pseudo-label sets, one per fold, each produced only by models that never saw that fold — fold-consistent, leak-free pseudo-labeling — even though it costs K times the inference work.


### 25. Ungrouped cross-validation leakage from duplicate/linked entities

**What happens.** When the same real-world entity (e.g., a patient) contributes multiple near-identical images or rows, a plain random K-fold split lets duplicates of the same entity land in both the training and validation folds. The model partially memorizes that entity instead of learning transferable signal, so local CV looks deceptively strong; the true, properly-grouped private test set contains no such leakage, and the CV-to-LB relationship silently breaks in a way that's easy to miss until the private reveal.

**Where it bit.** SIIM-ISIC Melanoma Classification, 2020. Sticky thread "True duplicates in this dataset" (65 votes) and the widely-adopted community fix "Triple Stratified Leak-Free KFold CV" (384 votes) were built specifically because naive K-fold leaked across patients/duplicates. Chris Deotte's own competition writeup is titled "21st Public - 53rd Private - Trust Your CV," directly naming the CV-vs-LB gap this dataset structure produced.

**Cheap check.** Before trusting CV, hash rows/images to check for exact or near-duplicates, and check for any natural grouping key (patient/customer/molecule ID); if duplicates cross fold boundaries under the current split, the CV number is inflated by memorization, not generalization.

**Fix.** Use GroupKFold (or a stratified-group variant) keyed on the real-world entity, and confirm it is leak-free by checking that no entity or duplicate hash appears in more than one fold, before trusting any CV-based decision.


### 26. Fold-mismatched multi-level stacking

**What happens.** A Level-1 pool of models is trained on inconsistent CV folds (different seeds, different splitting logic, or early-stopping that implicitly leaks validation-fold information beyond just the stopping decision). The Level-2 meta-model, trained on the resulting out-of-fold predictions, learns to exploit these small inconsistencies as if they were real signal — inflating CV further while adding nothing, or actively hurting, on the private leaderboard.

**Where it bit.** Otto Group Product Classification 1st place (2015, Titericz & Semenov, 33 L1 models → 3 L2 meta-models → weighted L3 blend) states 'every L1 model must share identical, leak-free CV folds... or L2 learns to exploit leakage instead of learning genuine complementary signal.' Playground Series S6E2 1st place (2026, Kawamata) independently confirms 'fold-matching is easy to violate silently — even Ridge stacking overfits if OOF-generation folds and meta-model folds diverge.'

**Cheap check.** Audit that every Level-1 model and the Level-2 meta-model share the exact same fold-assignment array (same seed, same split object), not merely 'K-fold with K=5' reimplemented per model or notebook. Confirm no model's early-stopping used information beyond just the stopping round.

**Fix.** Instantiate one CV-split object at the very start of the competition and reuse it by reference in every training script; never regenerate folds per model.


### 27. Judging a technique from a single seed/fold run

**What happens.** A feature, architecture tweak, or hyperparameter change is kept or discarded based on one run's CV/LB delta. For high-variance metrics (top-k ranking metrics like MAP@k, small-positive-count metrics, small datasets) the run-to-run noise from just the random seed or fold split can be comparable to or larger than the observed 'improvement,' so the keep/drop decision is effectively being made by chance while looking like a real ablation.

**Where it bit.** Kaggle Playground Series S5E6 (2025, Chris Deotte, 1st place) cites a public-notebook demonstration that averaging 100 separately-seeded 5-fold XGBoost runs (500 model fits total) lifted MAP@3 from an average of 0.376 per single fold-run to 0.380 combined — meaning any single seed's read on 'is this feature good' already carries noise of a magnitude comparable to many real feature-engineering gains reported elsewhere in the same competition family.

**Cheap check.** Before crediting a change, check whether its reported gain exceeds the seed-to-seed/fold-to-fold variance obtained by simply rerunning the identical unchanged configuration multiple times. If the 'improvement' sits inside that noise band, it isn't established.

**Fix.** For high-variance metrics or small datasets, default to multi-seed (and/or multi-fold) averaging as the baseline comparison unit for any keep/drop decision, not a single run.


### 28. Greedy hill-climbing ensemble selection overfitting the OOF pool

**What happens.** An automated forward-selection ('hill climbing') process repeatedly adds whichever candidate model most improves the blended OOF CV score, with no penalty for how many candidates were tried. With enough candidates, some of any CV gain is just fitting noise in that specific K-fold split; the ensemble keeps 'improving' on CV indefinitely while the corresponding private-LB score stalls or falls.

**Where it bit.** Kaggle Playground Series S5E12 (2025, Chris Deotte, 1st place) reports directly that in that competition 'every further [hill-climbing] addition raised CV but lowered public LB.' The companion Playground S5E3 (2025, 2nd place) case is described as a directly evidenced instance of CV-driven over-ensembling backfiring: three extra models raised CV from 0.898 to 0.900-0.901 but diluted the private LB score.

**Cheap check.** Log the CV delta AND an independent check (a held-out slice never used in hill climbing, or public LB) at every hill-climbing step. A step that raises CV while the independent check stays flat or drops is the signature, especially once the candidate pool is in the hundreds.

**Fix.** Stabilize the search (e.g. Ridge-weighted hill climbing rather than raw greedy averaging), cap ensemble size, and require agreement on an independently-held check before accepting an addition rather than trusting CV alone.

