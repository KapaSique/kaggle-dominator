# Grandmaster profiles — how the named winners actually operate

Person-level craft mined from interviews, writeups and public notebooks. A profile is a lens: when your competition resembles the fronts a person dominated, read their entry and steal the OPERATING STYLE, not just the tricks. Verified alongside `gm-methods.md`.


---

## Chris Deotte (kaggle.com/cdeotte, handle cdeotte) — Kaggle Grandmaster in Competitions (rank_current #14 all-time, rank_highest #4; 26 gold / 29 silver / 10 bronze competition medals), simultaneously Grandmaster in Notebooks (#3 all-time, 130 gold notebook medals) and Grandmaster in Datasets (#44 all-time, 16 gold), plus Discussions Legacy Grandmaster (rank #1 highest, before that track was retired in 2025). Data Scientist & Researcher at NVIDIA (San Diego, CA). 98 competitions entered, 74 published writeups, one of only 17 Kagglers ever awarded Kaggle's official 2023 Best Solution Writeup Award — recognized for 3 separate writeups simultaneously (GoDaddy 3rd, AMP-Parkinson's 4th, Novozymes 968th). All facts confirmed directly via his Kaggle profile API record.

Treats EDA as active reverse-engineering of the data-GENERATING process rather than passive plotting — histogram-of-means to recover a synthetic generator's hidden cluster count, digit-decomposition to recover an ID-like encoding hidden in a float column. Runs local CV and the public leaderboard as two separate, deliberately cross-checked information channels, and when local CV is structurally impossible to build cleanly (a single held-out test enzyme, a one-shot forecast window), is willing to actively extract information from the leaderboard itself via minimal, carefully-designed probing submissions — in one case running a probing optimization continuously for 3 months. Treats iteration speed as a first-class strategic resource rather than an implementation detail: RAPIDS cuDF/cuML on GPU is a near-constant across his competitions specifically so he can test 'dozens of models in minutes' or 'over 300 XGBoost models... thousands of different feature engineering ideas' in a month, turning brute-force search into a viable core strategy rather than an afterthought. Documents almost everything close to real time (74 published writeups, a standing habit of posting technique-focused discussion posts DURING live competitions), which compounds into reusable personal infrastructure — most visibly a 6-month cadence of banking one new named technique per Playground Series entry so later months assemble winning solutions from previously-banked pieces. Documents his own mistakes with the same rigor as his wins — publishing a full public post-mortem on the Novozymes submission he did NOT select that would have placed 1st private.

**Signature moves.**

- Histogram-of-means / EDA-driven reverse-engineering of a synthetic data generator's hidden hyperparameters
- Public-LB probing to recover hidden split boundaries and/or pseudo-labels when local CV is structurally infeasible
- Reframe absolute targets as period-over-period ratios/multipliers, calibrated with a probed constant
- Random-noise-column / random-submission null-hypothesis tests to floor-check whether a feature or model beats chance
- RAPIDS cuDF/cuML on GPU as a default speed multiplier enabling brute-force feature/model search (hundreds of models or thousands of FE ideas per competition)
- One-vs-rest binary Level-1 decomposition + cross-entropy Level-2 recalibration for multiclass ranking metrics
- Deliberate, explicitly-justified inclusion of a revealed-prefix 'test leak' in aggregate features when it mirrors real inference-time information
- Public, near-real-time documentation of individual techniques as standalone discussion posts, compounding into a personal reusable toolbox across competitions

**Study these.** GoDaddy Microbusiness Density Forecasting 3rd/Gold + AMP-Parkinson's 4th/Gold + Novozymes 968th/Gold writeups (all 3 named on his single 2023 Best Solution Writeup Award badge — kaggle.com/cdeotte profile record); Instant Gratification 'How to Score LB 0.975' (7th place solo, 109 votes); OTTO 'How To Build a GBT Ranker Model' (335 votes, his highest-voted OTTO discussion post, adopted competition-wide as the reference candidate-rerank template per independent endorsement from fellow Grandmaster Radek Osmulski); Playground Series 1st-place writeups for Backpack (S5E2, Feb 2025, confirmed competition_ranking 1) and Predicting Optimal Fertilizers (S5E6, Jul 2025, confirmed competition_ranking 1, CV MAP@3 0.386 / Private LB 0.38652).


---

## nyanp (kaggle.com/nyanpn; GitHub: nyanp), Competitions Grandmaster, Osaka, Japan — 6 gold + 7 silver + 2 bronze medals across 19 competitions, peak rank #9 all-time (currently #1818). Pre-Kaggle credential: author of tiny-dnn, an open-source header-only C++ deep-learning library.

Became GM off a solo 1st place (Optiver Realized Volatility Prediction) built almost entirely on feature engineering rather than architecture — his own post-competition ablation study (run purely to teach the community, using the late-submission window) proved a single LightGBM was already enough for 1st place private LB, and that shrinking his 7-way nearest-neighbor feature bank hurt the score far more than shrinking his 11-model ensemble did. Also a persistent leak-hunter and data-integrity watchdog on the forums, independent of whether he's actively competing in that particular competition: publicly found and published a second, independent leak in SETI Breakthrough Listen, and was first to publicly flag that Foursquare's private-leaderboard train set had silently swapped to a different, larger file mid-competition. Rather than rebuilding tooling every competition, he built and open-sourced nyaggle — a personal competition-infrastructure library (experiment tracking via a run_experiment() API integrated with MLflow, a feather-cached feature store, target encoding, adversarial validation, curated GBDT hyperparameter sets, time-series CV splitters) that has been reused across the Japanese Kaggle community (291 GitHub stars).

**Signature moves.**

- Recover a masked continuous variable from a stated discretization artifact ('tick-size leak'), then build several genuinely different nearest-neighbor feature families from it (2 distance metrics x 2 grouping axes x 3 measured quantities)
- Publish post-win ablation studies purely to quantify which piece of the solution actually mattered (proved feature diversity > model diversity)
- Maintain and open-source reusable personal competition infrastructure (nyaggle) instead of rebuilding tooling every competition
- Public leak-hunting/data-integrity forum posts as a recurring habit across competitions

**Study these.** 1st place (solo), Optiver Realized Volatility Prediction 2021/2022 — writeup plus a dedicated follow-up ablation-study thread with full quantified tables; github.com/nyanp/nyaggle.


---

## Marios Michailidis (KazAnova) — kaggle.com/kazanova, Competitions Grandmaster, peaked #1 in the world (Kaggle 'Champion' badge), 41 gold / 59 silver / 42 bronze medals, creator of StackNet, Chief Data Scientist roles at H2O.ai and dunnhumby

Entered Kaggle as a software developer who had already built a Java credit-scoring GUI he named 'KazAnova' — his handle since. He explicitly treats every pipeline decision (algorithm selection, hyperparameter optimization, feature engineering, missing-value treatment) as 'hyper parameters of the final solution' to be searched jointly rather than fixed independently, and does that hyperparameter search almost entirely manually rather than via grid search ('I feel I learn more about the algorithms... by doing this manually'), claiming that after 60+ competitions he can reach the top 90% of the best hyperparameters on the first try. He holds that model diversity beats a few very strong models, backed by a personal, cross-competition bank of previously-successful hyperparameter configurations he reuses and incrementally extends. His account of his breakthrough win (Acquire Valued Shoppers' Challenge, which took him into Kaggle's top 20) shows CV design driven directly by the competition's leakage structure — predicting one held-out offer from the rest, rather than a random split — built by hand-parsing roughly 300 million transaction rows in Java on a 4GB-RAM i3 laptop.

**Signature moves.**

- Treating the ENTIRE pipeline (not just model hyperparameters) as one joint search space
- Manual, experience-driven hyperparameter search over grid search, backed by a persistent cross-competition config bank
- Multi-level meta-stacking, sometimes to a 3rd 'meta-meta' level
- Model-diversity-over-raw-strength as an explicit design philosophy
- CV schemes designed around the competition's specific leakage/grouping structure rather than random splits
- Formalized his personal stacking practice into the open-source StackNet library, which won the Truly Native competition with a 4-layer meta-model stack

**Study these.** Original Kaggle-blog 'Profiling Top Kagglers: KazAnova' interview (2015); HackerEarth 'Winning Tips on Machine Learning Competitions' webinar Q&A (2016); 'Stacking Made Easy: An Introduction to StackNet' (2017)


---

## bestfitting (Kaggle handle: bestfitting, user_id 708283) — Competitions Grandmaster, verified rank_highest #1 in the world, 42 gold / 13 silver / 2 bronze medals across 57 competitions, grandmaster level 8, holder of Kaggle's rare 'Champion' (reached #1) and 'Challenger' (top-10) badges, 12 published solution writeups.

Builds pipelines around making the LOSS FUNCTION mirror the eval metric as closely as possible (FocalLoss+Lovasz for F1-sensitive multi-label tasks, SmoothL1 for QWK-style ordinal grading) rather than bolting on post-hoc corrections. Repeatedly transplants ArcFace/metric-learning machinery — originally built for face recognition — into completely different domains (protein-label transfer via antibody-ID, image-to-text-embedding retrieval) whenever a non-label identity key is available. Treats the public leaderboard with explicit, quantified suspicion: he simulates expected public/private divergence via held-out resampling before trusting any submission choice, and hedges class-prior uncertainty with deliberately diversified dual submissions rather than a single best guess. Writes short, technically dense, code-including writeups and is self-effacing about the process ('no secrets but hard work'); competes almost entirely solo across an unusually wide span of sub-fields (satellite imagery, medical imaging, generative models, image retrieval, embedding regression) rather than specializing in one.

**Signature moves.**

- ArcFace/metric-learning transplant across unrelated domains using a non-label identity grouping key
- Monte-Carlo resampling simulation to size expected LB shakeup before finalizing submissions
- Per-label linear (Ridge) stacking over receptive-field-diverse base models
- Classical CV filters (Dark Channel Prior dehazing) fused ahead of a CNN
- Repurposing an RNN cell's gating equations as a plain tabular feature-interaction layer
- Deliberately over-wide bottleneck projection heads on embedding backbones

**Study these.** Human Protein Atlas Image Classification 2019 (1st place solo): ArcFace antibody-ID label transfer improved score by 0.03+, called by him 'a huge improvement.' Planet: Understanding the Amazon from Space 2017 (1st place solo): quantified LB-shakeup risk via resampling, then deliberately abandoned public-LB optimization in the final week — 'the final result is a big surprise to me.' Stable Diffusion - Image to Prompts 2023 (1st place solo): a massively over-wide FC projection head gave a measured +0.006 lift he calls a surprise. Also won Human Protein Atlas - Single Cell Classification 2021 (1st, 'Fair Cell Activation Network' — a Puzzle-CAM-derived weak-supervision architecture), TalkingData AdTracking Fraud Detection 2018 (3rd, NN-based solution in a GBDT-dominated domain), and placed top-5 in Google Landmark Recognition 2020 (2nd) and TensorFlow Great Barrier Reef 2022 (5th).


---

## Jean-Francois Puget (CPMP) — kaggle.com/cpmpml, Competitions Grandmaster (peak rank #6), Director of Competitive ML / KGMON team lead at NVIDIA, tagline 'Kaggle is a legal drug.'

Holds a PhD in ML from, in his own words, 'a previous millennium' that he considers 'useless' given how much the field changed; spent the intervening decades in constraint programming and mathematical optimization before returning to ML and joining Kaggle in 2016. Publicly describes his opening moves as getting a fast baseline submission, mastering cross-validation as the core skill, and reading the competition forum (including host comments) plus the winning solutions of similar recent competitions before committing to an approach. His most recent major, richly-documented win — Predict Student Performance from Game Play, 1st place with team 'French Touch' in 2023 (earning him a Kaggle Best Solution Writeup Award) — shows this discipline in concrete form: the team made zero LB-driven decisions, quantified their own CV noise floor via 10 repeated bags and only accepted a change once it beat that floor, and built a local simulator of the live evaluation API before writing any model code so they never once hit a submission error across the whole competition.

**Signature moves.**

- Baseline-first submission, then CV-scheme mastery as the core discipline
- Forums + prior-similar-competition literature review as an explicit day-1 step
- Quantified CV-noise-floor gating: only accept a feature/change if it beats a measured noise threshold
- Building a local replica of a code-competition's live evaluation API before modeling begins
- Blending GBDT + NN roughly 50/50 for robustness rather than pushing one model family to its limit
- Reports data leaks/host bugs immediately rather than quietly exploiting them

**Study these.** NVIDIA GTC 'Competition and Community Insights from NVIDIA's Kaggle Grandmasters' Q&A; 'French Touch' 1st Place Solution writeup for Predict Student Performance from Game Play (2023) — his Kaggle Best Solution Writeup Award


---

## Guanshuo Xu (Kaggle handle: wowfattie, user_id 478989) — Competitions Grandmaster, verified rank_highest #1 in the world (currently ranked #10), 29 gold / 28 silver / 2 bronze medals across 85 competitions (the most prolific of the three by competition count), grandmaster level 5, 21 published solution writeups spanning medical imaging, forensics, NLP and LLM-era competitions.

Self-described philosophy from his Kaggle bio: 'Build a model is very simple, but build a simple model is the hardest thing there is' — favors one well-tuned architecture per pipeline slot over exotic ensembles, and is unusually candid about distrusting the public leaderboard outright when the domain warrants it ('we should not use public LB for model selection' — Alaska2). Explicitly builds on and credits prior years' competition solutions rather than starting from scratch (RSNA 2019 ideas carried into RSNA 2020, credited by name). Ranges across an extraordinarily wide domain span within the same year — medical imaging, image forensics, NLP ranking, and LLM-era detection — winning or placing top-10 repeatedly in each. Always publishes working inference/training code alongside writeups. Notably declined a cash prize for a 7th-place RSNA Intracranial Hemorrhage Detection finish (documented in his own writeup title), suggesting a rules/ethics-driven streak alongside the technical craft.

**Signature moves.**

- Dual-domain (spatial + frequency-domain) bottleneck feature stacking for forensic/signal tasks
- Neighbor-embedding delta features for sequential/volumetric data
- Brute-force choose-the-lower-loss postprocessing for hierarchical label consistency
- Deliberately mis-setting a data-derived hyperparameter to hedge an unverifiable test-distribution shift
- Cross-dataset soft-label harmonization (outlier-bounded) when merging label-taxonomy-mismatched external data
- Treating proxy/legacy-competition classifier outputs as engineered features for a genetic-algorithm-optimized linear meta-model under near-zero direct supervision

**Study these.** Alaska2 Image Steganalysis 2020 (1st place solo): dual pixel+DCT domain stacking, an approach independently validated by another competitor who had abandoned a similar DCT branch after seeing its low standalone score. RSNA STR Pulmonary Embolism Detection 2020 (1st place solo): neighbor-embedding delta features, brute-force consistency postprocessing, and a deliberate hyperparameter hedge (M=192 vs. the empirically-optimal M=128) against a believed private-set shift. APTOS 2019 Blindness Detection (1st place solo, his highest-voted writeup at 303 votes): cross-dataset label harmonization pushed the final ensemble from public 0.844/private 0.934 to 0.850/0.935. Jigsaw Rate Severity of Toxic Comments 2022 (1st place solo): proxy-classifier ensemble with a genetic-algorithm-optimized linear stack, achieved with only 2 submissions per a congratulatory community thread.


---

## Gilberto Titericz (Giba) — kaggle.com/titericz, Competitions Grandmaster, twice-confirmed #1-ranked competitor in the world (Kaggle 'Champion' badge, rank_highest=1), 66 gold / 58 silver / 34 bronze medals, Senior Data Scientist at NVIDIA RAPIDS

An electronics engineer by training who spent 16 years at Siemens, Nokia and Petrobras before self-teaching data science from 2008 and joining Kaggle in 2012; his first competition (Global Energy Forecasting 2012 - Wind Forecasting) placed 3rd with a bag of Matlab neural networks, and he became #1 by finishing 2nd in Springleaf Marketing Response. His own stated iteration cycle is strictly ordered: understand the problem/features/metric, build a CV strategy, only THEN feature-engineer and pick algorithms, optimize hyperparameters manually against CV (he explicitly avoids grid search as too slow), save every model's train+test predictions, repeat with a new algorithm, and only ensemble once several good predictions exist. For final submission selection under CV/LB uncertainty he doesn't trust either signal in isolation but computes a sample-size-weighted blend of the two. He puts in roughly 15h/week and calls second-level stacking his single most-used 'creative trick.'

**Signature moves.**

- Sample-size-weighted CV/LB blend formula for picking final submissions
- Strict ordering: understand data & metric -> CV scheme -> feature engineering -> algorithm -> ensemble
- Second-level (and third-level) stacking as his default, most-relied-on technique
- Power-weighted geometric mean for blending differently-calibrated final models
- Manual/brute-force hyperparameter search from accumulated experience rather than grid search
- Reads train/test feature distributions closely enough to spot host-injected structure (led directly to the Santander leak discovery)

**Study these.** Otto Group Product Classification Challenge 1st place writeup (2015, with Stanislav Semenov); Santander Value Prediction Challenge 1st place mini-writeup (2018, the leak discovery); 'Profiling Top Kagglers: Gilberto Titericz' original Kaggle-blog interview (2015)


---

## Abhishek Thakur — kaggle.com/abhishek, self-described 'world's first quadruple grandmaster' (historic Grandmaster tier across Competitions, Datasets, Notebooks, and the now-retired Discussions track), author of 'Approaching (Almost) Any Machine Learning Problem'

Distinguishes himself from the other four profiles here by operating as much as a systematizer and teacher as a competitor — his primary artifact is a free, self-published, code-first book distilled from 100+ competitions rather than a single legendary competition win narrative. Two of his signature techniques are taught with exact, runnable code rather than just described: entity embeddings for categorical variables in neural nets, sized via a concrete formula (embedding dimension = min(ceil(cardinality/2), 50)) rather than a fixed guess; and binning a continuous regression target (via Sturge's Rule or a flat 10-20 bins on larger data) purely to make StratifiedKFold usable on regression problems with skewed targets. He was already a known figure in the community for widely-used benchmark notebooks early in his Kaggle career — KazAnova's own 2015 interview cites 'this benchmark from Abhishek: Beating the benchmark in StumbleUpon Evergreen Challenge' as a learning resource he personally used.

**Signature moves.**

- Entity embeddings with a concrete, formula-driven dimension size rather than a guessed constant
- Binning-plus-StratifiedKFold as the standard fix for regression CV on skewed targets
- Publishing exact, runnable reference code rather than prose-only technique descriptions
- Building widely-reused public benchmark notebooks early in a competition's life
- Treating the full competition workflow as reusable, scriptable pipeline code rather than bespoke-per-competition scripts

**Study these.** 'Approaching (Almost) Any Machine Learning Problem' — free PDF + full code, github.com/abhishekkrthakur/approachingalmost


---

## Bojan Tunguz — kaggle.com/tunguz, Grandmaster across Competitions/Datasets/Notebooks tracks (9 gold in competitions alone), tagline 'XGBoost is all you need.', founder/CEO of TabulAI

The most publicly vocal GBDT-over-deep-learning advocate on Kaggle, with an explicit Day-1 ritual of submitting the sample submission purely to appear on the leaderboard before doing anything else. His single best-documented win, Home Credit Default Risk 1st place (2018, 6-person team 'Home Aloan'), is simultaneously his strongest evidence AND its own built-in nuance: teammate Michael Jahrer's section of the same writeup shows LightGBM beating their best neural net by ~0.01 AUC and calls NN's role 'minor' — while also noting the NN was still 'needed at the end to fight for the last 0.0001 boost.' Bojan's specific technical contribution on that team was aggressive feature-set compression: simple forward feature selection using plain Ridge regression, which cut a 1,600+-feature aggregation set down to ~240 usable features (later 287 once merged with a teammate's set), reaching CV 0.7985 / LB 0.802-0.803 on that reduced set alone.

**Signature moves.**

- Day-1 sample-submission ritual purely for leaderboard calibration
- GBDT-first default, deep learning reserved for CV/NLP/audio or late-stage blend seasoning
- Ridge-regression forward feature selection to compress bloated aggregation feature sets by ~7x
- Running multiple GBDT flavors (XGBoost/LightGBM/CatBoost) mainly for metafeature diversity, not raw individual strength
- Public, opinionated writing (Medium, Kaggle tagline) as part of his competitive identity

**Study these.** Home Credit Default Risk 1st place writeup, team 'Home Aloan' (2018); NVIDIA GTC Kaggle Grandmasters Q&A


---

## Philipp Singer — kaggle.com/philippsinger, display name 'Psi'. PhD in Computer Science, Vienna, Austria. Founding Data Scientist at Prior Labs (the startup behind the TabPFN tabular foundation model); previously Chief Data Scientist at H2O.ai. Competitions Grandmaster, 38 gold / 13 silver medals across 84 competitions, highest rank ever achieved #1. Won 6 in-person Kaggle Days events (Tokyo 2019; Beijing, Dubai, Berlin, London, Paris 2022). 38 published write-ups.

Builds from a first-principles physical/causal model of the problem before choosing an architecture — The Zoo's NFL model derives its permutation-invariant CNN structure from an explicit verbal model of 'who can influence whom' on a football field, not from a template. Treats the CV harness itself as a primary deliverable, engineering it (GroupKFold plus repeated resampling of realistic single-sample-per-group draws) until CV and public LB track almost perfectly, then trusts CV over LB for the rest of the competition — explicitly noting long stretches without submissions because the CV signal was already trusted. He is the connective figure across nearly every team in this cluster (co-author on Landmark 2021, NFL Big Data Bowl 2020, BirdCLEF 2021, QUEST 2020, LLM Science Exam 2023), functioning as a validation-discipline anchor across many domains and collaborators rather than a single-domain specialist.

**Signature moves.**

- Derive the model architecture from an explicit physical/causal narrative of the domain before writing code
- Build a repeated-resampling validation harness that mimics the exact test-time sampling process, then trust it over the public LB
- Diagnose and numerically correct season/protocol-level measurement drift via target-correlation smell tests rather than dropping the feature
- Process-level (not thread-level) parallelism to bag many seeds inside a fixed kernel-time budget
- Cross-competition, cross-domain team formation as a recurring co-author/anchor

**Study these.** 1st place, NFL Big Data Bowl 2020, with Dmitry Gordeev ('The Zoo') — original candidate cited a $50,000 prize, which could not be independently re-confirmed this pass; 1st place both tracks, Google Landmark 2021; 1st place, Kaggle LLM Science Exam 2023 (Team H2O LLM Studio); 2nd place, BirdCLEF 2021; 2nd place, Google QUEST Q&A Labeling 2020.


---

## Dmytro Poplavskiy (Kaggle handle: dmytropoplavskiy, user_id 743064) — Competitions Grandmaster, rank_highest #11, 13 gold / 1 silver medal across just 13 competitions (every one a medal, an unusually high hit rate), grandmaster level 2, 11 published solution writeups. Brisbane, Australia; Software Engineer at Topcon Positioning Systems; GitHub 'pdima'; member of the ODS.ai (Open Data Science) community.

Concentrates almost entirely on video/temporal detection-and-tracking problems, most often in medical imaging and sports analytics (NFL), plus occasional forays into very different domains (source-code cell ordering in Google AI4Code). Consistently fuses a 'traditional' perception backbone (a detector, or a 2D CNN) with explicit hand-engineered physical/geometric correction terms — velocity from optical flow, distance/orientation encodings — rather than trusting a network to learn motion physics implicitly from raw pixels. Prefers simplifying to a single-stage, end-to-end formulation once he's identified the right feature set, replacing more complex multi-stage pipelines used by others in the same competition. Candid in writeups about what didn't work, and explicitly treats cross-fold ranking instability as a leading indicator of leaderboard shakeup risk rather than ignoring it.

**Signature moves.**

- Single-stage per-entity-and-interval architectures replacing pairwise/multi-stage pipelines
- Velocity-compensated frame re-centering to cancel confounding motion before temporal stacking
- Temporal Shift Module with per-block-depth-varied shift amounts to mimic dilated temporal receptive fields
- Symmetric dictionary-keyed prediction pooling with duplicate-insertion model weighting
- Cross-fold performance-ranking disagreement as an explicit private-shakeup risk signal

**Study these.** NFL 1st and Future - Impact Detection 2021 (1st place solo): velocity-compensated TSM-based 2.5D classification, his highest-voted writeup at 176 votes. 1st and Future — Player Contact Detection 2023 (3rd place solo): single-stage joint transformer-decoder architecture replacing pairwise detection. RSNA Pneumonia Detection Challenge 2018 (2nd place, ODS.ai team). Also placed 2nd in Google AI4Code 2022 (code-notebook-cell ordering, a markedly different NLP/code domain) and 4th (private LB) in the 2018 Data Science Bowl nuclei-segmentation competition.


---

## Christof Henkel — kaggle.com/christofhenkel, display name 'Dieter'. Deep Learner at NVIDIA, Munich, Germany. Competitions Grandmaster currently ranked #2 all-time (highest rank ever achieved: #1), 52 gold / 18 silver / 4 bronze medals across 95 competitions; also a Notebooks Grandmaster (16 gold) and Discussions Legacy Grandmaster. 54 published solution write-ups, winner of Kaggle's 2023 Best Solution Writeup Award for the ASL Fingerspelling solution.

Cross-domain architecture transplantation is his signature: he repeatedly imports state-of-the-art building blocks from ONE domain (ASR/speech — Squeezeformer, Llama rotary attention; image-retrieval theory) into a DIFFERENT domain (sign-language landmark sequences; landmark image retrieval) and gets it working end-to-end through to production export (hand-ported TF-Lite, ONNX). His writeups are structured around a rigorous per-component ablation table with signed deltas, and give an explicit 'what did not help' section equal prominence to what did — negative results are first-class deliverable content, not an afterthought. He personally handles the deployment-engineering layer himself (manual PyTorch-to-TF-Lite porting, per-layer temporal masking hand-implemented because the framework doesn't provide it) rather than treating it as someone else's job.

**Signature moves.**

- Cross-domain architecture transplant (ASR Conformer/Squeezeformer -> sign-language keypoints; retrieval theory -> orthogonal local-global fusion)
- Exhaustive signed-delta ablation tables published alongside every solution
- Explicit, equally-weighted 'what did not help' negative-results section
- Manual low-level reimplementation for deployment constraints (hand-ported TF-Lite, hand-written per-layer temporal masking)
- GeM pooling + sub-center ArcFace with dynamic margins as a personal default retrieval toolkit

**Study these.** 1st place both tracks, Google Landmark Recognition & Retrieval 2021 (arXiv:2110.03786, with Philipp Singer); 1st place, Google - ASL Fingerspelling Recognition 2023 (with Darragh) — Kaggle 2023 Best Solution Writeup Award; 2nd place, BirdCLEF 2021 (arXiv:2107.07728, with Pfeiffer & Singer); 2nd place, Google QUEST Q&A Labeling 2020 (5-person team).


---

## Pascal Pfeiffer — kaggle.com/ilu000, real name shown on writeups. Senior Principal Data Scientist at H2O.ai, Cologne, Germany. Competitions Grandmaster, 27 gold / 14 silver medals across 84 competitions, highest rank ever achieved #3. 23 published write-ups; Kaggle Days Berlin and London 2022 winner.

The RAG-systems-engineering half of the H2O LLM Studio trio — his writeups emphasize disciplined empirical search over the retrieval stack (benchmarking roughly 300 local combinations of retrieval-model x LLM x Wikipedia-dump before freezing a final blend) and data-quality archaeology (manually auditing individual failed retrievals until finding that standard Wikipedia-dump parsers silently fail to render Lua-templated numbers in science articles, then switching the whole pipeline to the Cirrussearch dump to fix it). Comfortable with counter-intuitive training-time choices once empirically validated (training on noisier machine-generated context beating training on ground-truth context). Personally maintains and extends the team's own open-source tool (H2O LLM Studio), submitting a merged upstream PR mid-competition rather than keeping the fork private.

**Signature moves.**

- Manually audit individual retrieval failure cases to find the true systemic root cause (Lua-template rendering gaps), not just tune the aggregate metric
- Empirically benchmark hundreds of retrieval-model x corpus combinations rather than assuming a leaderboard-top embedding model transfers
- Deliberately train on noisier/generated context matching the inference distribution over cleaner ground-truth context
- Maintain and upstream a real open-source training framework (H2O LLM Studio) as competition infrastructure

**Study these.** 1st place, Kaggle LLM Science Exam 2023, Team H2O LLM Studio (with Singer & Babakhin) — posted and authored the winning writeup, 219 upvotes; co-author, 2nd place, BirdCLEF 2021 (arXiv:2107.07728).


---

## senkin13 (kaggle.com/senkin13), Competitions Grandmaster, DataRobot Japan — 13 gold + 11 silver + 9 bronze medals across 163 competitions (very high volume), peak rank #14; also hosts competitions (9 hosted).

A recsys/GBDT specialist who reuses the same retrieve-then-rank skeleton (co-visitation/ItemCF candidate generation -> heavy tabular feature engineering -> LightGBM/CatBoost reranker) across very different recommender problems, then spends the final days of the competition purely on engineering discipline: re-testing his LightGBM-vs-CatBoost prior from scratch every time rather than trusting the previous competition's winner, and rewriting his entire feature pipeline under deadline pressure the moment the current tool becomes the bottleneck. Treats inference-time engineering (TreeLite compilation, GPU CatBoost, horizontal sharding across multiple rented servers) as equally important as modeling, and is willing to merge teams very late (18 days before an OTTO deadline) when the standalone team's progress stalls.

**Signature moves.**

- On-disk feature store keyed to skip recomputation ('existing features will not be created again') across a multi-week competition
- Re-derive the LightGBM-vs-CatBoost choice from scratch every competition (LightGBM won H&M 2022; a late-competition switch to CatBoost Ranker won OTTO 2023, +0.0007/+0.002/+0.0012 across the three target heads)
- Emergency pandas->polars feature-pipeline rewrite under deadline pressure (40x on the biggest joins, OTTO 2023)
- Late-stage inference-engineering pass: TreeLite (2x), CatBoost-GPU (30x vs LightGBM-CPU), horizontal sharding across ~28 servers
- Merge teams late (18 days out) when it adds decorrelated strength rather than sticking with a stalled two-person team

**Study these.** 1st place, H&M Personalized Fashion Recommendations 2022 (team w/ 30CrMnSiA/h4211819); 2nd place, OTTO Multi-Objective Recommender System 2023 (merged team w/ 30CrMnSiA + Kazuki Onodera + psilogram).


---

## knshnb (Kaggle/GitHub/Twitter: knshnb; blog.knshnb.com), Competitions Grandmaster, Preferred Networks Inc. (Tokyo) — 5 gold medals from only 6 total competitions entered, peak rank #40 all-time.

Enters competitions extremely selectively (6 ever) but converts almost every entry into gold, always as part of a small, tightly-coordinated Preferred Networks trio (with charmq and Yiemon773/Yoichi Yamakawa). His documented pattern across 3 competitions (Happywhale 1st, Contrails 3rd, Stable-Diffusion-Image-to-Prompts 11th) is identical: build his own full pipeline in parallel with teammates' independent pipelines, iterate hardest on the single most CV-sensitive component (head/loss for classification, 3D-conv placement for segmentation), and squeeze cheap search budget out of a proxy setup when the real one is too expensive to search directly. Every writeup he authors ends with a rigorous, itemized 'what did not work' section treated as seriously as the positive results.

**Signature moves.**

- Cheap-proxy hyperparameter search (tune on tiny image size + smallest backbone with Optuna, transfer up to production scale)
- Push 2.5D-to-3D fusion into U-Net skip connections rather than after the whole network
- Deliberate LB probing to recover a hidden test statistic when validation can't estimate it
- Patch-tile a frozen encoder to beat its native resolution ceiling
- Ship an exhaustive, itemized 'what did not work' postmortem in every writeup

**Study these.** 1st place, Happywhale - Whale and Dolphin Identification 2022 (w/ charmq); 3rd place, Google Research Identify Contrails 2023 (w/ charmq + Yiemon773); 11th/gold, Stable Diffusion - Image to Prompts 2023 — three gold-medal team writeups fetched and quote-verified directly.


---

## charmq (Kaggle handle; Kaggle profile lists organization 'Rist', occupation Fellow/Engineer), Competitions Grandmaster — 18 gold + 13 silver + 7 bronze medals across 66 competitions, peak rank #8 all-time.

The connective node between two different top-Japanese-GM clusters: the Preferred-Networks trio (knshnb + Yiemon773) on computer-vision competitions, and a Rist-based duo/trio with Takoi on NLP and entity-matching competitions. Every writeup he co-authors with the PFN trio explicitly credits 'Preferred Networks, Inc. for allowing us to use computational resources,' and every one of his teams (both clusters) independently builds a full pipeline per member and ensembles the finished pipelines rather than splitting sub-tasks. He is comfortable moving between very different problem types — metric learning, 2.5D segmentation, CLIP-style retrieval, and fuzzy text+geo entity-matching — and on the entity-matching team was central to a late leak-discovery push that changed the outcome of the competition.

**Signature moves.**

- Build one complete independent pipeline per teammate, then ensemble the finished pipelines (documented in 3+ writeups)
- Run and stabilize the largest-scale backbones (resnetrs420, maxvit_large, efficientnet_l2), including ad hoc fixes like disabling AMP for maxvit and zeroing NaN outputs from efficientnet_l2
- Diagnose a data leak from the 'overfitting improves LB' signature, then exploit train/test row overlap for direct label recovery
- Cross-domain teaming: same person appears in CV-metric-learning-winning teams and NLP/entity-resolution-winning teams in the same year

**Study these.** 1st place Happywhale 2022; 3rd place Contrails 2023; 11th/gold SD-Image-to-Prompts 2023; AND 1st place Foursquare - Location Matching 2022 (team re-waiwai, w/ Takoi + pao) — four gold/near-gold finishes across two domains and two different teammate clusters, all quote-verified against primary writeups.


---

## Yiemon773 (Yoichi Yamakawa; kaggle.com/yoichi7yamakawa), Competitions Grandmaster, Preferred Networks — 9 gold + 13 silver + 5 bronze medals across 57 competitions, peak rank #41.

Became a Grandmaster off a solo gold, in his own words: 'This is My First Solo Gold Medal... I'm very impressed to win the gold medal among so many kaggle GMs, masters' (Happywhale, 10th place), before joining the knshnb/charmq Preferred-Networks trio for the next two competitions. His solo writeup shows a habit of isolating one variable at a time under deadline pressure — most notably reading a paper (arXiv:2010.05350) mid-competition and directly re-deriving a training-stability fix from its formula rather than blindly tuning around the instability. After joining team writeups, he contributes his own fully independent model branch (a 2D-model pipeline, distinct from teammates' 2.5D approaches), matching the trio's 'independent pipeline per member' discipline.

**Signature moves.**

- Mixup in embedding space (not pixel space), paired with soft-label ArcFace (own figure: CV +0.003 to +0.005)
- Progressive dynamic-margin warm-up: ramp ArcFace margin coefficient 0.2->1.0 over the first 5 epochs to stabilize early training on imbalanced classes
- Auxiliary species/category classification head stacked alongside the main ArcFace+Focal loss
- Contribute an independent, architecturally-distinct pipeline within a shared-goal team rather than a task-split sub-component

**Study these.** 10th place solo gold, Happywhale - Whale and Dolphin Identification 2022 (his first Kaggle solo gold); 3rd place Contrails 2023; 11th/gold SD-Image-to-Prompts 2023.


---

## Dmitry Gordeev — kaggle.com/dott1718, display name 'dott'. AI Research at NVIDIA, Vienna, Austria. Competitions Grandmaster, 17 gold / 10 silver medals across 50 competitions, highest rank ever achieved #5. Notably lean public footprint relative to medal count — only 18 kernels, 122 discussion posts, 16 write-ups total.

The other half of the long-running Singer/'Psi' partnership — The Zoo's writeup explicitly frames the winning idea as a joint verbal model of the problem worked out before any code, and closes with 'don't forget to give your upvotes to Psi as well — this model is a great example of teamwork,' signaling a deliberately even, non-solo-credit partnership across the many competitions the two have shared. A recurring teammate across very different domains (sports-tracking CNN regression, encoder-based NLP Q&A) rather than a single-architecture specialist — consistent with a role centered on problem framing and validation rigor shared with Singer more than a technique fingerprint separable from the team's joint output.

**Signature moves.**

- Joint problem-framing-before-architecture partnership with Philipp Singer across sports, NLP, and other domains
- Season/measurement-drift feature correction validated via target-correlation smell tests
- Deliberately minimal public footprint relative to medal count — competes and ships, posts rarely

**Study these.** 1st place, NFL Big Data Bowl 2020, with Philipp Singer ('The Zoo'); 2nd place, Google QUEST Q&A Labeling 2020 (5-person team including Henkel, Singer, CPMP, Jeblick).


---

## Takoi (Takoi Hirokazu; kaggle.com/takoihiraokazu), Competitions Grandmaster, Rist (Fukuoka) — 19 gold + 18 silver + 3 bronze medals across 98 competitions. Currently ranked #9 in the world (rank_current AND rank_highest both = 9 of 212,344 ranked users) — sitting at his own all-time peak.

Cross-domain: wins solo in NLP (CommonLit, alone) and in a tight Rist-based team in geospatial entity-matching (Foursquare, w/ charmq + pao). Runs large, heavily-optimized ensembles (19 models, Nelder-Mead-derived blend weights with negative weights allowed for models that hurt CV/LB), then layers a second, separate optimization pass on top — a piecewise post-processing calibration by predicted-value bucket, applied only after the blend is frozen. On the entity-matching win, was part of the team that diagnosed a data leak purely from CV/LB divergence symptoms rather than any host announcement, then exploited train/test row overlap for direct label recovery.

**Signature moves.**

- 19-model ensemble with Nelder-Mead-derived, sign-unconstrained blend weights
- Separate piecewise post-processing calibration by predicted-value bucket, Nelder-Mead then hand-tuned against public LB (+0.001~0.002 on top of an already-optimized blend)
- 4-stage entity-matching funnel: dual-signal KNN candidate generation (geo-distance + name-embedding cosine) -> cheap LightGBM funnel -> heavy transformer+CatBoost rerank -> post-process re-linking of newly-emerged candidate pairs
- Diagnose a leak from symptom, not announcement ('LB score is higher with overfitted models... training with many epochs leads to a worse CV score')

**Study these.** 2nd place, CommonLit Readability Prize 2021 (solo); 1st place, Foursquare - Location Matching 2022 (team re-waiwai, w/ charmq + pao).

