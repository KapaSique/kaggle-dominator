# Arsenal — tools, treasure troves, communities

Common to all competition types. Visit during RECON: half the top's work is knowing where ready-made diversity lives and not reinventing the wheel.

## Contents
- [Foundation models and AutoML](#foundation-models-and-automl)
- [Solution treasure troves](#solution-treasure-troves-the-recon-essential)
- [Playbooks and theory](#playbooks-and-theory)
- [Experiment tracking](#experiment-tracking)
- [GPU acceleration on kernels](#gpu-acceleration-on-kernels)
- [Communities](#communities)
- [Infrastructure skill](#infrastructure-skill)

---

## Foundation models and AutoML

**TabPFN v2.5 / TabPFN-3 (Prior Labs)** — a foundation model for tabular data. In-context prediction with no training: feed train+test, get predictions in seconds. **Beats AutoGluon 1.4** on small/medium tabular tasks. API at `ux.priorlabs.ai`. Use it as ONE of the predictors in the ensemble — not as a GBDT replacement. It has a size limit — on large data, trim features / subsample, or combine with GBDT.

**AutoML alternatives** (each one is a separate source of OOF diversity):
- **AutoGluon** — the strongest out-of-the-box tabular AutoML; builds its own stack.
- **H2O AutoML**, **LightAutoML (LAMA)**, **FLAML** — other engines. Different engines → different errors → a better ensemble.

Rule: the more *different in nature* the predictors in Hill Climbing, the higher the chance of escaping the flat zone. TabPFN + GBDT + AutoML = three different natures.

## Solution treasure troves (the RECON essential)

- **[farid.one/kaggle-solutions](https://farid.one/kaggle-solutions/)** / [github faridrashidi/kaggle-solutions](https://github.com/faridrashidi/kaggle-solutions) — 700+ competitions, links to winners' writeups, "kernels of the week." FIRST stop: find a competition of the same type/series, read what the top 1–3 used.
- **[anuj0456/kaggle_competition_solutions](https://github.com/anuj0456/kaggle_competition_solutions)** — a compilation from forums via Meta Kaggle.
- **[kyaiooiayk/Kaggle-Competitions-Analysis](https://github.com/kyaiooiayk/Kaggle-Competitions-Analysis)** — analysis of which methods won most often; the repo's key lesson: correct CV and early detection of LB probing decide the outcome.
- **Meta Kaggle** (a dataset on Kaggle itself) — metadata for all competitions/submissions/discussions. Load it into a kernel and analyze patterns.

## Playbooks and theory

- **[NVIDIA Kaggle Grandmasters Playbook](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)** — 7 battle-tested techniques from KGMoN (Théo Viel, Gilberto Titericz). cuML stacking, pseudo-labeling, and more.
- **The Kaggle Book (Packt)** — chapters on blending/stacking and pseudo-labeling; the de facto textbook.

## Experiment tracking

When variants number in the dozens, memory fails — keep a log.
- **[Weights & Biases](https://wandb.ai)** — `wandb.log()` in a kernel in one line, hyperparameter sweeps, run comparison. Needs internet in the kernel + a key.
- **[Neptune.ai](https://neptune.ai)** — an alternative, handier for comparing many models.
- Minimum without external services: a single `results.csv` (slug, features, model, params, CV, LB) in a private dataset, appended after each kernel.

## GPU acceleration on kernels

- **RAPIDS cuDF / cuML** — pandas/sklearn on GPU. Stacking that would take days on CPU runs in hours on a P100/T4. Set `enable_gpu=true`, import `cudf`/`cuml`. It speeds up tree ensembles and KNN/linear meta-models by an order of magnitude.
- **cuDF pandas accelerator** — `%load_ext cudf.pandas` speeds up existing pandas code with almost no changes.

## Communities

- **[Kaggle Discord](https://discord.com/servers/kaggle-1101210829807956100)** — official, with per-competition and local-language channels.
- **Telegram Kaggle groups** — team formation, strategy discussion.
- **NVIDIA KGMoN** — Grandmaster breakdowns on the NVIDIA Tech Blog.

## Infrastructure skill

If a separate `kaggle` infrastructure skill is installed (e.g. the community shepsci skill: modules for kagglehub, the CLI, an MCP server, badge collection, competition reports), use it for the plumbing (download/submit/push/kernel-poll, hackathon-rubric retrieval), and use `kaggle-dominator` for the strategy and technique selection. This skill assumes `~/.kaggle/kaggle.json` (or `KAGGLE_API_TOKEN`) is configured.
