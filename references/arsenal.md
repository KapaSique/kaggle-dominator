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

## Kaggle MCP server — recon tools the CLI does not expose

When the official Kaggle MCP server is connected and authorized (`authorize`; it needs its own
OAuth and will answer `Unauthenticated` until then — the CLI from `~/.kaggle/kaggle.json` is the
fallback and covers most plumbing), it adds recon capabilities worth reaching for:

**Front selection and metadata.** `get_competition` returns everything the selection gate needs in
ONE call — `awards_points`, `new_entrant_deadline`, `enabled_date`, `team_count`,
`max_daily_submissions`, `is_kernels_submissions_only`, `evaluation_metric`, `max_team_size`.
`search_competitions` lists by category/deadline. This is faster and more complete than parsing CLI
output; see `front-selection.md`.

**Simulation/ladder recon — the highest-value block.** `list_submission_episodes`,
`get_episode_replay`, and `get_episode_agent_logs` are how replay distillation is actually done:
they give you the leaders' *behaviour* when their code is private. `list_team_public_submissions`
lets you match an episode to the submission carrying a team's displayed score — without that match
you sample their weaker second agent. See `simulation.md`.

**Discussion and solution mining.** `list_forum_topics`, `get_forum_topic`, `list_topic_messages`,
`list_competition_topics` pull writeups and frontier chatter programmatically instead of by hand.
`search_notebooks` / `get_notebook_info` / `list_notebook_files` support the four-check notebook
screen (learned-playbook rule 3) before spending an accelerator hour.

**Hackathon/judged fronts.** `get_hackathon_overview`, `list_hackathon_tracks`,
`list_hackathon_write_ups`, `download_hackathon_write_ups`, `get_writeup*` retrieve the rubric and
past winning writeups — the rubric IS the metric in that format.

**Kernels and submission plumbing.** `create_notebook_session`, `get_notebook_session_status`,
`download_notebook_output`, `submit_to_competition`, `create_code_competition_submission`,
`get_accelerator_quota`. Submission calls remain approval-gated regardless of transport.

## Simulation engines

**`kaggle-environments`** — the engine behind Kaggle's simulation competitions
(`make("<env>")`, `env.run([...])`, replay rendering). **Pin the exact version and verify the pin
behaviourally**: engine math changes between releases, and a version assertion can pass while
`import` resolves an older copy earlier on `sys.path`. Details and the fixture pattern in
`simulation.md`.

**Official replay datasets.** Several simulation competitions publish a daily episode dataset
(e.g. `kaggle/kaggriculture-episodes-index`) with a `manifest.csv` carrying each episode's mean
ladder rating — letting you analyze the meta by rating band and track how fast it shifts. Look for
one before building your own scraper.

## Infrastructure skill

If a separate `kaggle` infrastructure skill is installed (e.g. the community shepsci skill: modules for kagglehub, the CLI, an MCP server, badge collection, competition reports), use it for the plumbing (download/submit/push/kernel-poll, hackathon-rubric retrieval), and use `kaggle-dominator` for the strategy and technique selection. This skill assumes `~/.kaggle/kaggle.json` (or `KAGGLE_API_TOKEN`) is configured.
