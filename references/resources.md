# Public learning resources

Curated, freely accessible sources for **learning the craft** — official
documentation, courses, books, and how to read public notebooks and writeups
responsibly. This complements [arsenal.md](arsenal.md), which lists *tools,
solution troves, and communities*; to avoid duplication, the winners'-solution
repositories and tooling live there, and the *educational* material lives here.

Everything below is public and legal to use. When you reuse public code, data,
or ideas, **attribute the author and respect the licence** — good attribution is
part of good practice, not an afterthought.

## Contents
- [Guided learning on Kaggle](#guided-learning-on-kaggle)
- [Official library documentation](#official-library-documentation)
- [Courses and books](#courses-and-books)
- [Method references](#method-references)
- [Reading public notebooks and writeups](#reading-public-notebooks-and-writeups)
- [Ethics, licences, and attribution](#ethics-licences-and-attribution)

---

## Guided learning on Kaggle

- **Kaggle Learn** (`kaggle.com/learn`) — short, hands-on micro-courses: *Intro
  to Machine Learning*, *Intermediate ML*, *Feature Engineering*, *Data
  Cleaning*, *Data Visualization*, *Model Explainability*. The fastest way to
  internalise the end-to-end loop.
- **Kaggle Notebooks** — the public *Code* tab of any competition. Filter by
  *Most Votes* to find well-explained baselines; read them to understand, not
  only to reuse.
- **Kaggle Discussions** — the *Getting Started* threads and each competition's
  discussion are where beginners' questions are answered by strong competitors.

## Official library documentation

Prefer primary docs over blog posts — they are current and correct.

- **scikit-learn** (`scikit-learn.org`) — model selection, `Pipeline`,
  cross-validation splitters (`StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`),
  metrics. The reference for honest validation.
- **pandas** (`pandas.pydata.org`) — data wrangling; read the *User Guide* on
  merging, groupby, and missing data.
- **LightGBM**, **XGBoost**, **CatBoost** — the gradient-boosting workhorses;
  each has a parameter-tuning guide worth reading once in full.
- **PyTorch** (`pytorch.org`) and **timm** / **Hugging Face** docs — for the
  CV/NLP track (see [deep-learning.md](deep-learning.md)).
- **NumPy**, **Matplotlib** / **seaborn** — arrays and plotting for EDA.

## Courses and books

- **"How to Win a Data Science Competition"** (Coursera, HSE) — the classic
  competition-craft course: validation, feature engineering, ensembling.
- **"Approaching (Almost) Any Machine Learning Problem"** — A. Thakur; a
  practical, competition-flavoured workflow book.
- **"The Kaggle Book"** (Banachewicz & Massaron, Packt) — end-to-end competition
  practice; strong chapters on validation, blending, and stacking.
- **fast.ai** (`course.fast.ai`) — practical deep learning, top-down.
- **NVIDIA Kaggle Grandmasters Playbook** — well-tested tabular techniques
  (linked from [arsenal.md](arsenal.md)).

## Method references

- **Papers With Code** (`paperswithcode.com`) — find the method behind a
  technique and its reference implementation.
- **arXiv** (`arxiv.org`) — primary sources for a model or trick you want to
  understand at depth before adapting it.
- **Meta Kaggle** (a public dataset on Kaggle) — historical competition,
  notebook, and submission metadata; analyse which methods have won which
  problem types.

## Reading public notebooks and writeups

Public notebooks and past-competition writeups are the highest-leverage learning
material on Kaggle. Read them as a student, not a copier:

1. **Reproduce** the notebook as published and confirm the score.
2. **Explain** each step to yourself — *why* this validation, *why* this feature,
   *why* this model. If you cannot explain it, you have not learned it yet.
3. **Extend** with one idea of your own, measured on your CV.
4. **Credit** the source in your notebook and notes, and check the licence
   before reusing code.

Solution-writeup collections (700+ competitions, winners' approaches) are listed
under *Solution treasure troves* in [arsenal.md](arsenal.md).

## Ethics, licences, and attribution

- Use only **public data**, the **provided training labels**, and
  **competition-rule-compliant** methods. Read the competition **Rules** before
  using any external data.
- **Attribute** every public notebook, dataset, or idea you build on, and honour
  its **licence** (many Kaggle notebooks are Apache 2.0, but check each one).
- Do not attempt to infer hidden/private test labels, probe the private
  leaderboard for leakage, bypass submission limits, or use multiple accounts —
  these violate competition rules and teach nothing durable.
- Avoiding leakage is simultaneously the ethical choice and the correct one: an
  honest pipeline is the one that generalises.

---

*Related:* [arsenal.md](arsenal.md) (tools, solution troves, communities),
[learning-craft.md](learning-craft.md) (practice checklists),
[grandmaster-playbook.md](grandmaster-playbook.md) (how the top competitors
think).
