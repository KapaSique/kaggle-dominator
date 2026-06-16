# Code competitions and judge-scored hackathons

Two formats with special submission rules. A code comp is scored by a metric on a hidden test via an inference notebook; a hackathon is scored by humans on a rubric, and there the writeup matters more than the model.

## Contents
- [Code competition](#code-competition)
  - [Regression guard](#regression-guard)
  - [Common traps](#common-traps-code-comp)
- [Judge-scored hackathon](#judge-scored-hackathon)
  - [The rubric is your metric](#the-rubric-is-your-metric)
  - [The writeup](#the-writeup)

---

## Code competition

The submission is a notebook that Kaggle runs on a **hidden test** under hard constraints. The metric is real, but the cost of a pipeline error is zero/invalid, not "a bit worse."

- **Understand the format thoroughly BEFORE modeling:** what the input is in the hidden run, the exact `submission.csv` format, the runtime limit (hours), GPU/CPU, **internet almost always off**.
- All weights/data/packages as private dataset inputs (since there's no internet). Verify the notebook works with `internet=off`.
- Reproduce the top public baseline, lock in BEST_KNOWN, then improve on top.
- Watch inference time — a solution that doesn't fit the limit = 0, however accurate.

### Regression guard

A new version is submitted only if it is **measurably no worse** than the current best (on CV + on public LB). Code comps are especially treacherous: an edit that lifts public can drop private or time out on the full hidden test. Therefore:
- Keep a BEST_KNOWN notebook that is guaranteed to run and give a known score.
- Every experiment is alongside, not instead of. Replace BEST_KNOWN only after confirmation.
- For the final you often pick 2 submissions: an aggressive one + the reliable BEST_KNOWN. Don't bet both on aggressive ones.

### Common traps (code comp)

- Crashing on a hidden-test edge case absent from train (empty input, a new class, a NaN).
- A timeout on the full test (it fit on the public slice).
- Package-version mismatch between your kernel and the submission environment.
- Leakage through careless preprocessing fitted on all of train including "the future."

## Judge-scored hackathon

No LB. Scored by **judges on a rubric**. The best model does NOT automatically win — what wins is whatever maximizes rubric points and convinces the judge.

### The rubric is your metric

- Get the **rubric and rules first**: criteria, weights, deadline, submission format. Via an installed `kaggle` infra skill if available (hackathon endpoints: overview, tracks, writeups).
- Decompose the rubric into a checklist: every weighted criterion = a sub-goal. Allocate effort by the weights, not by your taste.
- Study the winning writeups of past/adjacent hackathons — what landed with the judges.
- If there are tracks — pick the one where your edge hits the rubric hardest.

### The writeup

In a judged format, **presentation is half the score** (sometimes more). A strong piece of work with a weak writeup loses.
- A clear story: problem → why it matters → your solution → proof it works → impact.
- Show a metric/demo/example even when there's no ranking LB — judges trust numbers and visuals.
- Address each rubric criterion explicitly in the text — don't make the judge hunt for it.
- Reproducibility and cleanliness if the rubric asks for it: a working notebook, a clear repo, run instructions.
- The LLM agent is the main tool here: it writes the structure, the text, the explanations, and polishes the presentation against the criteria. That's its strong suit — use it to the fullest.
