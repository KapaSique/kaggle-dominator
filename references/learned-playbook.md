# Learned playbook

Promoted rules. Every entry is `claim — evidence — scope/limits`, and every number
below is a real leaderboard score or a real rank, never a remembered one. Read
this BEFORE writing code on a new front; it is ordered by measured value.

---

## 1. On a shared-public-package front, adoption speed beats your own deltas

**Claim.** When hundreds of teams mount the same public weights, the public
frontier moves faster than any increment you can grind. Watch it at the same
cadence as your own experiments, not after them.

**Evidence (RSNA Knee 2026, Research, 1900+ teams).** Days spent on our own
ensemble — own retrained members, per-target combiners, an OOF matrix, a planned
diffusion-feature extractor — returned **+0.001** (0.899 → 0.900). Meanwhile the
bronze line moved 0.902 → 0.911 in three days and our rank fell 151 → 355. One
adoption of the fresh community frontier then returned **+0.011 and 195 ranks**
in a single submission; a second adoption four days later returned **+0.005 and
133 ranks** (0.915 → 0.920, rank 238 → 105).

**Scope/limits.** Applies wherever a public package or notebook is the base most
of the field runs (code competitions, DL fronts with shared weights). It does NOT
apply to fronts where no strong public base exists. Adoption is not a substitute
for your own delta — stack them (see rule 2).

---

## 2. Carry one measured delta ONTO each frontier you adopt

**Claim.** Reproducing the frontier exactly lands you in the middle of the pack
that also reproduced it. Keep one small, leaderboard-measured edit and re-apply it
to every new base you adopt.

**Evidence (RSNA Knee).** Replacing the member combiner's percentile rank with the
raw probability — literally `r = np.asarray(m['pred'])` instead of
`pd.DataFrame(m['pred']).rank(pct=True)` — was worth **+0.001** when first measured
on 20 DINOv2 members (0.899 → 0.900). The same one-expression edit then transferred
intact onto **four structurally different later bases** (24 members + DINOv3 +
RadImageNet; two further revisions; a different author's lineage), moving rank
**170 → 119** on one of them. A finding that survives a change of pipeline is a
property of the metric, not of the pipeline.

**Scope/limits.** Verified for macro ROC AUC over member ensembles. The gain is
diluted when the base re-ranks your output downstream (there it moved the 4th
decimal, not the 3rd). Re-verify once per new metric.

---

## 3. Screen a public notebook in four checks before spending a run on it

Run these in order; each one rejected a real candidate on this front.

1. **Empty strings in `dataset_sources`** → private datasets the API cannot name
   because you lack access. The notebook is unreproducible by anyone but its
   author. *`prvsiyan` had 5 empty entries; three of his pipeline stages fell back
   and his 0.906 was unreachable. `analyticaobscura` had 3; rejected without a run.*
2. **`kernel_sources`, especially empty entries** → depends on another notebook's
   output. A non-empty one may resolve fine; an empty one will not. *v46's resolved
   and produced our best score at the time; v47's empty entry produced
   `m2 members not mounted` and cost 0.001 against its own sibling.*
3. **Last-run date** → a strong author can be sitting on stale code. *`tonylica` was
   ranked ahead of everyone at 0.920 while his listed notebook was a week old.*
4. **Author's leaderboard rank** → see rule 4.

**Scope/limits.** Metadata is one `kernels pull -m` and costs seconds; the run it
replaces costs an hour of accelerator. Always screen first.

---

## 4. Author rank is a FILTER, not an ordering

**Claim.** Where the notebook's author stands on the leaderboard reliably rejects
weak notebooks. It does NOT tell you which of two strong notebooks is better,
because authors do not publish what they submit.

**Evidence (RSNA Knee).** As a filter it worked: `salemali7` was the newest
notebook with 48 votes in a day and MORE mounted datasets than the one we took —
and its author sat at 0.909, below what we had already submitted. Rejected on one
command, saving a run. As an ordering it failed twice: `sofiaanjenje` (rank 294,
0.914) beat `mattiaangeli` (rank 91, 0.917) by **0.915 vs 0.914**; and two
notebooks by the SAME author, `amanatar`, scored **0.920 and 0.911** — a 0.009
spread with no external signal separating them.

**Scope/limits.** Votes and freshness are worse signals than author rank, not
better. Treat "author ahead of us" as the admission criterion and nothing more.

---

## 5. Run independent lineages in PARALLEL, not best-first in sequence

**Claim.** When two or more candidates pass screening, submit them together.
Sequential best-first stops at the first plausible winner and never sees the
better one.

**Evidence (RSNA Knee).** Three times in one week the "backup" beat the primary.
The decisive case: `amanatar`'s two notebooks submitted together returned **0.920**
and **0.911**; a best-first policy had even odds of stopping at 0.911 and would
have missed +0.009. Cost of the parallelism: one extra submission slot.

**Why it is nearly free.** Ranking uses your BEST submission, so a weak candidate
costs a slot and never the position. Running a kernel costs accelerator time but
no slot at all — so prepare candidates by running, and spend slots only on
survivors.

**Scope/limits.** Bounded by the daily submission limit; keep the last slot of the
day unspent. Also bounded by concurrent-accelerator behaviour — see rule 7.

---

## 6. Read the numbers inside an artifact, never its name

**Claim.** Titles, vote counts and log messages describe intent; only the recorded
numbers describe result.

**Evidence (RSNA Knee), three instances.**
- A published bundle looked like the portable form of a +0.0076 blend partner. Its
  own `receipt.json` recorded the component at **0.698** gold macro AUC against
  the author's 0.842 base, with `promotion_gate_passed: false` — its author had
  bootstrapped it 5000 times and declined to deploy it. Reading the receipt killed
  a multi-day extractor build.
- A kernel log printed `wrote 0.65*rank(parent)+0.35*(...)` while the code computed
  with `_RAD_ALPHA = 0.40`. The message was a **stale hard-coded string**. Trusting
  it would have meant "fixing" a change that had already applied.
- Kaggle's file listing reported **871 bytes for an 89 MB weights file** and 850
  bytes for a 183 KB archive. Verify sizes by downloading, never by listing.

**Scope/limits.** Universal. Cheapest habit in this playbook and the highest hit
rate.

---

## 7. Platform constraints surface as errors — read them literally

**Claim.** Several hard limits are invisible until an API call quotes them back at
you. Collect them once per competition rather than inferring them.

**Evidence (RSNA Knee).**
- **Accelerator restrictions are per-competition and enforced at submit time**, not
  at run time: `Submission not allowed: Your Notebook cannot use TPU in this
  competition`, then `... cannot use P100 GPUs in this competition`. Both were
  pre-flight rejections that consumed no slot. The allowed shape was
  `machine_shape: NvidiaTeslaT4`. This also explained a `CUDA error: no kernel
  image is available` that had looked like a bug in someone's code — it was simply
  a P100.
- **The kernel-push size limit counts non-executing cells.** A 1,013,322-byte
  notebook was refused with a 403; dropping its 27 markdown cells (46,289 chars
  that never run) brought it to 898,117 bytes and it pushed, with the SHA-256 of
  the concatenated code unchanged. Re-serialising the JSON compactly made it 0.8%
  *worse* — the file was already minified.
- **Isolate the cause with one free probe before theorising.** A 1 KB notebook
  carrying the same metadata verbatim settled "is it size or is it the accelerator"
  in a single push.

**Scope/limits.** Values are competition- and date-specific; re-derive them, but
expect the same *classes* of limit.

---

## 8. Ensemble diversity must come from a different pretraining regime

**Claim.** Adding more members of the same family does not help; adding a model
trained on different data with a different objective does. And member count is not
quality.

**Evidence (RSNA Knee).** Our own reseeded members of the same architecture
correlated 0.786 with the base and **hurt at every blend weight**. A model from a
different regime correlated 0.742 and gave **+0.0167 macro AUC on the 58
expert-annotated studies** at w≈0.3–0.5. The public frontier that jumped the whole
field embodied exactly this: DINOv2 + DINOv3 (self-supervised, different
pretraining) + RadImageNet (radiology-pretrained rather than natural images).
Separately, a 20-member ensemble beat a 24-member one by **0.920 vs 0.911** —
count is not quality.

**Scope/limits.** Measured on medical imaging. The mechanism (decorrelation via
pretraining regime) is general; the specific weights are not.

---

## 9. Check a gain against labels of the same KIND the leaderboard uses

**Claim.** A gain measured against proxy labels can be an artefact of those labels.
Find whatever small set of true labels exists and re-measure there before spending
on it.

**Evidence (RSNA Knee).** A +0.00756 blend gain had been measured on 4407
report-derived pseudo-labels. The partner's heads were *fitted* on such
pseudo-labels, so it could have been scoring lexicon noise the leaderboard does not
contain — a specific, testable suspicion. The competition's own `train.csv` carried
58 expert annotations; re-measured there the gain **roughly doubled** (+0.0167,
P(>0)=0.973) and the partner turned out to be our equal on real labels, not weaker.
The suspicion was refuted with evidence instead of being argued about.

**Scope/limits.** Small gold sets give wide confidence intervals (n=58 here). Use
them to check the SIGN and rough size of an effect, not to tune weights.

---

## 10. A flat parameter sweep is a positive result — it localises the cause

**Claim.** When a swept constant stops moving the score, the remaining gap must
live somewhere else. Stop sweeping and look there.

**Evidence (RSNA Knee).** Sweeping one blend weight across three submissions gave
0.911 / 0.912 / 0.912. Because the curve was measurably flat, the author's
remaining +0.005 could not be in that constant — so it had to be in the ~6.7 KB of
code he had added that morning. Adopting the code returned **+0.002 and 50 ranks**.
Continuing to chase his parameter value would have spent slots on a plateau.

**Scope/limits.** Requires at least three points to call a plateau, and the points
must be real scores.

---

## 11. A ladder rating peak is a sample-size artefact, not a capability

**Claim.** On rating-based ladders (TrueSkill/Elo/Bradley-Terry) an early reading is a
high-variance estimate. Its **peak** is systematically inflated and does not survive more
games, so a historical peak must never be used as an agent's strength — and cannot be
recovered by resubmitting the same artifact.

**Evidence (Pokemon TCG AI Battle, ladder, 6810 teams).** One artifact
(`makthanithin` public reproduction) recorded **1062.2** shortly after submission on
2026-06-21. The identical artifact resubmitted on 2026-07-10 scored **674.1**; an
intermediate snapshot on 2026-07-11 read **769.4**. Three readings, one unchanged agent,
a spread of **388 rating points** — the 1062.2 was a property of having played few games.
Independent public confirmation on another ladder: two uploads with the **same SHA-256**
`main.py` sat at 2182.2 and 1210.1 simultaneously (Kaggle created a separate rating
instance per upload), and the lower one drifted to 1865.6 with no code change.

**Scope/limits.** Applies to any live-rating ladder, not to fixed-test leaderboards. Record
`(rating, games_played, timestamp)` as one unit and compare agents only at comparable game
counts. The earlier account lesson "ladder drift ±200" understated it in the wrong
direction: drift is not merely noisy, the **early peak is biased upward**.

---

## 12. Before submitting to a ladder, ask what the submission DISPLACES

**Claim.** Ladder platforms play only the most recent N submissions of a team. A new
submission therefore does not merely add a candidate — it **evicts** one. A candidate that
is not clearly better than what it displaces has negative expected value regardless of how
interesting it is.

**Evidence (Pokemon TCG).** Two self-built agents submitted on 2026-08-11 (`BattleCore-A`
708.7, `Meta-A` 706.4) became the team's active pair, retiring the stronger reproduced
lineage. Team rank on 2026-08-19: **1790 / 6810**, with gold at top 23. The account's own
memory had flagged this as "possibly just not selected" five weeks earlier; the mechanism
turned out to be displacement by newer, weaker uploads.

**Scope/limits.** Ladder/simulation formats only — on fixed-test leaderboards ranking uses
your BEST submission, which is why parallel lineage testing is nearly free there (rule 5).
The two formats have **opposite** submission economics; do not carry the habit across.
Verify N and the selection rule per competition, and re-read standings after submitting to
confirm the intended artifact is live.

---

## 13. Check `awards_points` before opening a campaign, not after

**Claim.** Percentile performance is worthless as medal progress unless the competition
awards medals. This gate belongs at front-selection time, ahead of data, metric, and
notebooks.

**Evidence (this account, cross-front).** The two best percentile results ever recorded —
**103/3023 = top 3.41%** (`playground-series-s6e5`) and **206/2817 = top 7.31%**
(`stellar-class-s6e6`) — are both Playground with `awards_points=false` and earned nothing.
The account holds **zero medals** while executing at top-3.41% level. Meanwhile a
medal-bearing front reached **106/1918 = top 5.5%** (RSNA Knee), which is worth strictly
more than either despite the weaker percentile.

**Scope/limits.** The flag is per competition slug, not per event: a Featured event can pair
a medal-bearing ladder with a writeup track that awards none. Playground remains valid for
method calibration — label it that way in the campaign record so its percentiles are never
reported as medal progress. Full thresholds and the scan command are in
`references/front-selection.md`.
