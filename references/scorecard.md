# Practice scorecard — what actually worked, per competition

This is the account's REAL practice history — every competition actually worked, its measured result, the ONE lever that moved the needle, and the ONE thing that wasted time. It is a pattern memory: when studying a NEW competition, find its TYPE below and read what already worked/failed in that type before writing a line. Every row is measured (real LB/ladder/CSV), not remembered — see the per-competition memory note for the evidence.

**The one-sentence meta across all fronts:** *reproduce the freshest strong public base FIRST (it is the #1 lever by a wide margin), protect it as BEST_KNOWN, and add complexity only when a number says to — most losses were self-inflicted by over-engineering a working base or trusting a proxy that lied.*

---

## Best results on the account (percentile is the honest yardstick)

| Competition | Type | Best REAL result | Decisive lever |
|---|---|---|---|
| playground-series-s6e5 | Tabular | **103/3023 = top 3.41%** (best % on account) | reproduce best public single + OOF diversity |
| stellar-class-s6e6 | Tabular | **206/2817 = top 7.31%** (0.97244) | reproduce fresh ridge-flip public single (+195 places, one I/O move) |
| pokemon-tcg-ai-battle | Sim ladder | **1790/6810 = top 26.3%** (2026-08-19; the 1062.2 peak was a small-sample artefact that settled to 674-770) | reproduce public agent beat own hand-build by +289 — but two later self-built uploads **displaced** it and cost the rank |
| neurogolf-2026 | Code comp | BEST 7129.07 LB (reproduce chain +606 dominant) | reproduce newest AUDITED public base, overlay own true-rule wins |
| jed-agent-security | Red-team code | 51.75 public (~290/1288); dense candidate ~58 ready | reproduce pilkwang dense; BUT public ceiling < private prize |
| rsna-knee-abnormality-detection | DL code comp, shared public weights | **0.920 public, 105/1918 = top 5.5% — MEDAL-ELIGIBLE, best such on the account** | adopt the freshest public frontier, then re-apply one own measured delta on top of each new base |

**Pattern:** the top percentiles (3.41%, 7.31%) are TABULAR Playground, won almost entirely by reproduce-best-public + a little OOF diversity — the highest-ROI front for a solo autonomous practitioner. Simulation/red-team gave lower percentiles and hit ceilings (see below).

**Read percentile together with medal eligibility.** The 3.41% and 7.31% rows are Playground (`awards_points=false`) and earned NOTHING. RSNA's top-5.5% is worth more than both because the competition awards medals. Check `awards_points` before ranking fronts by percentile — this scorecard's own yardstick is misleading without it.

---

## By type — the transferable verdict

### Tabular / Playground — HIGHEST ROI, reproduce-best rules
- **s6e5 (top 3.41%) + s6e6 (top 7.31%)**: both carried by reproducing the strongest *public single* (not a hand-built model), plus modest OOF diversity. s6e6's +195-place jump was ONE move: pull the freshest "ridge-flip + probability-consensus" notebook output and submit it verbatim.
- **Probe the metric before modeling.** s6e6: sample-submission score 0.33333 = 1/3 → metric is `balanced_accuracy`, not accuracy → `class_weight="balanced"` is mandatory. One probe saved days of optimizing the wrong loss.
- **TabPFN is NOT always additive.** s6e6: TabPFN net-HURT the ensemble vs the ridge-flip anchor — measure it in, don't assume. (Contrast: on smaller/cleaner data it is often the strongest single — pool-test, never faith.)
- **A finished front decays.** s6e6 anchor was top-5.1% on 06-17, rank 257 by 06-27 (field grew + a new public method-class appeared). Re-study the field near the deadline; don't coast on a 10-day-old anchor.
- **Trap:** blending YOUR weaker signals onto a working public base = net-hurt. Adopting the strongest public *single* = win. Know the difference.

### Code competition (metric on hidden test via kernel) — reproduce-best + local grader mirror
- **neurogolf-2026**: the whole climb (6287→6507→7114→7128→7129) was reproducing the newest AUDITED single-public base; one reproduce move = **+606**, while 6 autonomous hand-rebuilds combined = **+1.35**. Hand-craft edge exhausts fast (1 win / 14 agent-sweep on genuinely-hard global-rule tasks).
- **Mirror the grader locally.** neurogolf `work/validate.py` = grader-exact replica → every candidate verified for free, no submit spent. JED `predicates.py`+`scoring.py` mirror proved a lever with zero GPU. FIND the shipped scorer first.
- **Cross-dump / cross-base merge COLLAPSES on private.** neurogolf: 44 cheaper graphs from different dumps local-verified 7150 but LB 7119 (−31). Only OWN true-rule graphs on the SINGLE current base transfer. Local validate does NOT catch this.
- **Trap:** slug-poison (a failed `kernels push` kills the slug forever → always a fresh incrementing slug).

### Red-team / agent-security (JED) — reproduce works, but mind the CEILING and the BOARD
- **jed-agent-security**: leaders' lever = replay-DENSE severity-stacking (many http.post(secret) per candidate → K×sev5); the technique sits openly in pilkwang's 213-vote notebook, dense config hidden in its profile menu. Reproduce → ~58 (+6 over our 51.75).
- **★ Wrong-board trap (cost days):** we hardened a PRIVATE-board attack (~10) while the prize gap was on the PUBLIC board (72-100) and our 51.75 was that gap — the private attack would have DROPPED us off the board. Confirm WHICH leaderboard the gap is on + the scoring mechanics (from the SDK) before committing to a lever.
- **★ Reproduce-best has a CEILING = the public frontier.** JED public NBs cap ~58-61; the prize cluster 72-100 is PRIVATE (12 teams, no public NB). Below the public frontier → reproduce (easy, high-ROI). AT the public frontier but below prize → you need ORIGINAL technique (deep research), reproduce is exhausted. Recognize the regime.
- **Eval proxy was systematically BIASED, not just noisy.** The GGUF eval-harness showed 16-42% tool-compliance where the real comp gives ~100% (guide24 = 51.75 = ~570 findings/570 cand). A proxy can be *directionally* wrong — the real comp submit is the only trustworthy compliance test.

### Simulation / agent ladder — one strong agent, reproduce still wins, the arena can mislead
- **★ The rating peak is a sample-size artefact.** One artifact read **1062.2** days after
  submission, **769.4** three weeks later, and **674.1** when resubmitted — 388 points of spread
  with the code unchanged. Public confirmation: two uploads with the SAME SHA-256 sat at 2182.2 and
  1210.1 because each got its own rating instance. Record `(rating, games_played, timestamp)` as one
  unit; never resubmit chasing a remembered peak.
- **★ A ladder submission DISPLACES rather than adds.** Only the last N submissions of a team keep
  playing, so uploading two weak agents retires your strong one — exactly what put us at 1790/6810.
  Fixed-test comps are the opposite (ranking uses your BEST), so the parallel-lineage habit from
  rule 5 is actively harmful here. Ask what each submission evicts.
- **★ Replay distillation is the ladder's "reproduce best public".** Leaders' code is hidden but
  their episodes are public: download the top teams' games, keep the schedule that REPEATS across
  opponents, distil it into a tape, gate paired-seat over untouched seeds. Public gates run 27-3,
  35-5/20 seeds, 90-10/10 seeds. Match episodes to the submission carrying the displayed score — a
  team's newest games are often from their weaker second agent.
- **★ Why our arenas lied three times (mechanism found).** Engine math differs between releases, and
  the version string can pass while `import` resolves an older copy earlier on `sys.path`. The fix is
  a **behavioural fixture**: replay a recorded game and assert the outcome reproduces exactly. Also
  rank with **Bradley-Terry** (what the competition uses), not raw winrate, and build opponents that
  differ in exactly ONE subsystem so a win is attributable.
- **pokemon-tcg**: reproduce public agent (makthanithin 1062.2) beat our own hand-built v3-adaptive (773.3) by +289. Even on a ladder, reproduce-best > hand-build.
- **orbit-wars**: BC-cloning a heuristic teacher reaches teacher-strength and STOPS (clone ≈ teacher − imitation loss); to beat a plateau you need search/RL ABOVE the teacher, not a faster copy. Final lock = v8-double asymmetric (floor + upside).
- **maze-crawler**: (a) run a FRESH death-cause diagnostic before building a fix — the real killer (62% factory_ram) differed from the remembered bottleneck; (b) isolate ONE component of a failed bundle — `ceiling_buffer` alone = +7.1% though the v20 bundle lost.
- **★ Ladder-drift ±200:** ratings drift heavily (pokemon v3 526→792). ONE snapshot is not truth — read a trend, never retire an agent on a single low reading.
- **Trap (paid 3×):** local winrate vs a SINGLE bot does not predict the ladder. Pool of 3-5 distinct styles + a past-you, always.

### CV / DL (ID-doc fraud, debris unlearning) — simple backbone first, then earn complexity
- **freuid-challenge**: simple `convnext_tiny` at 0.354 beat fullres/ensemble variants (0.02-0.15) — over-engineering a working DL base actively destroyed the score. OOD trap: private = unseen doc types + recaptured vs all-digital train → validate BLIND, don't overfit the train distribution.
- **neural-debris-removal**: blind-validation comp (no clean labels) → calibrate any local surrogate against your real submit→LB points FIRST; a pseudo-clean proxy showed Spearman 0.000 (dead) — don't optimize against it.

### DL on a SHARED public weights package — adoption speed is the whole game
- **rsna-knee (0.899 → 0.920, rank 365 → 105 of 1918)**: hundreds of teams mount one public package, so the leaderboard is steps with dense clusters. Our own-ensemble work (retrained members, per-target combiners, an OOF matrix, a planned diffusion-feature extractor) returned **+0.001** while the bronze line moved **0.902 → 0.911 in three days**. Two adoptions of the fresh frontier then returned **+0.011 (195 ranks)** and **+0.005 (133 ranks)**. Watch public notebooks at the SAME cadence as your own runs.
- **★ Carry one measured delta onto every base you adopt.** Reproducing the frontier exactly lands you mid-cluster with everyone else who did. `rank(pct=True)` → raw probability in the member combiner was +0.001 when first measured, and transferred INTACT onto four structurally different later bases (one moved rank 170 → 119). A finding that survives a pipeline change is a property of the metric.
- **★ The public package has a CEILING.** 0.92x is where reproduction tops out; the top-13 at 0.940 train their own models. Silver was one more adoption away, gold needs a different SOLUTION CLASS. Do not plan a GPU budget as if these were the same task.
- **Screen a notebook in four checks before spending a run** (each rejected a real candidate): empty strings in `dataset_sources` = private deps = unreproducible by anyone but the author; empty `kernel_sources` entries = the same for notebooks; last-run date (a strong author can sit on stale code); THEN author rank.
- **★ Author rank is a filter, not an ordering.** It correctly rejected the newest 48-vote notebook (author below us). But a rank-294 author's notebook beat a rank-91 author's (0.915 vs 0.914), and two notebooks by the SAME author scored **0.920 vs 0.911** — authors do not publish what they submit. So run independent lineages in PARALLEL; three times the "backup" won. Ranking uses your BEST submission, so a weak candidate costs a slot, never the position — and a kernel RUN costs no slot at all.
- **Leaderboard CSVs round to 3 decimals**, so apparent 96-team "ties" are display artefacts spanning ~95 real ranks. Inside such a cluster a sub-0.001 lever is exactly the right size, not cosmetic.
- **A flat parameter sweep is a POSITIVE result** — it localises the cause. Three submissions gave 0.911/0.912/0.912; because that constant was measurably spent, the author's remaining +0.005 had to be in the code he had added that morning. Adopting it returned +0.002.
- **Member count is not quality**: 20 members beat 24 by 0.009.
- **Traps:** competitions can FORBID accelerators, quoted at submit time (`cannot use TPU`, `cannot use P100 GPUs`) — a mystery `CUDA error: no kernel image is available` was just a P100. The ~1 MB kernel-push limit counts non-executing markdown (dropping 27 cells took 1,013,322 → 898,117 bytes with the code SHA unchanged; re-minifying JSON made it *worse*).

### Judge-scored hackathon (no LB) — the rubric is the metric, the writeup is half the score
- **triagegeist** (ED-triage, JUDGE-scored, no LB): flagship = "Second Look" safety-net framing; text-only features hit 100% acc = informative-missingness LEAK trap (a field's presence encodes the label) — catch it or the demo is a lie.
- **capstone-agents-for-business** (ADK multi-agent BI, rubric 100pts): decompose the rubric into a weighted checklist; a Verifier agent is the differentiator; submit = writeup + repo + video, presentation ≈ half the score.
- Verdict: no LB means the deliverable is the story + a demoable number, not the biggest model. Allocate effort by rubric weights, not by taste.

---

## The five self-inflicted losses to never repeat (measured across fronts)

1. **Over-engineering past the peak** — cost score in 6 of 8 audited comps (orbit, maze, freuid, s6e6-TabPFN, neurogolf cross-dump, JED-private). A simple working base + STOP beats a clever layer that degrades the metric.
2. **Trusting a proxy that lied** — local arena (sim), pseudo-clean surrogate (debris, Spearman 0), GGUF eval-compliance (JED, 16-42% vs 100%). Calibrate the proxy against real points or don't ship on it.
3. **Refining a stale base** — the public frontier moves in DAYS (neurogolf 6275→7128 in a week; RSNA's bronze line 0.902→0.911 in three days while our own tuning earned +0.001). Re-study the field every 1-2 days; one reproduce beat a whole hand-build cycle. **Watch the frontier at the same cadence as your own experiments, not after them.**
4. **Fabricated/remembered numbers** — pokemon carried invented ladder scores across sessions until a fresh `submissions --csv` corrected them. A number in memory is a CLAIM until the CSV confirms it.
5. **Working the wrong board / wrong window** — JED private vs public; and simulation comps that had submissions DISABLED before the listed deadline. Verify the board and that submissions are OPEN before investing a session.
6. **Testing screened candidates sequentially, best-first** — on RSNA the "backup" beat the primary three times, most sharply 0.920 vs 0.911 from the SAME author on the same day. When slots allow, submit independent lineages in parallel: ranking uses your BEST submission, so a loser costs a slot and never the position, and a kernel RUN costs no slot at all.
7. **Trusting a name, a vote count, or a log line instead of the recorded number** — a bundle that looked like a +0.0076 blend partner carried `promotion_gate_passed: false` and 0.698 in its own receipt; a kernel log printed an alpha the code no longer used; Kaggle's file listing claimed 871 bytes for an 89 MB file. Open the receipt, read the constant in the code, verify size by downloading.
8. **Submitting to a ladder without asking what it displaces** — two self-built agents (708.7, 706.4) evicted a stronger reproduced lineage and left us at 1790/6810 with gold at top 23. On rating ladders a submission is a REPLACEMENT, not an addition.
9. **Opening a campaign without checking `awards_points`** — top 3.41% and top 7.31% finishes earned zero because both were Playground. Percentile is only progress when the competition awards medals; check the flag before the data.
