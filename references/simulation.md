# Simulation / Agent — ladder competitions

> **Depth layer:** `gm-methods.md` section *Simulation, agents & RL ladders* — 18 named methods from Halite/Lux/Kore/ConnectX winners with evidence.

Bot vs bots, ladder rating, episodes. There are **no ensembles and no OOF** — you need one
strong, reliable agent. Logic rules, not volume. Submission limits are tight and the rating
signal is noisy in specific, mechanical ways that have cost this account real rank.

## Contents
- [How it differs from ML competitions](#how-it-differs-from-ml-competitions)
- [Rating mechanics you must model before trusting any number](#rating-mechanics-you-must-model-before-trusting-any-number)
- [Which artifact actually represents you](#which-artifact-actually-represents-you)
- [Optimize the ladder's objective, not a proxy quantity](#optimize-the-ladders-objective-not-a-proxy-quantity)
- [The arena: make it not lie](#the-arena-make-it-not-lie)
- [Replay mining and tape distillation](#replay-mining-and-tape-distillation)
- [Convergence: find where the top DIVERGES](#convergence-find-where-the-top-diverges)
- [How to strengthen the agent](#how-to-strengthen-the-agent)
- [Economic / market simulations](#economic--market-simulations)
- [Simple-and-reliable > clever-and-fragile](#simple-and-reliable--clever-and-fragile)
- [Submissions — early convergence, then a deliberate endgame lottery](#submissions--early-convergence-then-a-deliberate-endgame-lottery)
- [Defending your own replays](#defending-your-own-replays)

---

## How it differs from ML competitions

- The metric is winrate / ladder rank against a live pool of opponents, not a fixed test.
- One agent, not an ensemble. Improvement comes through the algorithm (search, heuristics,
  opponent modeling, schedule quality), not "more models."
- The noise is enormous and **structured**, not just random — see the next section. Without
  modeling how the rating is produced, any single number is worse than useless: it is
  confidently wrong.
- **Reproduce-best-public still wins here.** Measured on this account: a reproduced public
  agent (makthanithin, 1062.2 peak) beat our own hand-built adaptive agent (773.3) by +289.
  The ladder is not an exception to the reproduce-first rule.

## Rating mechanics you must model before trusting any number

Ladder ratings (TrueSkill / Elo / Bradley-Terry) are **estimates that converge with games
played**. Three consequences have each cost this account rank:

**1. An early rating is a small-sample estimate, and its peak is inflated.** A fresh
submission starts at high uncertainty; a lucky opening run produces a peak that does not
survive more games. *Measured (Pokemon TCG): an agent recorded 1062.2 shortly after
submission. The same artifact resubmitted later settled at 674.1, and a snapshot in between
read 769.4. The 1062.2 was never a property of the agent — it was a property of having
played few games.* **Never treat a historical peak as a capability.** Record
`(rating, games_played, timestamp)` as one unit, and compare agents only at comparable
game counts.

**2. Identical code can carry different ratings.** *Measured publicly (Kaggriculture): two
uploads with the same SHA-256 `main.py` sat at 2182.2 and 1210.1 simultaneously, because
Kaggle created a new rating instance for each. The lower one drifted to 1865.6 with no code
change at all.* A rating gap between two copies of the same agent is evidence about sample
size, not about code.

**3. Ratings drift in both directions and take real time to settle.** *Measured (Pokemon,
earlier): the same agent read 526 then 792 on different days.* Submit early so the rating
has time to converge before the deadline; never retire an agent on one low reading.

**Protocol.** Before any promote/retire decision on ladder evidence:
- pull a fresh reading (`kaggle competitions submissions -c <slug> --csv`, or MCP
  `search_competition_submissions`), never a remembered one;
- require a stable window (several readings, or a known game count), not one snapshot;
- back the ladder reading with arena evidence, because the arena you control can be run to
  arbitrary sample size while the ladder cannot.

## Which artifact actually represents you

This is the cheapest rank on the board and the easiest to lose silently.

- Platforms typically play **only the most recent N submissions** of a team (on Kaggle
  simulation comps, the last two). Uploading two weak agents therefore **retires your strong
  one**. *Measured (Pokemon TCG, 2026-08): two self-built agents at 708.7 and 706.4 were
  active while a stronger reproduced agent sat unplayed in history; team rank 1790/6810.*
- Some hosts let you select actively; others take the newest. **Verify which** before
  submitting, then re-read the standings after to confirm the intended artifact is live.
- **Before every submission ask: what does this displace?** A candidate that is not clearly
  better than the artifact it evicts is a negative-expected-value submission even if it is
  interesting.

Keep a one-line ledger per artifact: `id | SHA-256 of main | rating readings with dates and
game counts | currently active? | parent`.

## Optimize the ladder's objective, not a proxy quantity

Read the evaluation page and find out **what the rating consumes**. Frequently it is only
win/loss/tie — the *margin* is discarded.

*Kaggriculture example: rating updates on W/L/T only. An agent banking 140k coins that
loses still loses rating, and an agent printing 100k-170k against the built-in `starter`
can sit mid-ladder.* Optimizing mean score against a weak baseline is then optimizing a
quantity the ladder never reads. Use absolute score as a **filter** (did it crash? is it
in the right order of magnitude?) and head-to-head outcomes as the **objective**.

## The arena: make it not lie

This account has been burned by a misleading local arena **three times** (orbit-wars,
maze-crawler, pokemon-tcg). The public top players avoid this with four specific controls
— adopt all four.

**1. Pin the engine version, then verify the pin BEHAVIOURALLY.** Checking the version
string is not sufficient: `importlib.metadata.version()` reads freshly written package
metadata while `import` can still resolve an older copy earlier on `sys.path`, so a pip
install reports success, the version assertion passes, and you are measuring on the old
engine. *A public top author lost hours comparing measurements taken on three different
engines without realising it.* The fix is a **behavioural fixture**: replay one recorded
game with known inputs and assert the final state reproduces exactly (to the coin / to the
score). A fixture cannot be fooled by a stale import.

Engine math genuinely differs across releases — *in kaggle-environments 1.32.x the
strawberry price cliff is 62 units; in another build it is ~247*. Every number measured on
the wrong build is silently corrupt.

**2. Use a graded reference ladder, not a handful of bots.** Build/obtain a fixed set of
opponents spanning a wide skill range — including a past-best version of yourself and, if
available, a distilled top-meta agent. Beating the built-in starter tells you almost
nothing.

**3. Seat-swapped round robin.** Play every pairing from **both seats** across many seeds.
Seat and seed advantage are large; a one-seat result is not a measurement.

**4. Rank with the model the competition uses — usually Bradley-Terry, not raw winrate.**
Raw winrate treats "beat tier 0 four times" and "beat tier 5 twice" as equal. Bradley-Terry
fits each agent a latent strength from *whom* it beat, so wins over strong opponents count
more, and it is the same family of model used for final standings. Report it on an
Elo-like scale (400 points per 10× strength, mean anchored at 1500) for readability, and
add a small prior (half a phantom win each way) so undefeated agents do not diverge.

**5. Design the ladder to isolate ONE variable.** The strongest public reference set does
this deliberately: tiers 0-5 share a **byte-identical action scheduler** and differ only in
a `POLICY` dict, so any gap is caused by economic decisions alone; tiers 6-9 hold the
production plan constant and differ only in the **market layer**. *"Diff two of these and
the diff is the lesson."* Build your own variants the same way — one changed subsystem per
variant — or you will not be able to attribute a win.

**Promotion gate.** Promote only on a decisive paired-seat record over untouched seeds.
Public top agents gate at roughly this strength: *27-3, 35-5 over 20 seeds both seats,
90-10 over 10 untouched seeds*. A 7-7-2 result means "same policy", not "slightly better".

## Replay mining and tape distillation

The dominant method among Kaggriculture leaders, and it transfers to any comp that
publishes episodes. The idea: **you cannot read the leaders' source, but you can read their
actions**, and a repeated action schedule can be distilled into an agent.

**The pipeline.**
1. Download episodes for the current top teams (MCP: `list_submission_episodes`,
   `get_episode_replay`, `get_episode_agent_logs`; some comps also publish an official
   daily replay dataset indexed by mean rating).
2. **Match the episodes to the submission that produced the displayed score.** A team can
   have two active submissions; naively taking their newest games samples their *weaker*
   newer agent. *A public author hit exactly this bug and had to add a
   `--leader-submission-only` filter matching `publicScore` to the displayed team score.*
3. Look for **repetition across matches**, not brilliance in one. A replay is one match: the
   loser may be excellent and the winner may have had the seed. Accept a tape only if
   (a) the same schedule appears against several opponents, (b) the market/decision schedule
   is stable rather than reactive, and (c) the distilled version beats your previous agent
   **from both seats**.
4. **Deduplicate sources by hash.** *Two public notebooks presented as independent
   confirmations had the same SHA-256 anchor — they were one strategy, not two.*
5. Graft the distilled tape as the new base, keeping your own measured wrappers on top
   (see rule 2 of the learned playbook).

**Loss audit.** Where replays of your own losses exist, dissect them for systematic
mistakes rather than tuning blindly. Public leaders run explicit "live-loss audits" and
report them as the source of concrete promotions (opening feed denial, one-turn preemption).

## Convergence: find where the top DIVERGES

When you distil many leaders, they frequently **converge** on the same plan. *Kaggriculture:
across the top ten teams, mature farms were near-identical — 8 cows, 6 sheep, 3 quadrants,
12 hands, and similar seed counts; many tapes differed on only 2-5 decision turns.*

That is the most actionable recon result you can get, because it tells you the remaining
edge is **not** in the converged subsystem:

- Compare a promoted candidate to its parent by **counting changed turns per subsystem**.
  *The promotion from c16 to c18 changed 20 field turns but **112 market turns** — proof the
  live edge was in sale timing, not herd composition.*
- Swap subsystems between agents to attribute strength (*A's field + B's market vs the
  reverse*). If one direction is competitive and the other is weak, you have located the
  carrier.
- A converged meta also means **correlated risk**: everyone forking one public family loses
  together to whatever counters it.

## How to strengthen the agent

Measure every step against the reference ladder:
- **Search:** BFS / A* / minimax / MCTS / N-step lookahead instead of greedy.
- **Opponent modeling:** predict their move, play the best response; detect mirror matches
  explicitly. *A "clone-aware" wrapper that checks whether both farms are near-identical and
  then front-runs the shared dump by one turn produced 6-0 with a +1,865.7 mean margin — a
  small wrapper with a large mirror-match effect.*
- **Memory / state:** track the map, resources, opponent patterns across turns.
- **Terminal handling:** the last turns are cheap to get wrong and cheap to fix. *Discovering
  that step 718 executes while action index 719 does not, and delaying a terminal controller
  from step 712 to 717, produced a 90-10 gate from a one-line change.* Always check the exact
  first/last executable step.
- **Heuristic tuning:** objective-function weights — but confirm against the pool, not "logic".

Each complication is a separate version run against the same ladder. Helped measurably →
keep. Didn't → throw out. **Beware the clone ceiling:** *behaviour-cloning a heuristic teacher
reaches teacher-strength and stops (orbit-wars) — to pass a plateau you need search/RL above
the teacher, not a faster copy.*

## Economic / market simulations

A distinct and increasingly common subclass (Kaggriculture, trading/market ladders) where
**your own actions move the environment**. Generic agent advice underfits these; check each:

- **Your sale depresses the price.** Dumping inventory means the first units sell high and
  the rest hit the floor. Sell in **small metered batches** while the price holds. *Premium
  goods crash to the $1 floor after ~60-80 units of oversupply; ballast goods need ~3000.*
  Know each product's crash depth before writing the sell rule.
- **Action economy beats the payoff table.** Each unit gets one action per turn, so a
  high-margin option costing two extra moves can lose to a weaker one next door. Optimize
  **profit per action-turn**, not profit per item.
- **Inventory ≠ score.** *Unsold inventory does not count toward the bank; overflow past the
  shed cap is destroyed.* Liquidation timing is part of the strategy, not an afterthought.
- **Read the pipeline between subsystems.** *`HARVEST` puts items in a unit's inventory but
  `SELL` only sees the shed* — a missing transfer step silently strands your entire output.
- **Look for free income loops the brief does not advertise.** *Fertilizer is described as
  something you buy, yet the generic sell path accepts it, and each animal produces one per
  day — a real sidecar income stream.*
- **Compute the yield/bonus windows from the engine, not the overview table.** *Melon's
  documented `max_yield_day = 12` actually implies a watering window of ages 6-12 that caps
  out around age 10; missing water on days 6-10 silently returns ~70 units instead of ~96.*
- **Front-load irreversible investments.** Land, capacity and labor compound over the
  horizon; buying them late wastes the compounding. Equally, check which investments never
  repay inside the horizon (*the 4k quadrant is bought by ≈0% of top players in a 30-day
  game*).

## Simple-and-reliable > clever-and-fragile

Before complicating, check whether the complexity breaks a simple working strategy. A common
trap: lookahead/MCTS sounds great but times out, hits a bug at the map edge, or loses to a
dumb-but-stable bot. A simple agent that never crashes and makes no gross mistakes beats a
fragile genius. Stability is part of strength. Watch the per-move runtime limit — a timeout
usually counts as a loss/invalid.

## Submissions — early convergence, then a deliberate endgame lottery

Three rules that look contradictory and are not. Hold all three.

**1. Mid-campaign: submit only what beats the current best, and know what it displaces.**
The limit is small → don't spend one on an unverified "seems a bit better". Arena first,
ladder second. Re-read standings afterwards to confirm the intended artifact is live.

**2. Submit your strong agent EARLY.** Ratings converge with games played, so an agent
submitted late is still a small-sample estimate when the competition closes — and per the
rating mechanics above, that estimate is biased by whatever opponents it happened to draw.

**3. At the deadline, spend every remaining slot on VARIANTS OF THE STRONG AGENT.** Because
each upload gets its own rating instance, filling your slots creates several independent
rating draws of a competitive agent, and your final rank takes the best of them. *Two
independent 1st places report exactly this: Santa 2020's winner "submitted the maximum number
of agents to make the most of the luck factor" (1st vs 2nd was decided by ~1.7 rating points,
1536.2 vs 1534.5); Halite's winner submitted 22 agents, of which 16 would each have taken
first place on their own.*

**Why 3 does not contradict the displacement rule.** Displacement hurts when a *weaker* agent
evicts a stronger one. Filling the slots with clones or near-clones of your best agent evicts
nothing of value and buys extra lottery tickets on a noisy ladder. The rule is therefore:
never let a weak candidate occupy a slot, and never leave a slot empty at the deadline.
This only widens the lottery around an already-competitive agent — it cannot rescue a weak one.

## Defending your own replays

Replay distillation cuts both ways: if you can mine the leaders, the leaders (and their
imitation-learning agents) can mine you. The current state of the art in defence is
**sandbagging** — package a proven "safe" agent and your strongest candidate in ONE submission,
play the safe one for most matches (a 1st-place team used 85/15), log which one played each
match, and recover the strong agent's true win rate offline by joining those logs to public
match results. *Lux AI Season 3 1st place used this to hold rank while collecting ~1,000
matches/day of honest signal on the hidden model, and to stop a rival imitation-learning agent
from cleanly attributing wins to one strategy.*

The cost is real: your displayed rank is capped near what the safe agent alone achieves, so
this is only viable when the safe agent is already competitive, and only on platforms that
expose enough per-match side-channel data to reconstruct the true rates afterwards.
