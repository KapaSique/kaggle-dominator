# Simulation / Agent — ladder competitions

Bot vs bots, ladder rating, episodes. Here there are **no ensembles and no OOF** — you need one strong, reliable agent. Logic rules, not volume. Submission limits are tight — every one counts.

## Contents
- [How it differs from ML competitions](#how-it-differs-from-ml-competitions)
- [The cardinal validation rule](#the-cardinal-validation-rule)
- [How to strengthen the agent](#how-to-strengthen-the-agent)
- [Simple-and-reliable > clever-and-fragile](#simple-and-reliable--clever-and-fragile)
- [The local arena](#the-local-arena)
- [Submissions](#submissions)

---

## How it differs from ML competitions

- The metric is winrate / ladder rank against a live pool of opponents, not a fixed test.
- One agent, not an ensemble. Improvement comes through the algorithm (search, heuristics, opponent modeling), not "more models."
- The noise is enormous: the same agent gives different outcomes in two matches. Without multi-match evaluation, any number is random.

## The cardinal validation rule

**NEVER judge an agent against a single opponent.** One bot = a random number, not signal. Minimum **3–5 different styles** (aggressive, passive, random, a past version of yourself, a top public bot). Run dozens/hundreds of matches against the pool, look at mean winrate ± spread.

The local arena shows 70% but the ladder sags → **the arena isn't representative, fix it first** (wrong opponent pool, wrong map/seed, wrong engine version). Don't tune against a broken proxy.

## How to strengthen the agent

Through the algorithm, measuring each step against the pool:
- **Search:** BFS / A* / minimax / MCTS / lookahead N moves ahead instead of greedy.
- **Opponent modeling:** predict its move, play the best response.
- **Memory / state:** track the map, resources, the enemy's patterns across turns.
- **Heuristic tuning:** weights of the objective function — but confirm with winrate against the pool, not "logic."

Each complication is a separate version, run against the same pool. Helped measurably → keep it. Didn't → throw it out.

## Simple-and-reliable > clever-and-fragile

Before complicating — check whether the complexity breaks a simple working strategy. A common trap: lookahead/MCTS sounds great but times out, hits a bug at the map edge, or loses to a dumb-but-stable bot. A simple agent that never crashes and makes no gross mistakes beats a fragile genius. Stability is part of strength.

Watch the per-move runtime limit — a timeout usually counts as a loss/invalid.

## The local arena

- Build a harness: your agent × the opponent pool × many matches × different seeds/maps → a winrate table.
- Include a **past best version of yourself** in the pool — the direct answer to "is the new one actually stronger?"
- Run the arena on a kernel if it's heavy; light ones locally (this is orchestration, not training).
- If the competition has episode logs/replays (e.g. MCP `list_submission_episodes`, `get_episode_replay`) — dissect the losses, look for systematic mistakes.

## Submissions

- Submit only a version that is **measurably no worse** than the current best on the pool. The BEST_KNOWN agent is your rank insurance.
- The limit is small → don't spend a submission on an unverified "seems a bit better." Arena first, then ladder.
- The ladder moves slowly (it needs time to play matches) — submit early, not in the final hour.
