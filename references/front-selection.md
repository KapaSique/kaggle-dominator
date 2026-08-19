# Front selection — which competition to enter at all

The highest-leverage decision on the board is made before any modeling: **which front you
spend the campaign on**. This account's single largest measured loss was not a bad model —
it was investing top-percentile effort into competitions that award nothing. Run this
protocol before opening a new front, and re-run it when a campaign ends.

## Contents
- [The eligibility gate](#the-eligibility-gate-run-this-first)
- [Medal thresholds](#medal-thresholds-what-gold-actually-costs)
- [Freshness beats deadline proximity](#freshness-beats-deadline-proximity)
- [Can you still enter](#can-you-still-enter)
- [Format fit](#format-fit-against-your-real-constraints)
- [The scan](#the-scan-one-command)
- [Reading the leaderboard shape](#reading-the-leaderboard-shape)
- [Decision record](#decision-record)

---

## The eligibility gate (run this FIRST)

**A competition awards medals only if `awards_points` is true.** Check this before anything
else — before the data, before the metric, before the public notebooks.

*Measured on this account: `playground-series-s6e5` finished **103/3023 = top 3.41%** and
`stellar-class-s6e6` finished **206/2817 = top 7.31%**. Both are the best percentile results
on the account. Both are Playground, both have `awards_points=false`, and both earned
**nothing** — the account holds zero medals despite top-3.41% execution.*

Categories that award medals: `Featured`, `Research`, `Recruitment`, `Masters`.
Categories that do not: `Playground`, `Getting Started`, `Community`, `Analytics`, and most
judge-scored writeup tracks — **including some inside a Featured competition**. A Featured
prize track can be writeup-judged and medal-less while its sibling ladder awards medals;
check the flag per slug, never per event.

Playground is still legitimate as *practice or method calibration*. It is not legitimate as
a medal campaign. Label it as such in the campaign record so percentile results are never
reported as progress toward a medal.

## Medal thresholds — what gold actually costs

Kaggle's thresholds scale with the field, so "top 20" means different things in different
competitions:

| Teams | Bronze | Silver | Gold |
|---|---|---|---|
| 0-99 | top 40% | top 20% | top 10% |
| 100-249 | top 40% | top 20% | **top 10** |
| 250-999 | top 100 | top 50 | **top 10 + 0.2%×teams** |
| 1000+ | top 10% | top 5% | **top 10 + 0.2%×teams** |

Consequences that change strategy:

- **Gold is nearly flat in absolute rank while bronze/silver scale with the field.** At 1918
  teams gold is top 13; at 5232 teams it is top 20. Entering a bigger competition barely
  raises the gold bar but massively raises the bronze/silver bar.
- **The field grows until the deadline**, so compute thresholds against a *projected* team
  count, not today's. A front that is 4700 teams now may settle above 6000.
- **Silver is often one adoption away when gold is a different solution class.** *Measured
  (RSNA Knee): rank 106/1918 at 0.920 public. Silver (#96) required +0.001; gold (#13)
  required +0.020 — the first is another frontier adoption, the second needs own training.*
  Price these as two different projects and say which one you are funding.

## Freshness beats deadline proximity

The intuition "pick a competition ending soon so I can sprint into the top" is **backwards**.
A competition near its deadline has a mature public frontier, a settled meta, and hundreds of
teams already stacked on the best public package. Your marginal contribution is smallest
exactly when the deadline is closest.

The productive window is a competition that **started recently and still runs for weeks**:
the public frontier is unsettled, method classes are still being discovered, and an early
adopter compounds.

Compute both numbers for every candidate:

- `age_days = now - enabled_date` — how settled the frontier is.
- `days_left = deadline - now` — how much compounding you can still buy.

Prefer small `age_days` with comfortable `days_left`. Treat a front with large `age_days` and
small `days_left` as an adoption sprint at best, never as a build.

**Exception worth checking:** a stale-looking front where the leaderboard shows a *plateau*
(many teams on one score) is a shared-public-package front — there, late adoption plus one
carried delta is still a real lever. See `learned-playbook.md` rules 1-2.

## Can you still enter

`new_entrant_deadline` closes entry **before** the final deadline (commonly one week). A
competition can be open on the leaderboard and closed to you. Check it in the same call as
the deadline, and check `submissions_disabled` — some simulation comps stop accepting
submissions before the listed end.

Also check `max_team_size` and `merger_deadline` if teaming is on the table.

## Format fit against your real constraints

Score each candidate against your actual envelope, not an idealized one:

- `is_kernels_submissions_only` — code competition: hidden test, runtime limits, internet
  usually off. Read `code-and-hackathon.md` before committing.
- `max_daily_submissions` — the information channel width. A front with 1/day (e.g. ARC)
  gives you a tiny number of real measurements for the whole campaign; a front with 5/day
  supports parallel lineage testing (learned-playbook rule 5).
- **Accelerator demand.** If all heavy compute must run on Kaggle kernels, a 3D-video or
  large-DL front consumes the entire weekly quota for one lineage. An agent/simulation front
  usually needs **no accelerator at all**, which makes it the cheapest medal-eligible format
  under a constrained compute budget.
- **Domain experience already on the account.** Read `scorecard.md` for the measured verdict
  per type before assuming a front suits you.

## The scan — one command

```bash
python3 - <<'EOF'
import os, datetime
os.environ.setdefault("KAGGLE_USERNAME", "<your-username>")
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
now = datetime.datetime.now(datetime.timezone.utc)
def tz(d):
    return None if d is None else (d.replace(tzinfo=datetime.timezone.utc) if d.tzinfo is None else d)
def gold(t):
    if not t: return "?"
    if t < 100:  return f"top {max(1, round(t*0.10))}"
    if t < 250:  return "top 10"
    return f"top {10 + int(0.002*t)}"
rows = []
for cat in ["featured", "research", "recruitment", "masters"]:   # medal-bearing only
    for page in (1, 2, 3):
        r = api.competitions_list(category=cat, sort_by="latestDeadline", page=page)
        cs = r.competitions if hasattr(r, "competitions") else r
        if not cs: break
        for c in cs:
            d = tz(c.deadline)
            if d is None or d < now: continue
            ne = tz(c.new_entrant_deadline); ed = tz(c.enabled_date)
            rows.append((
                (d-now).days, d.strftime("%Y-%m-%d"), (now-ed).days if ed else None,
                c.team_count, gold(c.team_count), bool(c.awards_points),
                (ne is None or ne > now), bool(c.is_kernels_submissions_only),
                c.max_daily_submissions, c.ref.split("/")[-1], c.reward))
seen = set()
print(f"{'left':>5} {'deadline':<11} {'age':>4} {'teams':>6} {'gold':<9} {'M':<2} {'open':<5} {'K':<2} {'d/day':>5}  slug | reward")
for r in sorted(set(rows)):
    if r[9] in seen: continue
    seen.add(r[9])
    print(f"{r[0]:>5} {r[1]:<11} {str(r[2]):>4} {str(r[3]):>6} {r[4]:<9} "
          f"{'Y' if r[5] else '-':<2} {'Y' if r[6] else 'NO':<5} {'Y' if r[7] else '-':<2} "
          f"{str(r[8]):>5}  {r[9]} | {r[10]}")
EOF
```

Via MCP, `get_competition` returns the same fields for one slug in a single call
(`awards_points`, `new_entrant_deadline`, `enabled_date`, `max_daily_submissions`,
`is_kernels_submissions_only`, `team_count`) and is the faster path once a shortlist exists.
`search_competitions` covers the listing side. Note that the MCP server requires its own
authorization; the CLI path above works from `~/.kaggle/kaggle.json` alone, so keep it as the
fallback.

## Reading the leaderboard shape

Pull the top ~100 rows before committing. The shape tells you what kind of work is required:

- **Dense plateau near the medal line** (many teams within ~0.001) → shared-public-package
  front. Adoption speed plus one carried delta is the lever; sub-0.001 improvements are
  correctly sized, not cosmetic. Leaderboard CSVs round to 3 decimals, so visible "ties" can
  span ~95 real ranks.
- **A long smooth gradient** → genuine modeling differences; expect to build.
- **One or two outliers far above a flat field** → someone found a method class nobody else
  has. Recon that specifically; grinding the plateau will not reach them.
- **Very low absolute scores across the whole board** (e.g. leader at 2.8/100) → the problem
  is unsolved and the frontier is open. High risk, high ceiling, and public solutions are
  worth little.

## Decision record

Write the choice down with its evidence, so the next session does not re-litigate it:

```yaml
slug: <competition>
chosen_at_utc: <timestamp>
awards_points: true            # gate — never open a medal campaign on false
category: Featured
teams_now: 5232
gold_threshold: top 20         # 10 + 0.2%*teams
age_days: 19                   # young frontier
days_left: 42
entry_closes: 2026-09-23       # new_entrant_deadline
kernels_only: false
daily_submissions: 5
accelerator_need: none
why_this_front: <one sentence tying value, reachability and constraints together>
displaced: <which front drops to MONITOR_ONLY to pay for this>
```

Every open front costs attention even when idle. Opening one means demoting another —
record which, or the campaign silently becomes the split-attention pattern that produced
zero medals from top-percentile work.
