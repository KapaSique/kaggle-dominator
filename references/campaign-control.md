# Campaign control

Use this reference whenever several competitions compete for the same time, GPU,
submission, storage, or agent-attention budgets—or when the user explicitly names
one primary front.

## Campaign record

Keep one machine-readable record per tracked competition. YAML is convenient:

```yaml
slug: example-competition
state: ACTIVE
updated_at_utc: 2026-07-11T00:00:00Z
deadline_utc: 2026-08-01T23:59:00Z
submissions_enabled: true
metric:
  name: auc
  direction: maximize
position:
  rank: 123
  teams: 2500
  score: 0.9012
  leader_score: 0.9173
  target_rank: 10
  target_score: 0.9120
best_known:
  artifact: submissions/v17.csv
  score: 0.9012
  measured_at_utc: 2026-07-10T12:00:00Z
validation:
  protocol: 5-fold stratified OOF
  score: 0.9041
  lb_correlation_evidence: 6 matched experiments
budgets:
  hours_per_day: 8
  gpu_hours_per_day: 20
  submissions_per_day: 2
  submissions_remaining_today: 2
  storage_gb: 20
  agent_slots: 3
queue:
  - id: v18
    hypothesis: diverse model family improves ensemble
    status: local_gate
resume_trigger: null
```

Never copy example values into a live record. Unknowns remain `unknown` until a
fresh source confirms them.

## States

- `ACTIVE`: build and experiment within its recorded budgets.
- `MONITOR_ONLY`: refresh rank, score, deadline, submission availability, and new
  public-frontier signals. Do not train, launch kernels, or submit.
- `PAUSED`: preserve without routine polling. Add a precise resume trigger such as
  “new public baseline exceeds X,” “GPU quota restored,” or “user resumes.”
- `CLOSED`: archive final public/private results, artifacts, costs, and measured
  transferable lessons.

State transitions must record timestamp and reason. A user directive to put all
resources on one competition means exactly one `ACTIVE` campaign; demote the rest
to `MONITOR_ONLY` unless the user explicitly asks to stop monitoring too.

## Allocation algorithm

For each open front, estimate:

- **Value**: prize/medal/learning value and user priority.
- **Reachability**: gap to target, time remaining, validation quality, and available
  levers—not optimism.
- **Marginal return**: expected leaderboard or information gain from the next unit
  of time, compute, attention, or submission quota.
- **Urgency**: deadline proximity, expiring compute, or a transient public-frontier
  opportunity.
- **Risk**: weak validation, high shake-up probability, rule ambiguity, brittle
  inference, or poor reproducibility.

Use the qualitative score:

`priority = user_priority × value × reachability × marginal_return × urgency / risk`

The formula enforces the questions; fake precision is unnecessary. Cite the facts
behind each factor. The user's explicit campaign priority overrides the ranking,
but does not override compliance or authority gates.

## Resource envelope

Allocate five budgets separately:

1. **Attention** — research, coding, review, and agent slots.
2. **Compute** — GPU/CPU hours and concurrent Kaggle kernels.
3. **Submissions** — daily and total quota; the scarcest information channel.
4. **Storage/network** — datasets, checkpoints, outputs, and upload time.
5. **Deadline slack** — time reserved for clean reruns, failure recovery, and final
   submission verification.

Do not call a campaign “100% focused” while background jobs, scheduled workflows,
or autonomous agents still consume non-monitoring budgets elsewhere. Disable or
gate them and record the change.

## Candidate queue

Keep candidates in a queue with these fields:

```text
id | hypothesis | parent | expected upside | information value | cost
validation gate | status | result | decision | artifact
```

Prioritize high information per unit cost early and high confidence per scarce
submission near the deadline. De-duplicate candidates that test the same mechanism.

## Submission policy

Treat every submission as an experiment with an opportunity cost.

Submit only when:

- the file and provenance checks pass;
- it tests a decision-relevant hypothesis or is a strong final candidate;
- the relevant offline/arena gate passes, or the submission is an explicitly
  approved metric-calibration probe;
- the remaining quota still covers higher-priority candidates and final insurance;
- explicit submission authority exists.

Recommended reserve:

- early campaign: spend on validation calibration and large method-class deltas;
- middle campaign: spend on promoted candidates and ensemble ablations;
- final phase: reserve at least one safe reproducible candidate and enough time to
  recover from a failed kernel or malformed artifact.

Never submit several nearly identical variants just because quota remains.

## Stop, continue, or pivot

Continue a branch when at least one is true:

- a predeclared metric improved beyond noise;
- it adds validated ensemble diversity;
- it falsifies a high-value uncertainty cheaply;
- recon reveals a newer public frontier or unexplored method class.

Stop or pivot when:

- repeated controlled tests show no gain and no diversity;
- local validation is not predictive and the next work does not repair it;
- the branch's expected upside is smaller than measurement noise or opportunity
  cost;
- rules, runtime, data access, or deadline make the result infeasible;
- a stronger reproducible baseline dominates it on score, robustness, and cost.

A plateau is a reason to change the method class or improve recon, not to repeat
the same tuning with more seeds.

## Monitoring report

For `MONITOR_ONLY` fronts, report only:

- fresh rank/teams and best score;
- leader and target-cutoff gaps;
- deadline/submission-status changes;
- material new public baseline or rule change;
- exact trigger that would justify reactivation.

Do not generate candidate code, launch kernels, or consume submissions during a
monitor-only refresh.
