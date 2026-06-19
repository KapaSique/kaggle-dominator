#!/usr/bin/env python3
"""pareto_select.py — Pareto-frontier candidate selection for kaggle-dominator.

Borrowed & adapted from GEPA's ParetoCandidateSelector (gepa-ai/gepa, ICLR 2026
Oral) and EvoSkill's held-out Pareto frontier (sentient-agi/EvoSkill). Stdlib-only.

WHY: keeping ONE global BEST_KNOWN lets a candidate that's better ON AVERAGE silently
replace a proven peak — and conversely discards a candidate that's worse on average
but best on some regime. We instead keep the PARETO FRONT: a candidate survives iff no
other candidate DOMINATES it (>= on every instance AND > on at least one). For Kaggle,
"instances" = pool opponents / seeds / CV folds / metric axes.

This is exactly the guard the v3/search failures needed: a challenger may REPLACE the
active BEST_KNOWN only if it Pareto-DOMINATES it on the real pool — not if it merely
wins on average or on a single self-mirror (which lied 3x: orbit v11, maze v20, pokemon
v3). Pareto-dominance is scale-free and computed per-instance, so mixed axes (winrate,
ladder rating, CV) are compared element-wise without normalization.

Usage:
  pareto_select.py scores.json [--best CURRENT_BEST_NAME]
where scores.json = {"candidate": {"instance_label": score, ...}, ...}
Higher score = better on every instance (flip signs upstream if lower=better).
"""
import json
import sys


def dominates(a, b, instances):
    """a dominates b iff a >= b on every instance and a > b on at least one."""
    ge_all = all(a[i] >= b[i] for i in instances)
    gt_any = any(a[i] > b[i] for i in instances)
    return ge_all and gt_any


def pareto_front(scores):
    cands = list(scores)
    instances = sorted({i for c in scores for i in scores[c]})
    for c in cands:                       # missing measurement = worst possible
        for i in instances:
            scores[c].setdefault(i, float("-inf"))
    front = [c for c in cands
             if not any(dominates(scores[o], scores[c], instances) for o in cands if o != c)]
    return front, instances


def main():
    if len(sys.argv) < 2:
        print("usage: pareto_select.py scores.json [--best CURRENT_BEST_NAME]")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        scores = json.load(f)
    best = sys.argv[sys.argv.index("--best") + 1] if "--best" in sys.argv else None

    front, instances = pareto_front(scores)
    agg = {c: sum(scores[c][i] for i in instances) / len(instances) for c in scores}

    print(f"Instances ({len(instances)}): {', '.join(instances)}")
    print(f"\nPareto front ({len(front)}/{len(scores)} candidates — non-dominated):")
    for c in sorted(front, key=lambda x: agg[x], reverse=True):
        per = {i: round(scores[c][i], 4) for i in instances}
        print(f"  {c}: agg≈{agg[c]:.4f}  {per}")
    dominated = [c for c in scores if c not in front]
    if dominated:
        print("\nDominated (discard — strictly beaten on every axis by a front member):")
        for c in sorted(dominated, key=lambda x: agg[x], reverse=True):
            print(f"  {c}: agg≈{agg[c]:.4f}")

    print("\n--- recommendation ---")
    if best is not None:
        if best not in scores:
            print(f"BEST '{best}' not in scores — measure it on the SAME pool first.")
        elif any(dominates(scores[c], scores[best], instances) for c in front if c != best):
            dom = [c for c in front if c != best and dominates(scores[c], scores[best], instances)]
            print(f"REPLACE: {dom} Pareto-DOMINATE current best '{best}' → safe, measured upgrade.")
        elif best in front:
            others = [c for c in front if c != best]
            tag = f" Co-optimal (diversity, NOT a replacement): {others}" if others else " Sole front member."
            print(f"HOLD '{best}': on the Pareto front.{tag}")
            print("  Rule: never replace a front-member best with a non-dominating challenger — that's the v3/search trap.")
        else:
            dom = [c for c in front if dominates(scores[c], scores[best], instances)]
            print(f"WARNING: current best '{best}' is DOMINATED by {dom} — it was never the real peak. Re-examine pool/metric.")
    else:
        if len(front) == 1:
            print(f"Single non-dominated candidate: {front[0]} → clear pick.")
        else:
            print(f"No single dominator — {len(front)} co-optimal candidates. KEEP THE FRONT (diversity); "
                  "don't collapse to the average-best (GEPA / hill-climbing doctrine).")


if __name__ == "__main__":
    main()
