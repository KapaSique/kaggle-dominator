#!/usr/bin/env python3
"""Append-only evidence registry for the kaggle-dominator evolution engine."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
from typing import Any


class EvolutionError(ValueError):
    """Raised when evolution state or input does not meet its contract."""


REQUIRED_EVIDENCE_FIELDS = frozenset(
    {
        "candidate_id",
        "parent_id",
        "competition",
        "competition_type",
        "claim",
        "scope_limits",
        "metric",
        "direction",
        "metric_direction_verified",
        "baseline_score",
        "candidate_score",
        "noise_floor",
        "confirmations",
        "validation_regimes",
        "code_sha",
        "data_fingerprint",
        "config_hash",
        "seeds",
        "runtime_minutes",
        "runtime_ratio",
        "vram_gb",
        "artifacts",
        "regressions",
        "forbidden_actions",
        "changed_paths",
        "transferable",
        "status",
        "created_at_utc",
    }
)

STRING_EVIDENCE_FIELDS = frozenset(
    {
        "candidate_id",
        "parent_id",
        "competition",
        "competition_type",
        "claim",
        "scope_limits",
        "metric",
        "code_sha",
        "data_fingerprint",
        "config_hash",
        "status",
    }
)

NUMBER_EVIDENCE_FIELDS = frozenset(
    {
        "baseline_score",
        "candidate_score",
        "noise_floor",
        "runtime_minutes",
        "runtime_ratio",
        "vram_gb",
    }
)

STRING_LIST_EVIDENCE_FIELDS = frozenset(
    {
        "validation_regimes",
        "artifacts",
        "regressions",
        "forbidden_actions",
        "changed_paths",
    }
)

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def read_json(path: Path) -> dict:
    """Read a JSON object or raise a contract-specific error."""
    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise EvolutionError(f"invalid JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise EvolutionError(f"JSON at {path} must contain an object")
    return payload


def read_jsonl(path: Path) -> list[dict]:
    """Read an append-only JSONL file; a missing ledger is empty."""
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise EvolutionError(f"cannot read JSONL at {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvolutionError(
                f"invalid JSONL at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise EvolutionError(f"JSONL at {path}:{line_number} must contain objects")
        records.append(payload)
    return records


def append_jsonl(path: Path, payload: dict) -> None:
    """Append one JSON object without rewriting previous ledger events."""
    if not isinstance(payload, dict):
        raise EvolutionError("JSONL payload must be an object")
    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise EvolutionError(f"JSONL payload is not serializable: {error}") from error
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
    except OSError as error:
        raise EvolutionError(f"cannot append JSONL at {path}: {error}") from error


def _normalize_utc(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvolutionError("UTC timestamp must be a non-empty string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise EvolutionError(f"invalid UTC timestamp {value!r}") from error
    if parsed.tzinfo is None:
        raise EvolutionError("UTC timestamp must include an explicit timezone")
    normalized = parsed.astimezone(timezone.utc)
    return normalized.isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_timestamps(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _normalize_utc(value) if key.endswith("_at_utc") else _normalize_timestamps(value)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [_normalize_timestamps(item) for item in payload]
    return payload


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_evidence(evidence: dict) -> None:
    """Validate the evaluated-candidate evidence schema before it enters a ledger."""
    if not isinstance(evidence, dict):
        raise EvolutionError("evidence must be an object")
    missing = sorted(REQUIRED_EVIDENCE_FIELDS.difference(evidence))
    if missing:
        raise EvolutionError(f"evidence is missing required fields: {', '.join(missing)}")
    for field in STRING_EVIDENCE_FIELDS:
        value = evidence[field]
        if not isinstance(value, str) or not value.strip():
            raise EvolutionError(f"evidence.{field} must be a non-empty string")
    if not isinstance(evidence["direction"], str) or evidence["direction"] not in {
        "higher",
        "lower",
    }:
        raise EvolutionError("evidence.direction must be 'higher' or 'lower'")
    for field in NUMBER_EVIDENCE_FIELDS:
        if not _is_number(evidence[field]):
            raise EvolutionError(f"evidence.{field} must be a number")
    if not isinstance(evidence["confirmations"], int) or isinstance(
        evidence["confirmations"], bool
    ):
        raise EvolutionError("evidence.confirmations must be an integer")
    for field in STRING_LIST_EVIDENCE_FIELDS:
        value = evidence[field]
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value
        ):
            raise EvolutionError(f"evidence.{field} must be a list of non-empty strings")
    if not isinstance(evidence["seeds"], list) or not all(
        _is_number(seed) for seed in evidence["seeds"]
    ):
        raise EvolutionError("evidence.seeds must be a list of numbers")
    for field in ("metric_direction_verified", "transferable"):
        if not isinstance(evidence[field], bool):
            raise EvolutionError(f"evidence.{field} must be a boolean")
    _normalize_utc(evidence["created_at_utc"])
    try:
        json.dumps(evidence, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise EvolutionError(f"evidence must be JSON serializable: {error}") from error


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class EvolutionStore:
    """Local, event-sourced store for immutable manifests and candidate evidence."""

    def __init__(self, root: Path, skill_root: Path) -> None:
        self.root = Path(root)
        self.skill_root = Path(skill_root)

    @property
    def _evidence_path(self) -> Path:
        return self.root / "evidence.jsonl"

    @property
    def _ledger_path(self) -> Path:
        return self.root / "ledger.jsonl"

    def save_manifest(self, manifest: dict) -> Path:
        """Persist a run manifest under its stable run identifier."""
        if not isinstance(manifest, dict):
            raise EvolutionError("manifest must be an object")
        run_id = manifest.get("run_id")
        if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
            raise EvolutionError("manifest.run_id must be a safe non-empty identifier")
        normalized = _normalize_timestamps(manifest)
        try:
            encoded = json.dumps(normalized, indent=2, sort_keys=True, allow_nan=False) + "\n"
        except (TypeError, ValueError) as error:
            raise EvolutionError(f"manifest must be JSON serializable: {error}") from error
        destination = self.root / "manifests" / f"{run_id}.json"
        if destination.exists():
            if read_json(destination) == normalized:
                return destination
            raise EvolutionError(f"manifest {run_id!r} is immutable and already exists")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(encoded, encoding="utf-8")
        except OSError as error:
            raise EvolutionError(f"cannot save manifest at {destination}: {error}") from error
        return destination

    def record_evidence(self, evidence: dict) -> dict:
        """Record evidence once, followed by its OBSERVED and EVALUATED events."""
        validate_evidence(evidence)
        normalized = _normalize_timestamps(evidence)
        candidate_id = normalized["candidate_id"]
        for existing in read_jsonl(self._evidence_path):
            if existing.get("candidate_id") == candidate_id:
                immutable_evidence = {
                    key: value for key, value in existing.items() if key != "state"
                }
                validate_evidence(immutable_evidence)
                if existing.get("state") != "EVALUATED":
                    raise EvolutionError(
                        f"stored evidence for {candidate_id!r} has an invalid state"
                    )
                self._repair_evaluated_events(candidate_id)
                return existing

        record = {**normalized, "state": "EVALUATED"}
        append_jsonl(self._evidence_path, record)
        self._repair_evaluated_events(candidate_id)
        return record

    def _repair_evaluated_events(self, candidate_id: str) -> None:
        """Append only the missing valid suffix of a candidate's event sequence."""
        expected_events = ("OBSERVED", "EVALUATED")
        recorded_events = [
            event.get("event")
            for event in read_jsonl(self._ledger_path)
            if event.get("candidate_id") == candidate_id
        ]
        if tuple(recorded_events) != expected_events[: len(recorded_events)]:
            raise EvolutionError(f"invalid ledger history for {candidate_id!r}")
        for event in expected_events[len(recorded_events) :]:
            append_jsonl(
                self._ledger_path,
                {
                    "candidate_id": candidate_id,
                    "event": event,
                    "occurred_at_utc": _now_utc(),
                },
            )

    def latest_state(self, candidate_id: str) -> str | None:
        """Return the final recorded state for a candidate, if it has one."""
        if not isinstance(candidate_id, str) or not candidate_id:
            raise EvolutionError("candidate_id must be a non-empty string")
        state: str | None = None
        for event in read_jsonl(self._ledger_path):
            if event.get("candidate_id") == candidate_id and isinstance(event.get("event"), str):
                state = event["event"]
        return state

    def status(self) -> dict:
        """Summarize the current event-sourced state without mutating it."""
        evidence = read_jsonl(self._evidence_path)
        ledger = read_jsonl(self._ledger_path)
        candidates = sorted(
            {
                candidate_id
                for event in ledger
                if isinstance((candidate_id := event.get("candidate_id")), str)
            }
        )
        manifests_dir = self.root / "manifests"
        manifest_count = len(list(manifests_dir.glob("*.json"))) if manifests_dir.exists() else 0
        return {
            "evidence_count": len(evidence),
            "ledger_event_count": len(ledger),
            "manifest_count": manifest_count,
            "states": {candidate_id: self.latest_state(candidate_id) for candidate_id in candidates},
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd() / ".dominator" / "evolution",
        help="directory containing evolution runtime state",
    )
    parser.add_argument(
        "--skill-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="root of the packaged skill",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="save a run manifest JSON object")
    plan.add_argument("manifest", type=Path)
    record = commands.add_parser("record", help="record evaluated evidence JSON")
    record.add_argument("evidence", type=Path)
    commands.add_parser("status", help="print evidence and state counts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = EvolutionStore(args.root, args.skill_root)
    try:
        if args.command == "plan":
            result: object = {"manifest": str(store.save_manifest(read_json(args.manifest)))}
        elif args.command == "record":
            result = store.record_evidence(read_json(args.evidence))
        else:
            result = store.status()
    except EvolutionError as error:
        print(f"evolution error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
