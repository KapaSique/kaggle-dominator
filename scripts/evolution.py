#!/usr/bin/env python3
"""Append-only evidence registry for the kaggle-dominator evolution engine."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
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
ALLOWED_PROMOTION_PATH = "references/learned-playbook.md"
MAX_RUNTIME_RATIO = 2.0
MIN_CONFIRMATIONS = 2


@dataclass(frozen=True)
class GateResult:
    """The deterministic decision for one candidate promotion request."""

    passed: bool
    reasons: tuple[str, ...]
    improvement: float


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
    if evidence["noise_floor"] < 0:
        raise EvolutionError("evidence.noise_floor must be non-negative")
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

    @property
    def _promotions_path(self) -> Path:
        return self.root / "promotions.jsonl"

    @property
    def _learned_playbook_path(self) -> Path:
        return self.skill_root / ALLOWED_PROMOTION_PATH

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
        initial_events = tuple(recorded_events[: len(expected_events)])
        if initial_events != expected_events[: len(initial_events)]:
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

    def _evidence_for(self, candidate_id: str) -> dict | None:
        for evidence in read_jsonl(self._evidence_path):
            if evidence.get("candidate_id") == candidate_id:
                return evidence
        return None

    def _promotion_events(self) -> list[dict]:
        events = read_jsonl(self._promotions_path)
        self._validate_promotion_history(events)
        return events

    @staticmethod
    def _required_string(payload: dict, field: str, context: str) -> str:
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip():
            raise EvolutionError(f"{context}.{field} must be a non-empty string")
        return value

    def _validate_promotion_history(self, events: list[dict]) -> None:
        """Reject malformed immutable promotion history before using any event."""
        promotion_ids: dict[str, dict] = {}
        promoted_candidates: set[str] = set()
        rolled_back: set[str] = set()
        for index, event in enumerate(events, start=1):
            context = f"promotions[{index}]"
            if "state" in event:
                raise EvolutionError(f"{context}.state is not valid in promotion history")
            event_type = event.get("event")
            if event_type == "PROMOTED":
                promotion_id = self._required_string(event, "promotion_id", context)
                candidate_id = self._required_string(event, "candidate_id", context)
                occurred_at_utc = self._required_string(event, "occurred_at_utc", context)
                _normalize_utc(occurred_at_utc)
                if promotion_id in promotion_ids:
                    raise EvolutionError(f"duplicate promotion_id {promotion_id!r}")
                if candidate_id in promoted_candidates:
                    raise EvolutionError(f"duplicate promoted candidate {candidate_id!r}")
                evidence = event.get("evidence")
                if not isinstance(evidence, dict):
                    raise EvolutionError(f"{context}.evidence must be an object")
                immutable_evidence = {key: value for key, value in evidence.items() if key != "state"}
                validate_evidence(immutable_evidence)
                if evidence.get("state") != "EVALUATED":
                    raise EvolutionError(f"{context}.evidence.state must be EVALUATED")
                if evidence.get("candidate_id") != candidate_id:
                    raise EvolutionError(f"{context}.evidence candidate_id does not match promotion")
                if not _is_number(event.get("improvement")):
                    raise EvolutionError(f"{context}.improvement must be a number")
                promotion_ids[promotion_id] = event
                promoted_candidates.add(candidate_id)
            elif event_type == "ROLLED_BACK":
                promotion_id = self._required_string(event, "promotion_id", context)
                candidate_id = self._required_string(event, "candidate_id", context)
                occurred_at_utc = self._required_string(event, "occurred_at_utc", context)
                self._required_string(event, "reason", context)
                _normalize_utc(occurred_at_utc)
                promotion = promotion_ids.get(promotion_id)
                if promotion is None:
                    raise EvolutionError(f"{context} references unknown promotion_id {promotion_id!r}")
                if promotion["candidate_id"] != candidate_id:
                    raise EvolutionError(f"{context}.candidate_id does not match promotion")
                if promotion_id in rolled_back:
                    raise EvolutionError(f"duplicate rollback for promotion_id {promotion_id!r}")
                rolled_back.add(promotion_id)
            else:
                raise EvolutionError(f"{context}.event must be PROMOTED or ROLLED_BACK")

    @contextmanager
    def _promotion_lock(self) -> Any:
        """Serialize promotion/rollback mutation across store instances."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.root / ".promotion.lock", os.O_CREAT | os.O_RDWR, 0o600)
        except OSError as error:
            raise EvolutionError(f"cannot open promotion lock: {error}") from error
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        except OSError as error:
            raise EvolutionError(f"cannot lock promotion state: {error}") from error
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _existing_promotion(self, candidate_id: str) -> dict | None:
        for event in self._promotion_events():
            if event.get("event") == "PROMOTED" and event.get("candidate_id") == candidate_id:
                return event
        return None

    @staticmethod
    def _same_utc_date(left: str, right: str) -> bool:
        return _normalize_utc(left)[:10] == _normalize_utc(right)[:10]

    def _promotion_occurred_today(self, candidate_id: str, now_utc: str) -> bool:
        return any(
            event.get("event") == "PROMOTED"
            and event.get("candidate_id") != candidate_id
            and isinstance(event.get("occurred_at_utc"), str)
            and self._same_utc_date(event["occurred_at_utc"], now_utc)
            for event in self._promotion_events()
        )

    @staticmethod
    def _improvement(evidence: dict) -> float:
        candidate_score = Decimal(str(evidence["candidate_score"]))
        baseline_score = Decimal(str(evidence["baseline_score"]))
        if evidence["direction"] == "higher":
            return float(candidate_score - baseline_score)
        return float(baseline_score - candidate_score)

    @staticmethod
    def _improvement_exceeds_noise(evidence: dict) -> bool:
        candidate_score = Decimal(str(evidence["candidate_score"]))
        baseline_score = Decimal(str(evidence["baseline_score"]))
        improvement = (
            candidate_score - baseline_score
            if evidence["direction"] == "higher"
            else baseline_score - candidate_score
        )
        return improvement > Decimal(str(evidence["noise_floor"]))

    @staticmethod
    def _is_string_list(value: object) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) and item for item in value)

    def _gate_reasons(
        self,
        candidate_id: str,
        evidence: dict | None,
        verification: object,
        comparison: object,
        now_utc: str,
    ) -> tuple[list[str], float]:
        if evidence is None:
            return ["candidate_not_found"], 0.0

        try:
            validate_evidence({key: value for key, value in evidence.items() if key != "state"})
        except EvolutionError:
            return ["evidence_invalid"], 0.0
        if evidence.get("state") != "EVALUATED":
            return ["evidence_invalid"], 0.0

        reasons: list[str] = []
        improvement = self._improvement(evidence)
        if evidence["status"] != "succeeded":
            reasons.append("candidate_not_succeeded")
        if not evidence["metric_direction_verified"]:
            reasons.append("metric_direction_unverified")
        if improvement <= 0:
            reasons.append("improvement_not_positive")
        if not self._improvement_exceeds_noise(evidence):
            reasons.append("improvement_not_above_noise")
        if evidence["confirmations"] < MIN_CONFIRMATIONS:
            reasons.append("insufficient_confirmations")
        if not evidence["transferable"]:
            reasons.append("claim_not_transferable")
        if len(set(evidence["validation_regimes"])) < MIN_CONFIRMATIONS:
            reasons.append("insufficient_validation_regimes")
        if evidence["regressions"]:
            reasons.append("regressions_present")
        if evidence["forbidden_actions"]:
            reasons.append("forbidden_actions_present")
        if any(path != ALLOWED_PROMOTION_PATH for path in evidence["changed_paths"]):
            reasons.append("protected_path_changed")
        if evidence["runtime_ratio"] > MAX_RUNTIME_RATIO:
            reasons.append("runtime_ratio_exceeded")

        if not isinstance(verification, dict):
            reasons.append("verifier_invalid")
        else:
            if verification.get("candidate_id") != candidate_id:
                reasons.append("verifier_candidate_mismatch")
            if verification.get("verdict") != "PASS":
                reasons.append("verifier_not_pass")
            if verification.get("fresh_context") is not True:
                reasons.append("verifier_not_fresh")
            if not isinstance(verification.get("reviewer_id"), str) or not verification["reviewer_id"]:
                reasons.append("verifier_reviewer_missing")
            if not self._is_string_list(verification.get("checked_artifacts")) or not verification[
                "checked_artifacts"
            ]:
                reasons.append("verifier_checked_artifacts_missing")
            if not isinstance(verification.get("issues"), list):
                reasons.append("verifier_issues_invalid")
            elif verification["issues"]:
                reasons.append("verifier_issues_present")

        if not isinstance(comparison, dict):
            reasons.append("comparator_invalid")
        else:
            if comparison.get("candidate_id") != candidate_id:
                reasons.append("comparator_candidate_mismatch")
            if comparison.get("blind") is not True:
                reasons.append("comparator_not_blind")
            if comparison.get("winner") != "challenger":
                reasons.append("comparator_did_not_select_challenger")
            if not isinstance(comparison.get("rubric"), dict):
                reasons.append("comparator_rubric_missing")

        if self._existing_promotion(candidate_id) is not None:
            reasons.append("already_promoted")
        if self._promotion_occurred_today(candidate_id, now_utc):
            reasons.append("promotion_already_occurred_today")
        return reasons, improvement

    def _gate_candidate_at(
        self,
        candidate_id: str,
        verification: dict,
        comparison: dict,
        now_utc: str,
    ) -> GateResult:
        reasons, improvement = self._gate_reasons(
            candidate_id,
            self._evidence_for(candidate_id),
            verification,
            comparison,
            now_utc,
        )
        result = GateResult(not reasons, tuple(reasons), improvement)
        if self._evidence_for(candidate_id) is None:
            return result

        state = self.latest_state(candidate_id)
        if result.passed:
            if state != "VERIFIED":
                append_jsonl(
                    self._ledger_path,
                    {
                        "candidate_id": candidate_id,
                        "event": "VERIFIED",
                        "occurred_at_utc": now_utc,
                    },
                )
        elif state != "REJECTED" and self._existing_promotion(candidate_id) is None:
            append_jsonl(
                self._ledger_path,
                {
                    "candidate_id": candidate_id,
                    "event": "REJECTED",
                    "occurred_at_utc": now_utc,
                    "reasons": list(result.reasons),
                },
            )
        return result

    def gate_candidate(
        self, candidate_id: str, verification: dict, comparison: dict
    ) -> GateResult:
        """Fail closed unless all evidence, verifier, and comparison gates pass."""
        if not isinstance(candidate_id, str) or not candidate_id:
            raise EvolutionError("candidate_id must be a non-empty string")
        with self._promotion_lock():
            self._promotion_events()
            return self._gate_candidate_at(candidate_id, verification, comparison, _now_utc())

    def _verified_learned_playbook_destination(self) -> tuple[Path, Path]:
        """Return a regular destination within a non-symlinked skill subtree."""
        root = self.skill_root
        try:
            root.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise EvolutionError(f"cannot create skill root {root}: {error}") from error
        if root.is_symlink():
            raise EvolutionError("skill_root must not be a symlink")
        resolved_root = root.resolve(strict=True)
        parent = root
        for component in Path(ALLOWED_PROMOTION_PATH).parent.parts:
            parent = parent / component
            if parent.exists():
                if parent.is_symlink() or not parent.is_dir():
                    raise EvolutionError(f"learned playbook path component is unsafe: {parent}")
            else:
                try:
                    parent.mkdir()
                except OSError as error:
                    raise EvolutionError(f"cannot create learned playbook parent {parent}: {error}") from error
            resolved_parent = parent.resolve(strict=True)
            try:
                resolved_parent.relative_to(resolved_root)
            except ValueError as error:
                raise EvolutionError("learned playbook parent escapes skill_root") from error
        destination = parent / Path(ALLOWED_PROMOTION_PATH).name
        if destination.is_symlink():
            raise EvolutionError("learned playbook destination must not be a symlink")
        resolved_destination = destination.resolve(strict=False)
        try:
            resolved_destination.relative_to(resolved_root)
        except ValueError as error:
            raise EvolutionError("learned playbook destination escapes skill_root") from error
        return parent, destination

    def _render_learned_playbook(self) -> None:
        events = self._promotion_events()
        rolled_back = {
            event.get("promotion_id")
            for event in events
            if event.get("event") == "ROLLED_BACK" and isinstance(event.get("promotion_id"), str)
        }
        active = [
            event
            for event in events
            if event.get("event") == "PROMOTED"
            and event.get("promotion_id") not in rolled_back
        ]
        active.sort(key=lambda event: (event.get("occurred_at_utc", ""), event.get("candidate_id", "")))
        lines = [
            "# Learned playbook",
            "",
            "<!-- Generated by scripts/evolution.py from active promotion events. Do not edit. -->",
            "",
        ]
        for promotion in active:
            evidence = promotion["evidence"]
            lines.extend(
                [
                    f"## {promotion['candidate_id']}",
                    "",
                    f"- Promotion ID: `{promotion['promotion_id']}`",
                    f"- Promoted at (UTC): `{promotion['occurred_at_utc']}`",
                    f"- Claim: {evidence['claim']}",
                    f"- Scope limits: {evidence['scope_limits']}",
                    (
                        "- Metric: "
                        f"{evidence['metric']} ({evidence['direction']} is better); "
                        f"baseline {evidence['baseline_score']}, candidate {evidence['candidate_score']}."
                    ),
                    (
                        "- Evidence identifiers: "
                        f"code `{evidence['code_sha']}`, data `{evidence['data_fingerprint']}`, "
                        f"config `{evidence['config_hash']}`."
                    ),
                    "",
                ]
            )
        rendered = "\n".join(lines)
        parent, destination = self._verified_learned_playbook_destination()
        descriptor: int | None = None
        temporary: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}.", suffix=".tmp", dir=parent
            )
            temporary = Path(temporary_name)
            mode = os.fstat(descriptor).st_mode
            if not stat.S_ISREG(mode):
                raise EvolutionError("learned playbook temporary file is not regular")
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = None
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            if destination.is_symlink():
                raise EvolutionError("learned playbook destination became a symlink")
            temporary.replace(destination)
        except OSError as error:
            raise EvolutionError(f"cannot render learned playbook at {destination}: {error}") from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def promote(self, candidate_id: str, verification: dict, comparison: dict) -> dict:
        """Append a promotion only after a passing gate, then render active learnings."""
        if not isinstance(candidate_id, str) or not candidate_id:
            raise EvolutionError("candidate_id must be a non-empty string")
        with self._promotion_lock():
            self._promotion_events()
            existing = self._existing_promotion(candidate_id)
            if existing is not None:
                self._render_learned_playbook()
                return existing
            occurred_at_utc = _now_utc()
            gate = self._gate_candidate_at(candidate_id, verification, comparison, occurred_at_utc)
            if not gate.passed:
                return {
                    "candidate_id": candidate_id,
                    "event": "REJECTED",
                    "reasons": list(gate.reasons),
                    "improvement": gate.improvement,
                }
            evidence = self._evidence_for(candidate_id)
            if evidence is None:
                raise EvolutionError(f"candidate {candidate_id!r} is missing evidence")
            promotion = {
                "promotion_id": f"{candidate_id}@{occurred_at_utc}",
                "event": "PROMOTED",
                "candidate_id": candidate_id,
                "occurred_at_utc": occurred_at_utc,
                "improvement": gate.improvement,
                "evidence": evidence,
            }
            append_jsonl(self._promotions_path, promotion)
            append_jsonl(
                self._ledger_path,
                {
                    "candidate_id": candidate_id,
                    "event": "PROMOTED",
                    "occurred_at_utc": occurred_at_utc,
                    "promotion_id": promotion["promotion_id"],
                },
            )
            self._render_learned_playbook()
            return promotion

    def rollback(self, promotion_id: str, reason: str) -> dict:
        """Append a rollback event and regenerate the reference without deleting history."""
        if not isinstance(promotion_id, str) or not promotion_id:
            raise EvolutionError("promotion_id must be a non-empty string")
        if not isinstance(reason, str) or not reason.strip():
            raise EvolutionError("rollback reason must be a non-empty string")
        with self._promotion_lock():
            promotions = self._promotion_events()
            promotion = next(
                (
                    event
                    for event in promotions
                    if event.get("event") == "PROMOTED" and event.get("promotion_id") == promotion_id
                ),
                None,
            )
            if promotion is None:
                raise EvolutionError(f"unknown promotion_id {promotion_id!r}")
            existing = next(
                (
                    event
                    for event in promotions
                    if event.get("event") == "ROLLED_BACK" and event.get("promotion_id") == promotion_id
                ),
                None,
            )
            if existing is not None:
                self._render_learned_playbook()
                return existing
            rollback = {
                "event": "ROLLED_BACK",
                "promotion_id": promotion_id,
                "candidate_id": promotion["candidate_id"],
                "reason": reason.strip(),
                "occurred_at_utc": _now_utc(),
            }
            append_jsonl(self._promotions_path, rollback)
            append_jsonl(
                self._ledger_path,
                {
                    "candidate_id": promotion["candidate_id"],
                    "event": "ROLLED_BACK",
                    "promotion_id": promotion_id,
                    "occurred_at_utc": rollback["occurred_at_utc"],
                },
            )
            self._render_learned_playbook()
            return rollback

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
    for command_name, help_text in (
        ("gate", "evaluate promotion gates from verifier and comparator JSON"),
        ("promote", "promote a passing candidate from verifier and comparator JSON"),
    ):
        command = commands.add_parser(command_name, help=help_text)
        command.add_argument("candidate_id")
        command.add_argument("verification", type=Path)
        command.add_argument("comparison", type=Path)
    rollback = commands.add_parser("rollback", help="roll back a previous promotion")
    rollback.add_argument("promotion_id")
    rollback.add_argument("reason")
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
        elif args.command == "gate":
            result = store.gate_candidate(
                args.candidate_id,
                read_json(args.verification),
                read_json(args.comparison),
            )
        elif args.command == "promote":
            result = store.promote(
                args.candidate_id,
                read_json(args.verification),
                read_json(args.comparison),
            )
        elif args.command == "rollback":
            result = store.rollback(args.promotion_id, args.reason)
        else:
            result = store.status()
    except EvolutionError as error:
        print(f"evolution error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=lambda value: value.__dict__))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
