import contextlib
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "evolution.py"
SPEC = importlib.util.spec_from_file_location("evolution", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
evolution = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evolution
SPEC.loader.exec_module(evolution)

EvolutionError = evolution.EvolutionError
EvolutionStore = evolution.EvolutionStore
read_jsonl = evolution.read_jsonl


def valid_evidence() -> dict:
    return {
        "candidate_id": "cand-1",
        "parent_id": "incumbent-1",
        "competition": "example-competition",
        "competition_type": "tabular",
        "claim": "The challenger improves a repeatable validation score.",
        "scope_limits": "Only the measured dataset and feature family.",
        "metric": "auc",
        "direction": "higher",
        "metric_direction_verified": True,
        "baseline_score": 0.951,
        "candidate_score": 0.952,
        "noise_floor": 0.0002,
        "confirmations": 3,
        "validation_regimes": ["folds-v1", "seeds-11-22-33"],
        "code_sha": "git-sha",
        "data_fingerprint": "sha256-data",
        "config_hash": "sha256-config",
        "seeds": [11, 22, 33],
        "runtime_minutes": 120.0,
        "runtime_ratio": 1.3,
        "vram_gb": 16.0,
        "artifacts": ["artifacts/candidate/oof.parquet"],
        "regressions": [],
        "forbidden_actions": [],
        "changed_paths": ["references/learned-playbook.md"],
        "transferable": True,
        "status": "succeeded",
        "created_at_utc": "2026-07-17T00:00:00Z",
    }


def passing_verifier(candidate_id: str = "cand-1") -> dict:
    return {
        "candidate_id": candidate_id,
        "verdict": "PASS",
        "fresh_context": True,
        "reviewer_id": "verifier-1",
        "checked_artifacts": ["evidence.json", "oof.parquet"],
        "issues": [],
    }


def passing_comparator(candidate_id: str = "cand-1") -> dict:
    return {
        "candidate_id": candidate_id,
        "winner": "challenger",
        "blind": True,
        "rubric": {
            "score": 5,
            "stability": 5,
            "runtime": 4,
            "reproducibility": 5,
        },
    }


def write_json_artifact(root: Path, relative_path: str, payload: dict) -> tuple[str, str]:
    """Create a test artifact and return its relative path and content digest."""
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return relative_path, hashlib.sha256(destination.read_bytes()).hexdigest()


def provenance_manifest(
    root: Path,
    candidate_id: str = "cand-1",
    verifier: dict | None = None,
    comparator: dict | None = None,
    evidence: dict | None = None,
) -> dict:
    """Create a complete immutable provenance envelope for promotion tests."""
    verifier = verifier or passing_verifier(candidate_id)
    comparator = comparator or passing_comparator(candidate_id)
    evidence = evidence or valid_evidence() | {"candidate_id": candidate_id}
    run_id = f"run-{candidate_id}"
    evidence_path, evidence_hash = write_json_artifact(
        root, f"artifacts/{run_id}/raw-evidence.json", evidence
    )
    incumbent_path, incumbent_hash = write_json_artifact(
        root, f"artifacts/{run_id}/blind-incumbent.json", {"score": 0.951}
    )
    challenger_path, challenger_hash = write_json_artifact(
        root, f"artifacts/{run_id}/blind-challenger.json", {"score": 0.952}
    )
    proposer_path, proposer_hash = write_json_artifact(
        root, f"artifacts/{run_id}/proposer.json", evidence
    )
    verifier_path, verifier_hash = write_json_artifact(
        root, f"artifacts/{run_id}/verifier.json", verifier
    )
    comparator_path, comparator_hash = write_json_artifact(
        root, f"artifacts/{run_id}/comparator.json", comparator
    )
    comparator_inputs = [
        {"label": "incumbent", "path": incumbent_path, "sha256": incumbent_hash},
        {"label": "challenger", "path": challenger_path, "sha256": challenger_hash},
    ]
    return {
        "run_id": run_id,
        "candidate_id": candidate_id,
        "created_at_utc": "2026-07-17T00:00:00Z",
        "artifacts": [
            {
                "role": "proposer",
                "worker_id": "proposer-1",
                "output_path": proposer_path,
                "sha256": proposer_hash,
                "created_at_utc": "2026-07-17T00:00:00Z",
                "terminal_status": evidence["status"],
                "input_artifacts": [{"path": evidence_path, "sha256": evidence_hash}],
            },
            {
                "role": "verifier",
                "worker_id": "verifier-1",
                "output_path": verifier_path,
                "sha256": verifier_hash,
                "created_at_utc": "2026-07-17T00:01:00Z",
                "terminal_status": "PASS",
                "input_artifacts": [{"path": evidence_path, "sha256": evidence_hash}],
            },
            {
                "role": "comparator",
                "worker_id": "comparator-1",
                "output_path": comparator_path,
                "sha256": comparator_hash,
                "created_at_utc": "2026-07-17T00:02:00Z",
                "terminal_status": "challenger",
                "input_artifacts": comparator_inputs,
            },
        ],
        "comparator_package": {"candidate_token": candidate_id, "inputs": comparator_inputs},
    }


class EvolutionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.state_dir = Path(self.temp_dir.name) / "state"
        self.skill_root = Path(self.temp_dir.name) / "skill"
        self.store = EvolutionStore(self.state_dir, self.skill_root)

    def test_record_evidence_creates_evaluated_state_and_append_only_events(self) -> None:
        record = self.store.record_evidence(valid_evidence())

        self.assertEqual(record["state"], "EVALUATED")
        self.assertEqual(self.store.latest_state("cand-1"), "EVALUATED")
        self.assertEqual(len(read_jsonl(self.state_dir / "evidence.jsonl")), 1)
        self.assertEqual(
            [event["event"] for event in read_jsonl(self.state_dir / "ledger.jsonl")],
            ["OBSERVED", "EVALUATED"],
        )

    def test_duplicate_candidate_is_a_no_op(self) -> None:
        first = self.store.record_evidence(valid_evidence())
        duplicate = valid_evidence()
        duplicate["claim"] = "A conflicting later claim must not replace history."

        second = self.store.record_evidence(duplicate)

        self.assertEqual(second, first)
        self.assertEqual(len(read_jsonl(self.state_dir / "evidence.jsonl")), 1)
        self.assertEqual(len(read_jsonl(self.state_dir / "ledger.jsonl")), 2)

    def test_malformed_duplicate_is_rejected_without_mutating_ledgers(self) -> None:
        self.store.record_evidence(valid_evidence())
        evidence_path = self.state_dir / "evidence.jsonl"
        ledger_path = self.state_dir / "ledger.jsonl"
        evidence_before = evidence_path.read_bytes()
        ledger_before = ledger_path.read_bytes()
        wrong_type = valid_evidence()
        wrong_type["direction"] = {"unexpected": "higher"}

        for malformed in ({"candidate_id": "cand-1"}, wrong_type):
            with self.subTest(malformed=malformed):
                with self.assertRaises(EvolutionError):
                    self.store.record_evidence(malformed)
                self.assertEqual(evidence_path.read_bytes(), evidence_before)
                self.assertEqual(ledger_path.read_bytes(), ledger_before)

    def test_list_direction_raises_evolution_error(self) -> None:
        evidence = valid_evidence()
        evidence["direction"] = ["higher"]

        with self.assertRaises(EvolutionError):
            evolution.validate_evidence(evidence)

    def test_retry_after_first_ledger_failure_repairs_both_events(self) -> None:
        original_append = evolution.append_jsonl
        ledger_attempts = 0

        def fail_first_ledger_append(path: Path, payload: dict) -> None:
            nonlocal ledger_attempts
            if path == self.state_dir / "ledger.jsonl":
                ledger_attempts += 1
                if ledger_attempts == 1:
                    raise EvolutionError("injected first ledger append failure")
            original_append(path, payload)

        with patch.object(evolution, "append_jsonl", side_effect=fail_first_ledger_append):
            with self.assertRaises(EvolutionError):
                self.store.record_evidence(valid_evidence())

        self.assertEqual(len(read_jsonl(self.state_dir / "evidence.jsonl")), 1)
        self.assertEqual(read_jsonl(self.state_dir / "ledger.jsonl"), [])

        self.store.record_evidence(valid_evidence())

        self.assertEqual(len(read_jsonl(self.state_dir / "evidence.jsonl")), 1)
        self.assertEqual(
            [event["event"] for event in read_jsonl(self.state_dir / "ledger.jsonl")],
            ["OBSERVED", "EVALUATED"],
        )
        self.assertEqual(self.store.latest_state("cand-1"), "EVALUATED")

    def test_retry_after_second_ledger_failure_repairs_only_evaluated_event(self) -> None:
        original_append = evolution.append_jsonl
        ledger_attempts = 0

        def fail_second_ledger_append(path: Path, payload: dict) -> None:
            nonlocal ledger_attempts
            if path == self.state_dir / "ledger.jsonl":
                ledger_attempts += 1
                if ledger_attempts == 2:
                    raise EvolutionError("injected second ledger append failure")
            original_append(path, payload)

        with patch.object(evolution, "append_jsonl", side_effect=fail_second_ledger_append):
            with self.assertRaises(EvolutionError):
                self.store.record_evidence(valid_evidence())

        self.assertEqual(len(read_jsonl(self.state_dir / "evidence.jsonl")), 1)
        self.assertEqual(
            [event["event"] for event in read_jsonl(self.state_dir / "ledger.jsonl")],
            ["OBSERVED"],
        )

        self.store.record_evidence(valid_evidence())

        self.assertEqual(len(read_jsonl(self.state_dir / "evidence.jsonl")), 1)
        self.assertEqual(
            [event["event"] for event in read_jsonl(self.state_dir / "ledger.jsonl")],
            ["OBSERVED", "EVALUATED"],
        )
        self.assertEqual(self.store.latest_state("cand-1"), "EVALUATED")

    def test_save_manifest_copies_normalized_manifest_under_run_id(self) -> None:
        manifest = {
            "run_id": "run-2026-07-17",
            "created_at_utc": "2026-07-17T03:00:00+03:00",
            "focus": "example-competition",
        }

        saved = self.store.save_manifest(manifest)

        self.assertEqual(saved, self.state_dir / "manifests" / "run-2026-07-17.json")
        self.assertEqual(read_jsonl(self.state_dir / "evidence.jsonl"), [])
        self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["created_at_utc"], "2026-07-17T00:00:00Z")

    def test_record_normalizes_evidence_timestamp_to_utc(self) -> None:
        evidence = valid_evidence()
        evidence["created_at_utc"] = "2026-07-17T03:00:00+03:00"

        record = self.store.record_evidence(evidence)

        self.assertEqual(record["created_at_utc"], "2026-07-17T00:00:00Z")

    def test_malformed_json_and_invalid_evidence_raise_evolution_error(self) -> None:
        malformed = self.state_dir / "malformed.json"
        malformed.parent.mkdir(parents=True)
        malformed.write_text("{not-json", encoding="utf-8")
        invalid = valid_evidence()
        del invalid["candidate_id"]

        with self.assertRaises(EvolutionError):
            evolution.read_json(malformed)
        with self.assertRaises(EvolutionError):
            self.store.record_evidence(invalid)

    def test_status_reports_current_evaluated_candidate(self) -> None:
        self.store.record_evidence(valid_evidence())

        status = self.store.status()

        self.assertEqual(status["states"], {"cand-1": "EVALUATED"})
        self.assertEqual(status["evidence_count"], 1)


class EvolutionProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.state_dir = root / "state"
        self.store = EvolutionStore(self.state_dir, root / "skill")

    def test_valid_separated_registered_artifacts_pass_gate(self) -> None:
        self.store.record_evidence(valid_evidence())
        manifest = provenance_manifest(self.state_dir)
        self.store.save_manifest(manifest)

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertTrue(result.passed)

    def test_bare_fresh_and_blind_booleans_cannot_bypass_provenance(self) -> None:
        self.store.record_evidence(valid_evidence())

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertFalse(result.passed)
        self.assertIn("provenance_manifest_missing", result.reasons)

    def test_manifest_rejects_same_worker_for_proposer_and_verifier(self) -> None:
        manifest = provenance_manifest(self.state_dir)
        manifest["artifacts"][1]["worker_id"] = "proposer-1"

        with self.assertRaisesRegex(EvolutionError, "worker"):
            self.store.save_manifest(manifest)

    def test_manifest_rejects_shared_output_path(self) -> None:
        manifest = provenance_manifest(self.state_dir)
        manifest["artifacts"][2]["output_path"] = manifest["artifacts"][0]["output_path"]

        with self.assertRaisesRegex(EvolutionError, "output_path"):
            self.store.save_manifest(manifest)

    def test_manifest_rejects_verifier_input_from_proposer_output(self) -> None:
        manifest = provenance_manifest(self.state_dir)
        proposer = manifest["artifacts"][0]
        manifest["artifacts"][1]["input_artifacts"] = [
            {"path": proposer["output_path"], "sha256": proposer["sha256"]}
        ]

        with self.assertRaisesRegex(EvolutionError, "verifier.*proposer"):
            self.store.save_manifest(manifest)

    def test_manifest_rejects_missing_hash_or_utc_timestamp(self) -> None:
        for field, value in (("sha256", ""), ("created_at_utc", "not-a-timestamp")):
            with self.subTest(field=field):
                manifest = provenance_manifest(self.state_dir)
                manifest["artifacts"][1][field] = value
                with self.assertRaises(EvolutionError):
                    self.store.save_manifest(manifest)

    def test_gate_rejects_output_that_no_longer_matches_registered_hash(self) -> None:
        self.store.record_evidence(valid_evidence())
        manifest = provenance_manifest(self.state_dir)
        self.store.save_manifest(manifest)
        verifier_path = self.state_dir / manifest["artifacts"][1]["output_path"]
        verifier_path.write_text(json.dumps({**passing_verifier(), "fresh_context": False}), encoding="utf-8")

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertFalse(result.passed)
        self.assertIn("verifier_provenance_invalid", result.reasons)

    def test_gate_rejects_registered_proposer_for_a_different_evidence_object(self) -> None:
        self.store.record_evidence(valid_evidence())
        manifest = provenance_manifest(self.state_dir)
        substituted = valid_evidence()
        substituted["claim"] = "A different measured claim."
        _, proposer_hash = write_json_artifact(
            self.state_dir, manifest["artifacts"][0]["output_path"], substituted
        )
        manifest["artifacts"][0]["sha256"] = proposer_hash
        self.store.save_manifest(manifest)

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertFalse(result.passed)
        self.assertIn("proposer_provenance_mismatch", result.reasons)

    def test_manifest_rejects_identity_leakage_or_a_b_comparator_package(self) -> None:
        for package in (
            {
                "candidate_token": "cand-1",
                "inputs": [
                    {"label": "A", "path": "artifacts/a.json", "sha256": "a" * 64},
                    {"label": "B", "path": "artifacts/b.json", "sha256": "b" * 64},
                ],
            },
            {
                "candidate_token": "cand-1",
                "model_author": "leaked",
                "inputs": [],
            },
        ):
            with self.subTest(package=package):
                manifest = provenance_manifest(self.state_dir)
                manifest["comparator_package"] = package
                with self.assertRaises(EvolutionError):
                    self.store.save_manifest(manifest)


class EvolutionPromotionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        self.state_dir = root / "state"
        self.skill_root = root / "skill"
        self.store = EvolutionStore(self.state_dir, self.skill_root)

    @property
    def learned_playbook(self) -> Path:
        return self.skill_root / "references" / "learned-playbook.md"

    def record(self, candidate_id: str = "cand-1", **changes: object) -> dict:
        evidence = valid_evidence()
        evidence["candidate_id"] = candidate_id
        evidence.update(changes)
        record = self.store.record_evidence(evidence)
        self.store.save_manifest(
            provenance_manifest(self.state_dir, candidate_id, evidence=evidence)
        )
        return record

    def gate(self, candidate_id: str = "cand-1", **changes: object) -> object:
        self.record(candidate_id, **changes)
        return self.store.gate_candidate(
            candidate_id, passing_verifier(candidate_id), passing_comparator(candidate_id)
        )

    def promotion_event(
        self,
        candidate_id: str = "cand-old",
        promotion_id: str = "cand-old@2026-07-17T00:00:00Z",
        occurred_at_utc: str = "2026-07-17T00:00:00Z",
    ) -> dict:
        evidence = valid_evidence()
        evidence.update({"candidate_id": candidate_id, "state": "EVALUATED"})
        return {
            "event": "PROMOTED",
            "promotion_id": promotion_id,
            "candidate_id": candidate_id,
            "occurred_at_utc": occurred_at_utc,
            "improvement": 0.001,
            "evidence": evidence,
        }

    def test_passing_gate_promotes_and_rollback_removes_generated_claim(self) -> None:
        self.record()

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())
        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, ())
        self.assertGreater(result.improvement, 0)

        promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())
        self.assertEqual(self.store.latest_state("cand-1"), "PROMOTED")
        self.assertIn("cand-1", self.learned_playbook.read_text(encoding="utf-8"))
        self.assertIn(valid_evidence()["claim"], self.learned_playbook.read_text(encoding="utf-8"))

        rollback = self.store.rollback(promotion["promotion_id"], "regression discovered")
        self.assertEqual(rollback["event"], "ROLLED_BACK")
        self.assertEqual(self.store.latest_state("cand-1"), "ROLLED_BACK")
        self.assertNotIn(valid_evidence()["claim"], self.learned_playbook.read_text(encoding="utf-8"))

    def test_gate_rejects_unverified_metric_direction(self) -> None:
        result = self.gate(metric_direction_verified=False)

        self.assertFalse(result.passed)
        self.assertIn("metric_direction_unverified", result.reasons)

    def test_gate_uses_lower_is_better_direction_for_improvement(self) -> None:
        result = self.gate(direction="lower", candidate_score=0.9506)

        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.improvement, 0.0004)

    def test_gate_rejects_delta_at_noise_floor(self) -> None:
        result = self.gate(candidate_score=0.9512)

        self.assertFalse(result.passed)
        self.assertIn("improvement_not_above_noise", result.reasons)

    def test_gate_rejects_fewer_than_two_confirmations(self) -> None:
        result = self.gate(confirmations=1)

        self.assertFalse(result.passed)
        self.assertIn("insufficient_confirmations", result.reasons)

    def test_gate_rejects_transferable_claim_with_one_regime(self) -> None:
        result = self.gate(validation_regimes=["folds-v1"])

        self.assertFalse(result.passed)
        self.assertIn("insufficient_validation_regimes", result.reasons)

    def test_gate_rejects_non_transferable_claim(self) -> None:
        result = self.gate(transferable=False)

        self.assertFalse(result.passed)
        self.assertIn("claim_not_transferable", result.reasons)

    def test_gate_rejects_non_succeeded_candidate(self) -> None:
        result = self.gate(status="failed")

        self.assertFalse(result.passed)
        self.assertIn("candidate_not_succeeded", result.reasons)

    def test_gate_rejects_regressions(self) -> None:
        result = self.gate(regressions=["fairness regression"])

        self.assertFalse(result.passed)
        self.assertIn("regressions_present", result.reasons)

    def test_gate_rejects_forbidden_actions(self) -> None:
        result = self.gate(forbidden_actions=["submitted to Kaggle"])

        self.assertFalse(result.passed)
        self.assertIn("forbidden_actions_present", result.reasons)

    def test_gate_rejects_protected_path_changes(self) -> None:
        result = self.gate(changed_paths=["SKILL.md"])

        self.assertFalse(result.passed)
        self.assertIn("protected_path_changed", result.reasons)

    def test_gate_rejects_runtime_ratio_above_two(self) -> None:
        result = self.gate(runtime_ratio=2.01)

        self.assertFalse(result.passed)
        self.assertIn("runtime_ratio_exceeded", result.reasons)

    def test_gate_rejects_stale_or_non_fresh_verifier(self) -> None:
        self.record()
        verifier = passing_verifier()
        verifier["fresh_context"] = False
        verifier["verdict"] = "STALE"

        result = self.store.gate_candidate("cand-1", verifier, passing_comparator())

        self.assertFalse(result.passed)
        self.assertIn("verifier_not_fresh", result.reasons)
        self.assertIn("verifier_not_pass", result.reasons)

    def test_gate_rejects_verifier_issues(self) -> None:
        self.record()
        verifier = passing_verifier()
        verifier["issues"] = ["artifact mismatch"]

        result = self.store.gate_candidate("cand-1", verifier, passing_comparator())

        self.assertFalse(result.passed)
        self.assertIn("verifier_issues_present", result.reasons)

    def test_gate_rejects_verifier_without_checked_artifacts(self) -> None:
        self.record()
        verifier = passing_verifier()
        verifier["checked_artifacts"] = []

        result = self.store.gate_candidate("cand-1", verifier, passing_comparator())

        self.assertFalse(result.passed)
        self.assertIn("verifier_checked_artifacts_missing", result.reasons)

    def test_gate_rejects_non_blind_comparator(self) -> None:
        self.record()
        comparison = passing_comparator()
        comparison["blind"] = False

        result = self.store.gate_candidate("cand-1", passing_verifier(), comparison)

        self.assertFalse(result.passed)
        self.assertIn("comparator_not_blind", result.reasons)

    def test_gate_rejects_incumbent_winner(self) -> None:
        self.record()
        comparison = passing_comparator()
        comparison["winner"] = "incumbent"

        result = self.store.gate_candidate("cand-1", passing_verifier(), comparison)

        self.assertFalse(result.passed)
        self.assertIn("comparator_did_not_select_challenger", result.reasons)

    def test_duplicate_promotion_is_idempotent(self) -> None:
        self.record()

        first = self.store.promote("cand-1", passing_verifier(), passing_comparator())
        second = self.store.promote("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual(second, first)
        events = read_jsonl(self.state_dir / "promotions.jsonl")
        self.assertEqual([event["event"] for event in events], ["PROMOTED"])

    def test_duplicate_gate_preserves_promoted_terminal_state(self) -> None:
        self.record()
        self.store.promote("cand-1", passing_verifier(), passing_comparator())

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertFalse(result.passed)
        self.assertIn("already_promoted", result.reasons)
        self.assertEqual(self.store.latest_state("cand-1"), "PROMOTED")

    def test_gate_rejects_second_promotion_on_same_utc_day(self) -> None:
        self.record()
        self.store.promote("cand-1", passing_verifier(), passing_comparator())
        self.record("cand-2")

        result = self.store.gate_candidate(
            "cand-2", passing_verifier("cand-2"), passing_comparator("cand-2")
        )

        self.assertFalse(result.passed)
        self.assertIn("promotion_already_occurred_today", result.reasons)

    def test_rejected_gate_appends_event_without_changing_generated_reference(self) -> None:
        self.record(regressions=["bad regression"])
        before = self.learned_playbook.read_bytes() if self.learned_playbook.exists() else None

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertFalse(result.passed)
        self.assertEqual(self.store.latest_state("cand-1"), "REJECTED")
        after = self.learned_playbook.read_bytes() if self.learned_playbook.exists() else None
        self.assertEqual(after, before)

    def test_gate_fails_closed_for_corrupt_stored_evidence(self) -> None:
        append = {"candidate_id": "cand-1", "state": "EVALUATED"}
        self.state_dir.mkdir(parents=True)
        (self.state_dir / "evidence.jsonl").write_text(json.dumps(append) + "\n", encoding="utf-8")

        result = self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertFalse(result.passed)
        self.assertEqual(result.reasons, ("evidence_invalid",))

    def test_validate_evidence_rejects_negative_noise_floor(self) -> None:
        evidence = valid_evidence()
        evidence["noise_floor"] = -0.01

        with self.assertRaisesRegex(EvolutionError, "noise_floor"):
            self.store.record_evidence(evidence)

    def test_gate_requires_positive_improvement_even_with_negative_noise_floor(self) -> None:
        evidence = valid_evidence()
        evidence.update({"candidate_score": 0.950, "noise_floor": -0.01, "state": "EVALUATED"})

        with patch.object(evolution, "validate_evidence"):
            reasons, _ = self.store._gate_reasons(
                "cand-1", evidence, passing_verifier(), passing_comparator(), "2026-07-17T00:00:00Z"
            )

        self.assertIn("improvement_not_positive", reasons)

    def test_render_rejects_symlinked_playbook_without_touching_victim(self) -> None:
        victim = Path(self.temp_dir.name) / "victim.md"
        victim.write_bytes(b"do not overwrite")
        self.learned_playbook.parent.mkdir(parents=True)
        self.learned_playbook.symlink_to(victim)
        self.record()

        with self.assertRaises(EvolutionError):
            self.store.promote("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual(victim.read_bytes(), b"do not overwrite")

    def test_render_rejects_symlinked_reference_directory_without_touching_victim(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        outside.mkdir()
        victim = outside / "learned-playbook.md"
        victim.write_bytes(b"do not overwrite")
        self.skill_root.mkdir()
        (self.skill_root / "references").symlink_to(outside, target_is_directory=True)
        self.record()

        with self.assertRaises(EvolutionError):
            self.store.promote("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual(victim.read_bytes(), b"do not overwrite")

    def test_promote_retry_reconciles_playbook_after_render_failure(self) -> None:
        self.record()
        with patch.object(self.store, "_render_learned_playbook", side_effect=EvolutionError("injected")):
            with self.assertRaisesRegex(EvolutionError, "injected"):
                self.store.promote("cand-1", passing_verifier(), passing_comparator())

        promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual([event["event"] for event in read_jsonl(self.state_dir / "promotions.jsonl")], ["PROMOTED"])
        self.assertIn(promotion["evidence"]["claim"], self.learned_playbook.read_text(encoding="utf-8"))

    def test_rollback_retry_reconciles_playbook_after_render_failure(self) -> None:
        self.record()
        promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())
        with patch.object(self.store, "_render_learned_playbook", side_effect=EvolutionError("injected")):
            with self.assertRaisesRegex(EvolutionError, "injected"):
                self.store.rollback(promotion["promotion_id"], "regression")

        rollback = self.store.rollback(promotion["promotion_id"], "regression")

        self.assertEqual(rollback["event"], "ROLLED_BACK")
        self.assertEqual(
            [event["event"] for event in read_jsonl(self.state_dir / "promotions.jsonl")],
            ["PROMOTED", "ROLLED_BACK"],
        )
        self.assertNotIn(valid_evidence()["claim"], self.learned_playbook.read_text(encoding="utf-8"))

    def test_promote_retry_recovers_missing_terminal_ledger_event(self) -> None:
        self.record()
        original_append = evolution.append_jsonl

        def fail_promoted_ledger_append(path: Path, payload: dict) -> None:
            if path == self.state_dir / "ledger.jsonl" and payload.get("event") == "PROMOTED":
                raise EvolutionError("injected promoted ledger failure")
            original_append(path, payload)

        with patch.object(evolution, "append_jsonl", side_effect=fail_promoted_ledger_append):
            with self.assertRaisesRegex(EvolutionError, "injected promoted ledger failure"):
                self.store.promote("cand-1", passing_verifier(), passing_comparator())

        promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())
        terminal_events = [
            event for event in read_jsonl(self.state_dir / "ledger.jsonl") if event["event"] == "PROMOTED"
        ]

        self.assertEqual(len(read_jsonl(self.state_dir / "promotions.jsonl")), 1)
        self.assertEqual(len(terminal_events), 1)
        self.assertEqual(self.store.latest_state("cand-1"), "PROMOTED")
        self.assertIn(promotion["evidence"]["claim"], self.learned_playbook.read_text(encoding="utf-8"))

    def test_new_rollback_recovers_missing_promoted_terminal_before_rollback(self) -> None:
        self.record()
        original_append = evolution.append_jsonl

        def fail_promoted_ledger_append(path: Path, payload: dict) -> None:
            if path == self.state_dir / "ledger.jsonl" and payload.get("event") == "PROMOTED":
                raise EvolutionError("injected promoted ledger failure")
            original_append(path, payload)

        with patch.object(evolution, "append_jsonl", side_effect=fail_promoted_ledger_append):
            with self.assertRaisesRegex(EvolutionError, "injected promoted ledger failure"):
                self.store.promote("cand-1", passing_verifier(), passing_comparator())

        promotion = read_jsonl(self.state_dir / "promotions.jsonl")[0]
        rollback = self.store.rollback(promotion["promotion_id"], "regression")
        retry = self.store.rollback(promotion["promotion_id"], "regression")
        terminal_events = [
            event
            for event in read_jsonl(self.state_dir / "ledger.jsonl")
            if event["event"] in {"PROMOTED", "ROLLED_BACK"}
        ]

        self.assertEqual([event["event"] for event in terminal_events], ["PROMOTED", "ROLLED_BACK"])
        self.assertEqual(rollback, retry)
        self.assertEqual(self.store.latest_state("cand-1"), "ROLLED_BACK")
        self.assertNotIn(valid_evidence()["claim"], self.learned_playbook.read_text(encoding="utf-8"))

    def test_rollback_retry_recovers_missing_terminal_ledger_event(self) -> None:
        self.record()
        promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())
        original_append = evolution.append_jsonl

        def fail_rolled_back_ledger_append(path: Path, payload: dict) -> None:
            if path == self.state_dir / "ledger.jsonl" and payload.get("event") == "ROLLED_BACK":
                raise EvolutionError("injected rollback ledger failure")
            original_append(path, payload)

        with patch.object(evolution, "append_jsonl", side_effect=fail_rolled_back_ledger_append):
            with self.assertRaisesRegex(EvolutionError, "injected rollback ledger failure"):
                self.store.rollback(promotion["promotion_id"], "regression")

        rollback = self.store.rollback(promotion["promotion_id"], "regression")
        terminal_events = [
            event for event in read_jsonl(self.state_dir / "ledger.jsonl") if event["event"] == "ROLLED_BACK"
        ]

        self.assertEqual([event["event"] for event in read_jsonl(self.state_dir / "promotions.jsonl")], ["PROMOTED", "ROLLED_BACK"])
        self.assertEqual(len(terminal_events), 1)
        self.assertEqual(rollback["event"], "ROLLED_BACK")
        self.assertEqual(self.store.latest_state("cand-1"), "ROLLED_BACK")
        self.assertNotIn(valid_evidence()["claim"], self.learned_playbook.read_text(encoding="utf-8"))

    def test_terminal_ledger_recovery_rejects_out_of_order_history(self) -> None:
        self.record()
        promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())
        evolution.append_jsonl(
            self.state_dir / "ledger.jsonl",
            {
                "candidate_id": "cand-1",
                "event": "ROLLED_BACK",
                "promotion_id": promotion["promotion_id"],
                "occurred_at_utc": promotion["occurred_at_utc"],
            },
        )
        ledger_before = (self.state_dir / "ledger.jsonl").read_bytes()

        with self.assertRaises(EvolutionError):
            self.store.promote("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual((self.state_dir / "ledger.jsonl").read_bytes(), ledger_before)

    def test_terminal_ledger_recovery_rejects_nonterminal_event_after_promotion(self) -> None:
        self.record()
        promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())
        evolution.append_jsonl(
            self.state_dir / "ledger.jsonl",
            {
                "candidate_id": "cand-1",
                "event": "VERIFIED",
                "occurred_at_utc": promotion["occurred_at_utc"],
            },
        )
        ledger_before = (self.state_dir / "ledger.jsonl").read_bytes()

        with self.assertRaises(EvolutionError):
            self.store.promote("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual((self.state_dir / "ledger.jsonl").read_bytes(), ledger_before)

    def test_history_rejects_two_promotions_on_one_normalized_utc_date(self) -> None:
        first = self.promotion_event(
            candidate_id="cand-first",
            promotion_id="first@2026-07-17T00:00:00Z",
            occurred_at_utc="2026-07-17T03:00:00+03:00",
        )
        second = self.promotion_event(
            candidate_id="cand-second",
            promotion_id="second@2026-07-17T22:00:00Z",
            occurred_at_utc="2026-07-17T22:00:00Z",
        )
        path = self.state_dir / "promotions.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
        promotions_before = path.read_bytes()

        with self.assertRaises(EvolutionError):
            self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual(path.read_bytes(), promotions_before)

    def test_history_rejects_nonfinite_or_boolean_improvement(self) -> None:
        for improvement in (float("nan"), float("inf"), True):
            with self.subTest(improvement=improvement):
                event = self.promotion_event()
                event["improvement"] = improvement
                path = self.state_dir / "promotions.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(event) + "\n", encoding="utf-8")

                with self.assertRaises(EvolutionError):
                    self.store._render_learned_playbook()

    def test_nonfinite_promotion_history_returns_cli_error_without_writes(self) -> None:
        self.record()
        event = self.promotion_event()
        event["improvement"] = float("nan")
        promotions_path = self.state_dir / "promotions.jsonl"
        promotions_path.parent.mkdir(parents=True, exist_ok=True)
        promotions_path.write_text(json.dumps(event) + "\n", encoding="utf-8")
        promotions_before = promotions_path.read_bytes()
        ledger_before = (self.state_dir / "ledger.jsonl").read_bytes()
        verifier = Path(self.temp_dir.name) / "verifier.json"
        comparator = Path(self.temp_dir.name) / "comparator.json"
        verifier.write_text(json.dumps(passing_verifier()), encoding="utf-8")
        comparator.write_text(json.dumps(passing_comparator()), encoding="utf-8")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = evolution.main(
                [
                    "--root",
                    str(self.state_dir),
                    "--skill-root",
                    str(self.skill_root),
                    "gate",
                    "cand-1",
                    str(verifier),
                    str(comparator),
                ]
            )

        self.assertEqual(result, 2)
        self.assertIn("evolution error:", stderr.getvalue())
        self.assertEqual(promotions_path.read_bytes(), promotions_before)
        self.assertEqual((self.state_dir / "ledger.jsonl").read_bytes(), ledger_before)

    def test_promotion_uses_one_timestamp_across_utc_boundary(self) -> None:
        evolution.append_jsonl(
            self.state_dir / "promotions.jsonl",
            self.promotion_event(
                candidate_id="cand-old",
                promotion_id="old@2026-07-18T00:00:00Z",
                occurred_at_utc="2026-07-18T00:00:00Z",
            ),
        )
        self.record("cand-1")

        with patch.object(
            evolution,
            "_now_utc",
            side_effect=["2026-07-17T23:59:59Z", "2026-07-18T00:00:00Z"],
        ) as now:
            promotion = self.store.promote("cand-1", passing_verifier(), passing_comparator())

        self.assertEqual(promotion["occurred_at_utc"], "2026-07-17T23:59:59Z")
        self.assertEqual(now.call_count, 1)

    def test_concurrent_candidates_allow_at_most_one_same_day_promotion(self) -> None:
        first = EvolutionStore(self.state_dir, self.skill_root)
        second = EvolutionStore(self.state_dir, self.skill_root)
        first.record_evidence(valid_evidence())
        first.save_manifest(provenance_manifest(self.state_dir))
        second_evidence = valid_evidence()
        second_evidence["candidate_id"] = "cand-2"
        second.record_evidence(second_evidence)
        second.save_manifest(provenance_manifest(self.state_dir, "cand-2", evidence=second_evidence))
        barrier = threading.Barrier(2)
        original = EvolutionStore._promotion_occurred_today

        def synchronized(store: EvolutionStore, candidate_id: str, now_utc: str) -> bool:
            try:
                barrier.wait(timeout=0.2)
            except threading.BrokenBarrierError:
                pass
            return original(store, candidate_id, now_utc)

        with patch.object(EvolutionStore, "_promotion_occurred_today", new=synchronized):
            with ThreadPoolExecutor(max_workers=2) as workers:
                results = list(
                    workers.map(
                        lambda args: args[0].promote(
                            args[1], passing_verifier(args[1]), passing_comparator(args[1])
                        ),
                        ((first, "cand-1"), (second, "cand-2")),
                    )
                )

        self.assertEqual(sum(result["event"] == "PROMOTED" for result in results), 1)
        self.assertEqual(
            len([event for event in read_jsonl(self.state_dir / "promotions.jsonl") if event["event"] == "PROMOTED"]),
            1,
        )

    def test_corrupt_promotion_history_blocks_gate_and_rollback_without_writes(self) -> None:
        self.record()
        corrupt = {"event": "PROMOTED", "promotion_id": "bad", "candidate_id": "cand-1"}
        evolution.append_jsonl(self.state_dir / "promotions.jsonl", corrupt)
        ledger_before = (self.state_dir / "ledger.jsonl").read_bytes()
        promotions_before = (self.state_dir / "promotions.jsonl").read_bytes()

        with self.assertRaises(EvolutionError):
            self.store.gate_candidate("cand-1", passing_verifier(), passing_comparator())
        with self.assertRaises(EvolutionError):
            self.store.rollback("bad", "reason")

        self.assertEqual((self.state_dir / "ledger.jsonl").read_bytes(), ledger_before)
        self.assertEqual((self.state_dir / "promotions.jsonl").read_bytes(), promotions_before)

    def test_corrupt_promotion_history_rejects_duplicate_candidate_and_broken_rollback(self) -> None:
        first = self.promotion_event()
        duplicate = self.promotion_event(candidate_id="cand-old", promotion_id="other@2026-07-17T01:00:00Z")
        broken = {
            "event": "ROLLED_BACK",
            "promotion_id": "missing@2026-07-17T02:00:00Z",
            "candidate_id": "missing",
            "reason": "bad",
            "occurred_at_utc": "2026-07-17T02:00:00Z",
        }
        for events in ((first, duplicate), (broken,)):
            with self.subTest(events=events):
                path = self.state_dir / "promotions.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
                with self.assertRaises(EvolutionError):
                    self.store._render_learned_playbook()

    def test_corrupt_promotion_history_rejects_unexpected_outer_state(self) -> None:
        event = self.promotion_event()
        event["state"] = "PROMOTED"
        evolution.append_jsonl(self.state_dir / "promotions.jsonl", event)

        with self.assertRaises(EvolutionError):
            self.store._render_learned_playbook()

    def test_corrupt_promotion_history_returns_cli_error(self) -> None:
        self.record()
        evolution.append_jsonl(self.state_dir / "promotions.jsonl", {"event": "UNKNOWN"})
        verifier = Path(self.temp_dir.name) / "verifier.json"
        comparator = Path(self.temp_dir.name) / "comparator.json"
        verifier.write_text(json.dumps(passing_verifier()), encoding="utf-8")
        comparator.write_text(json.dumps(passing_comparator()), encoding="utf-8")
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            result = evolution.main(
                [
                    "--root",
                    str(self.state_dir),
                    "--skill-root",
                    str(self.skill_root),
                    "gate",
                    "cand-1",
                    str(verifier),
                    str(comparator),
                ]
            )

        self.assertEqual(result, 2)
        self.assertIn("evolution error:", stderr.getvalue())


class EvolutionCliTests(unittest.TestCase):
    def test_plan_record_and_status_commands_use_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            skill_root = root / "skill"
            manifest_path = root / "manifest.json"
            evidence_path = root / "evidence.json"
            manifest_path.write_text(json.dumps({"run_id": "run-1"}), encoding="utf-8")
            evidence_path.write_text(json.dumps(valid_evidence()), encoding="utf-8")
            common = ["--root", str(state_dir), "--skill-root", str(skill_root)]

            plan_output = io.StringIO()
            with contextlib.redirect_stdout(plan_output):
                self.assertEqual(evolution.main([*common, "plan", str(manifest_path)]), 0)
            record_output = io.StringIO()
            with contextlib.redirect_stdout(record_output):
                self.assertEqual(evolution.main([*common, "record", str(evidence_path)]), 0)
            status_output = io.StringIO()
            with contextlib.redirect_stdout(status_output):
                self.assertEqual(evolution.main([*common, "status"]), 0)

            self.assertIn('"manifest"', plan_output.getvalue())
            self.assertIn('"state": "EVALUATED"', record_output.getvalue())
            self.assertIn('"cand-1": "EVALUATED"', status_output.getvalue())

    def test_record_cli_rejects_list_direction_without_raw_type_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = valid_evidence()
            evidence["direction"] = ["higher"]
            evidence_path = root / "invalid-evidence.json"
            evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
            stderr = io.StringIO()

            with contextlib.redirect_stderr(stderr):
                result = evolution.main(
                    ["--root", str(root / "state"), "record", str(evidence_path)]
                )

            self.assertEqual(result, 2)
            self.assertIn("evolution error: evidence.direction", stderr.getvalue())

    def test_gate_promote_and_rollback_commands_use_json_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state_dir = root / "state"
            evidence_path = root / "evidence.json"
            verifier_path = root / "verifier.json"
            comparator_path = root / "comparator.json"
            evidence_path.write_text(json.dumps(valid_evidence()), encoding="utf-8")
            verifier_path.write_text(json.dumps(passing_verifier()), encoding="utf-8")
            comparator_path.write_text(json.dumps(passing_comparator()), encoding="utf-8")
            common = ["--root", str(state_dir), "--skill-root", str(root / "skill")]

            self.assertEqual(evolution.main([*common, "record", str(evidence_path)]), 0)
            EvolutionStore(state_dir, root / "skill").save_manifest(provenance_manifest(state_dir))
            gate_output = io.StringIO()
            with contextlib.redirect_stdout(gate_output):
                self.assertEqual(
                    evolution.main(
                        [
                            *common,
                            "gate",
                            "cand-1",
                            str(verifier_path),
                            str(comparator_path),
                        ]
                    ),
                    0,
                )
            promote_output = io.StringIO()
            with contextlib.redirect_stdout(promote_output):
                self.assertEqual(
                    evolution.main(
                        [
                            *common,
                            "promote",
                            "cand-1",
                            str(verifier_path),
                            str(comparator_path),
                        ]
                    ),
                    0,
                )
            promotion_id = json.loads(promote_output.getvalue())["promotion_id"]
            rollback_output = io.StringIO()
            with contextlib.redirect_stdout(rollback_output):
                self.assertEqual(
                    evolution.main(
                        [*common, "rollback", promotion_id, "regression discovered"]
                    ),
                    0,
                )

            self.assertIn('"passed": true', gate_output.getvalue())
            self.assertIn('"candidate_id": "cand-1"', promote_output.getvalue())
            self.assertIn('"event": "ROLLED_BACK"', rollback_output.getvalue())


if __name__ == "__main__":
    unittest.main()
