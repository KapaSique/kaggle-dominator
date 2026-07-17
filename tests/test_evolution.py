import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
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
        return self.store.record_evidence(evidence)

    def gate(self, candidate_id: str = "cand-1", **changes: object) -> object:
        self.record(candidate_id, **changes)
        return self.store.gate_candidate(
            candidate_id, passing_verifier(candidate_id), passing_comparator(candidate_id)
        )

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
