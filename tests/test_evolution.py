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


if __name__ == "__main__":
    unittest.main()
