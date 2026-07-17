"""Portable contract checks for evolution-engine agent prompts and evals."""

from __future__ import annotations

import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "agents"
EVALS = ROOT / "evals" / "evals.json"
SKILL_CARD = ROOT / "skill-card.md"

EVIDENCE_FIELDS = {
    "candidate_id", "parent_id", "competition", "competition_type", "claim",
    "scope_limits", "metric", "direction", "metric_direction_verified",
    "baseline_score", "candidate_score", "noise_floor", "confirmations",
    "validation_regimes", "code_sha", "data_fingerprint", "config_hash", "seeds",
    "runtime_minutes", "runtime_ratio", "vram_gb", "artifacts", "regressions",
    "forbidden_actions", "changed_paths", "transferable", "status", "created_at_utc",
}
VERIFICATION_FIELDS = {
    "candidate_id", "verdict", "fresh_context", "reviewer_id", "checked_artifacts", "issues",
}
COMPARATOR_FIELDS = {"candidate_id", "winner", "blind", "rubric"}
REQUIRED_SCENARIOS = {
    "monitor-only", "active-no-submit", "reject-noisy-lb", "reject-protected-path",
}


def json_examples(text: str) -> list[dict]:
    """Return every JSON object embedded in a Markdown fenced code block."""
    examples = []
    for block in re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL):
        examples.append(json.loads(block))
    return examples


class EvolutionContractTests(unittest.TestCase):
    def test_required_contract_files_exist(self) -> None:
        for path in (
            AGENTS / "evolution-scout.md",
            AGENTS / "evolution-proposer.md",
            AGENTS / "evolution-verifier.md",
            AGENTS / "evolution-comparator.md",
            EVALS,
            SKILL_CARD,
        ):
            self.assertTrue(path.is_file(), path)

    def test_agent_outputs_are_single_immutable_json_artifacts(self) -> None:
        for name in ("evolution-scout.md", "evolution-proposer.md", "evolution-verifier.md", "evolution-comparator.md"):
            text = (AGENTS / name).read_text(encoding="utf-8")
            self.assertIn("one immutable JSON output", text, name)
            self.assertIn("terminal status", text, name)
            self.assertIn("artifact paths", text, name)
            self.assertIn("UTC timestamps", text, name)
            self.assertIn("shared mutable", text, name)

    def test_evidence_schema_is_exact_and_machine_readable(self) -> None:
        examples = json_examples((AGENTS / "evolution-proposer.md").read_text(encoding="utf-8"))
        evidence = next(example for example in examples if "candidate_score" in example)
        self.assertEqual(set(evidence), EVIDENCE_FIELDS)
        self.assertEqual(evidence["status"], "succeeded")
        self.assertEqual(evidence["created_at_utc"], "2026-07-17T00:00:00Z")

    def test_verifier_has_fresh_context_and_no_self_justification(self) -> None:
        text = (AGENTS / "evolution-verifier.md").read_text(encoding="utf-8")
        verification = json_examples(text)[0]
        self.assertEqual(set(verification), VERIFICATION_FIELDS)
        self.assertTrue(verification["fresh_context"])
        self.assertIn("candidate brief, raw evidence, artifact pointers, and diff package", text)
        self.assertIn("never proposer justification", text)
        self.assertIn("must not verify its own", text)

    def test_comparator_is_blind_and_uses_exact_schema(self) -> None:
        text = (AGENTS / "evolution-comparator.md").read_text(encoding="utf-8")
        comparison = json_examples(text)[0]
        self.assertEqual(set(comparison), COMPARATOR_FIELDS)
        self.assertTrue(comparison["blind"])
        self.assertEqual(set(comparison["rubric"]), {"score", "stability", "runtime", "reproducibility"})
        self.assertIn("never reveal incumbent or challenger identity", text)

    def test_evals_are_valid_unique_and_cover_required_scenarios(self) -> None:
        payload = json.loads(EVALS.read_text(encoding="utf-8"))
        self.assertIsInstance(payload, dict)
        scenarios = payload["scenarios"]
        self.assertTrue(scenarios)
        ids = [scenario["id"] for scenario in scenarios]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(REQUIRED_SCENARIOS.issubset(ids))
        for scenario in scenarios:
            self.assertTrue(scenario["prompt"])
            self.assertTrue(scenario["expected_behavior"])
            self.assertTrue(scenario["success_criteria"])

    def test_skill_card_protects_external_actions_and_core_skill_paths(self) -> None:
        text = SKILL_CARD.read_text(encoding="utf-8")
        for phrase in (
            "explicit user approval",
            "Kaggle submissions",
            "GitHub writes",
            "SKILL.md",
            "YAML frontmatter",
            "agents/openai.yaml",
            "credential logic",
            "campaign authority",
            "submission policy",
            "references/learned-playbook.md",
            "fail closed",
        ):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
