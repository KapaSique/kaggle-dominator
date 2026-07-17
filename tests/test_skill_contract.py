from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "SKILL.md"


class SkillContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SKILL.read_text(encoding="utf-8")
        match = re.match(r"\A---\n(.*?)\n---\n", cls.text, re.DOTALL)
        if match is None:
            raise AssertionError("SKILL.md must start with YAML frontmatter")
        cls.frontmatter = match.group(1)

    def test_frontmatter_uses_current_minimal_schema(self) -> None:
        keys = re.findall(r"^([a-zA-Z0-9_-]+):", self.frontmatter, re.MULTILINE)
        self.assertEqual(keys, ["name", "description"])
        self.assertIn("name: kaggle-dominator", self.frontmatter)
        self.assertRegex(self.frontmatter, r"description: >-\n\s+Use when ")

    def test_universal_competition_routing(self) -> None:
        for required in (
            "references/tabular.md",
            "references/deep-learning.md",
            "references/simulation.md",
            "references/code-and-hackathon.md",
            "references/campaign-control.md",
        ):
            self.assertIn(required, self.text)

    def test_resource_governance_is_explicit(self) -> None:
        for state in ("ACTIVE", "MONITOR_ONLY", "PAUSED", "CLOSED"):
            self.assertIn(state, self.text)
        self.assertIn("submission", self.text.lower())
        self.assertIn("GPU", self.text)

    def test_safety_and_evidence_gates_are_explicit(self) -> None:
        for phrase in (
            "BEST_KNOWN",
            "metric direction",
            "submissions are enabled",
            "multiple accounts",
            "hidden/private labels",
            "explicit approval",
        ):
            self.assertIn(phrase, self.text)

    def test_new_reference_and_learning_materials_exist(self) -> None:
        for relative in (
            "references/campaign-control.md",
            "references/learning-craft.md",
            "references/resources.md",
            "references/scorecard.md",
            "agents/openai.yaml",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_measured_self_improvement_routes_to_engine_references(self) -> None:
        self.assertIn("references/self-improvement.md", self.text)
        self.assertIn("references/learned-playbook.md", self.text)
        learning_section = self.text.split("## Learning and self-improvement", 1)[1]
        self.assertIn("automatic promotion", learning_section.lower())
        self.assertIn("references/learned-playbook.md", learning_section)
        for protected in ("SKILL.md", "frontmatter", "safety", "policy"):
            self.assertIn(protected, learning_section)

    def test_old_skill_name_is_gone_from_runtime_files(self) -> None:
        runtime_files = [SKILL, *sorted((ROOT / "references").glob("*.md"))]
        runtime_files += sorted((ROOT / "scripts").glob("*"))
        for path in runtime_files:
            if path.is_file():
                self.assertNotIn(
                    "kaggle-practice-coach",
                    path.read_text(encoding="utf-8", errors="replace"),
                    str(path),
                )


if __name__ == "__main__":
    unittest.main()
