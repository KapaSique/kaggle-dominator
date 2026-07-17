from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


class WorkflowPolicyTests(unittest.TestCase):
    def read(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def test_non_pokemon_submit_and_keep_warm_schedules_are_disabled(self) -> None:
        for name in ("jed-dojim.yml", "keep-warm.yml", "s6e6-tabpfn3.yml"):
            self.assertNotIn("  schedule:", self.read(name), name)

    def test_scheduled_monitor_is_scoped_to_pokemon_campaign(self) -> None:
        text = self.read("nightly-agent.yml")
        self.assertIn("pokemon-tcg-ai-battle", text)
        self.assertNotIn("COMP_SLUGS: ${{ vars.COMP_SLUGS }}", text)

    def test_scheduled_recon_is_scoped_to_pokemon_ladder(self) -> None:
        text = self.read("re-recon.yml")
        self.assertIn('COMP_SLUGS: "pokemon-tcg-ai-battle"', text)
        self.assertNotIn("COMP_SLUGS: ${{ vars.COMP_SLUGS }}", text)

    def test_curator_wrappers_only_forward_to_local_engine(self) -> None:
        scripts = {
            "skill_curator.sh": 'exec python3 scripts/evolution.py "$@"',
            "curator_verify.sh": 'exec python3 scripts/evolution.py gate "$@"',
        }
        prohibited = (
            "claude -p",
            "kaggle",
            "git ",
            "gh ",
            "github",
            "anthropic",
            "pokemon-tcg",
        )
        for name, expected_forwarder in scripts.items():
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8").lower()
            self.assertIn(expected_forwarder.lower(), text, name)
            for forbidden in prohibited:
                self.assertNotIn(forbidden, text, f"{name}: {forbidden}")

    def test_curator_workflows_are_manual_read_only_validation(self) -> None:
        prohibited = (
            "schedule:",
            "contents: write",
            "pull-requests: write",
            "secrets.",
            "github_token",
            "git push",
            "git commit",
            "gh pr create",
            "gh pr merge",
            "claude -p",
            "kaggle",
            "pokemon-tcg",
        )
        copies = (
            ROOT / "scripts" / "skill-curator.yml",
            WORKFLOWS / "skill-curator.yml",
        )
        rendered = []
        for path in copies:
            text = path.read_text(encoding="utf-8").lower()
            rendered.append(text)
            self.assertIn("workflow_dispatch:", text, str(path))
            self.assertIn("contents: read", text, str(path))
            self.assertIn("python3 scripts/evolution.py status", text, str(path))
            for forbidden in prohibited:
                self.assertNotIn(forbidden, text, f"{path}: {forbidden}")
        self.assertEqual(rendered[0], rendered[1], "workflow copies must not drift")


if __name__ == "__main__":
    unittest.main()
