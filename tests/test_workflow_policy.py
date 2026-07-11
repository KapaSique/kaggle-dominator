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

    def test_curator_harvests_only_current_campaign(self) -> None:
        text = self.read("skill-curator.yml")
        command = (
            "./scripts/skill_curator.sh . pokemon-tcg-ai-battle "
            "pokemon-tcg-ai-battle-challenge-strategy"
        )
        self.assertIn(command, text)


if __name__ == "__main__":
    unittest.main()

