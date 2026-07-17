import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import yaml


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
            "skill_curator.sh": 'exec python3 "$script_dir/evolution.py" "$@"',
            "curator_verify.sh": 'exec python3 "$script_dir/evolution.py" gate "$@"',
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

    def test_curator_wrappers_forward_exact_argv_from_an_external_cwd(self) -> None:
        wrappers = {
            "skill_curator.sh": ["status", "--root", "state with spaces"],
            "curator_verify.sh": [
                "candidate-1",
                "verification file.json",
                "comparison.json",
                "--unused=kept",
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external_cwd = root / "external-cwd"
            fake_bin = root / "bin"
            capture = root / "captured-argv.json"
            external_cwd.mkdir()
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n"
                ': > "$ARGV_CAPTURE"\n'
                'for argument in "$@"; do printf "%s\\n" "$argument" >> "$ARGV_CAPTURE"; done\n',
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            environment = os.environ | {
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "ARGV_CAPTURE": str(capture),
            }
            for name, arguments in wrappers.items():
                before = sorted(path.relative_to(external_cwd) for path in external_cwd.rglob("*"))
                subprocess.run(
                    [str(ROOT / "scripts" / name), *arguments],
                    cwd=external_cwd,
                    env=environment,
                    check=True,
                )
                forwarded = capture.read_text(encoding="utf-8").splitlines()
                expected = [str(ROOT / "scripts" / "evolution.py")]
                if name == "curator_verify.sh":
                    expected.append("gate")
                self.assertEqual(forwarded, [*expected, *arguments], name)
                after = sorted(path.relative_to(external_cwd) for path in external_cwd.rglob("*"))
                self.assertEqual(after, before, f"{name} must not write in the caller cwd")

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
            raw = path.read_text(encoding="utf-8")
            text = raw.lower()
            rendered.append(raw)
            workflow = yaml.load(raw, Loader=yaml.BaseLoader)
            self.assertEqual(set(workflow), {"name", "on", "permissions", "jobs"}, str(path))
            self.assertEqual(workflow["on"], {"workflow_dispatch": {}}, str(path))
            self.assertEqual(workflow["permissions"], {"contents": "read"}, str(path))
            self.assertEqual(set(workflow["jobs"]), {"validate"}, str(path))
            steps = workflow["jobs"]["validate"]["steps"]
            self.assertEqual(
                [step.get("uses") for step in steps[:2]],
                ["actions/checkout@v4", "actions/setup-python@v5"],
                str(path),
            )
            validation_lines = [
                line.strip()
                for step in steps
                if "run" in step
                for line in step["run"].splitlines()
                if line.strip()
            ]
            self.assertEqual(
                validation_lines,
                [
                    "python3 -m py_compile scripts/evolution.py",
                    "bash -n scripts/skill_curator.sh scripts/curator_verify.sh",
                    "python3 scripts/evolution.py status",
                ],
                str(path),
            )
            for forbidden in prohibited:
                self.assertNotIn(forbidden, text, f"{path}: {forbidden}")
        self.assertEqual(rendered[0], rendered[1], "workflow copies must not drift")


if __name__ == "__main__":
    unittest.main()
