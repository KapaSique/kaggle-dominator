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

    @staticmethod
    def parse_workflow(raw: str) -> dict:
        workflow = yaml.load(raw, Loader=yaml.BaseLoader)
        assert isinstance(workflow, dict)
        return workflow

    def test_repository_workflows_have_no_scheduled_bypass(self) -> None:
        prohibited = (
            "contents: write",
            "pull-requests: write",
            "git commit",
            "git push",
            "gh pr",
            "claude -p",
            "anthropic",
            "secrets.",
            "kaggle competitions submit",
            "kaggle kernels push",
        )
        for path in sorted(WORKFLOWS.glob("*.yml")):
            raw = path.read_text(encoding="utf-8")
            workflow = self.parse_workflow(raw)
            triggers = workflow.get("on", {})
            self.assertIsInstance(triggers, dict, path)
            self.assertNotIn("schedule", triggers, path)
            if "schedule" in triggers:
                permissions = workflow.get("permissions", {})
                self.assertIsInstance(permissions, dict, path)
                self.assertTrue(
                    all(value == "read" for value in permissions.values()), path
                )
                for forbidden in prohibited:
                    self.assertNotIn(forbidden, raw.lower(), f"{path}: {forbidden}")

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

    def test_retired_legacy_scripts_are_fail_closed_and_cannot_mutate_state(self) -> None:
        prohibited = (
            "claude -p",
            "kaggle kernels push",
            "kaggle competitions submit",
            "best_known",
            "grandmaster-playbook",
            "skill.md",
            "git commit",
            "git push",
            "gh ",
        )
        for name in ("kaggle_monitor.sh", "kaggle_eval_loop.sh", "re_recon.sh"):
            path = ROOT / "scripts" / name
            text = path.read_text(encoding="utf-8").lower()
            self.assertIn("is retired and performs no action", text, name)
            self.assertIn("exit 2", text, name)
            for forbidden in prohibited:
                self.assertNotIn(forbidden, text, f"{name}: {forbidden}")

    def test_runtime_references_and_scripts_have_no_legacy_bypass_commands(self) -> None:
        prohibited = (
            "claude -p",
            "kaggle kernels push",
            "kaggle competitions submit",
            "git commit",
            "git push",
            "anthropic_api_key",
        )
        paths = [*ROOT.joinpath("scripts").glob("*.sh"), *ROOT.joinpath("references").glob("*.md")]
        for path in paths:
            text = path.read_text(encoding="utf-8").lower()
            for forbidden in prohibited:
                self.assertNotIn(forbidden, text, f"{path}: {forbidden}")

        autonomous = (ROOT / "references" / "autonomous.md").read_text(encoding="utf-8").lower()
        for forbidden in ("headless", "cron", "github actions", "best_known", "grandmaster-playbook.md"):
            self.assertNotIn(forbidden, autonomous, f"autonomous.md: {forbidden}")
        self.assertIn("trusted codex daily orchestration", autonomous)
        self.assertIn("scripts/evolution.py", autonomous)
        self.assertIn("references/learned-playbook.md", autonomous)
        self.assertIn("explicit approval", autonomous)

    def test_retired_workflows_are_manual_read_only_validation(self) -> None:
        prohibited = (
            "schedule:",
            "contents: write",
            "pull-requests: write",
            "secrets.",
            "github_token",
            "git push",
            "git commit",
            "gh pr",
            "claude -p",
            "kaggle competitions submit",
            "kaggle kernels push",
            "anthropic",
        )
        copies = (
            ROOT / "scripts" / "nightly-agent.yml",
            WORKFLOWS / "nightly-agent.yml",
        )
        expected_lines = [
            "python3 -m py_compile scripts/evolution.py",
            "bash -n scripts/kaggle_monitor.sh scripts/kaggle_eval_loop.sh scripts/re_recon.sh",
            "python3 -m unittest -v tests.test_workflow_policy",
        ]
        rendered = []
        for path in (*copies, WORKFLOWS / "re-recon.yml"):
            raw = path.read_text(encoding="utf-8")
            rendered.append(raw)
            workflow = self.parse_workflow(raw)
            self.assertEqual(set(workflow), {"name", "on", "permissions", "jobs"}, path)
            self.assertEqual(workflow["on"], {"workflow_dispatch": {}}, path)
            self.assertEqual(workflow["permissions"], {"contents": "read"}, path)
            self.assertEqual(set(workflow["jobs"]), {"validate-retirement"}, path)
            steps = workflow["jobs"]["validate-retirement"]["steps"]
            self.assertEqual(
                [step.get("uses") for step in steps[:2]],
                ["actions/checkout@v4", "actions/setup-python@v5"],
                path,
            )
            validation_lines = [
                line.strip()
                for step in steps
                if "run" in step
                for line in step["run"].splitlines()
                if line.strip()
            ]
            self.assertEqual(validation_lines, expected_lines, path)
            for forbidden in prohibited:
                self.assertNotIn(forbidden, raw.lower(), f"{path}: {forbidden}")
        self.assertEqual(rendered[0], rendered[1], "nightly workflow copies must not drift")

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
