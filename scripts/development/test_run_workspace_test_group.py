"""Check scientific routing without invoking Cargo or numerical workloads."""

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "workspace_test_groups", Path(__file__).with_name("run-workspace-test-group.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ScientificRoutingTests(unittest.TestCase):
    def test_native_stack_matches_command_host_without_mutating_parent(self):
        with patch.dict(runner.os.environ, {"RUST_MIN_STACK": "2097152", "RUST_TEST_THREADS": "1"}):
            environment = runner.test_environment()
            self.assertEqual(environment["RUST_MIN_STACK"], str(64 * 1024 * 1024))
            self.assertEqual(environment["RUST_TEST_THREADS"], "1")
            self.assertEqual(runner.os.environ["RUST_MIN_STACK"], "2097152")

    def test_standalone_benchmarks_have_separate_bounded_invocations(self):
        commands = [command for group in runner.GROUPS for command in runner.benchmark_commands(group)]
        self.assertEqual(len(commands), len(runner.STANDALONE_BENCHMARK_SMOKES))
        for command in commands:
            self.assertNotIn("--skip", command)
            iterations, samples = command[command.index("--") + 1:]
            self.assertGreater(int(iterations), 0)
            self.assertLessEqual(int(iterations), 1000)
            self.assertEqual(int(samples), 3)
        for group in runner.GROUPS:
            expected = {package for package in runner.GROUPS[group]
                        if package not in runner.PACKAGE_TEST_FEATURE_PROFILES}
            for command in runner.benchmark_commands(group):
                actual = {command[index + 1] for index, token in enumerate(command)
                          if token == "--package"}
                self.assertEqual(actual, expected)
            for command in runner.test_commands(group, include_extended=False):
                self.assertNotIn("--all-targets", command)
                self.assertTrue({"--lib", "--bins", "--tests", "--examples"}.issubset(command))

    def test_ordinary_commands_skip_exactly_declared_group_patterns(self):
        policy_path = Path(__file__).resolve().parents[2] / "qualification/scientific/policy.json"
        policy = json.loads(policy_path.read_text())
        for group in runner.GROUPS:
            expected = [pattern for cohort in policy["cohorts"] if cohort.get("group") == group
                        for pattern in cohort["ordinary_skip"]]
            with self.subTest(group=group):
                ordinary = runner.test_commands(group, include_extended=False)
                extended = runner.test_commands(group, include_extended=True)
                self.assertEqual(len(ordinary), len(extended))
                for command, complete in zip(ordinary, extended):
                    self.assertNotIn("--skip", complete)
                    suffix = ["--", *(item for pattern in expected for item in ("--skip", pattern))] if expected else []
                    self.assertEqual(command, complete + suffix)

    def test_extended_commands_do_not_depend_on_skip_policy(self):
        with patch.object(runner, "scientific_filters", side_effect=AssertionError("read skip policy")):
            for group in runner.GROUPS:
                with self.subTest(group=group):
                    self.assertTrue(runner.test_commands(group, include_extended=True))

    def test_policy_patterns_are_scoped_to_the_selected_group(self):
        policy = {"schema_version": 1, "cohorts": [
            {"group": "runtime", "ordinary_skip": ["first::", "second::"]},
            {"group": "runtime", "ordinary_skip": ["third::"]},
            {"group": "tooling-distributed", "ordinary_skip": ["other::"]},
            {"ordinary_skip": ["unassigned::"]},
        ]}
        with patch.object(Path, "read_text", return_value=json.dumps(policy)):
            self.assertEqual(runner.scientific_filters("runtime"), ("first::", "second::", "third::"))
            self.assertEqual(runner.scientific_filters("language"), ())


if __name__ == "__main__":
    unittest.main()
