import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ci_changes", Path(__file__).with_name("ci-changes.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ChangeRoutingTests(unittest.TestCase):
    def test_prose_and_images(self):
        self.assertTrue(module.documentation_only(["docs/runtime/remote.md", "docs/figure.svg", "CHANGELOG.md"]))

    def test_executable_docs_and_unknown_inputs_retain_native(self):
        for path in ["docs/example.m", "docs/example.fea", "LICENSE.md", "Cargo.lock",
                     "crates/a/src/lib.rs", ".github/workflows/ci.yml", "scripts/tool.py"]:
            self.assertFalse(module.documentation_only(["docs/index.md", path]), path)

    def test_empty_inventory_retains_native(self):
        self.assertFalse(module.documentation_only([]))

    def test_new_branch_and_explicit_validation_retain_native(self):
        self.assertTrue(module.native_required({"before": "0" * 40, "after": "a" * 40}, "push"))
        self.assertTrue(module.native_required({}, "workflow_call"))

    @patch.object(module.subprocess, "run")
    @patch.object(module.subprocess, "check_output", return_value=b"docs/removed.md\0")
    def test_complete_push_tree_diff(self, diff, run):
        run.return_value.returncode = 0
        self.assertFalse(module.native_required({"before": "a" * 40, "after": "b" * 40}, "push"))
        self.assertEqual(diff.call_args.args[0][-2:], ["a" * 40, "b" * 40])

    @patch.object(module.subprocess, "run")
    @patch.object(module.subprocess, "check_output", return_value=b"crates/removed.rs\0")
    def test_deleted_source_retains_native(self, diff, run):
        run.return_value.returncode = 0
        self.assertTrue(module.native_required({"before": "a" * 40, "after": "b" * 40}, "push"))


if __name__ == "__main__":
    unittest.main()
