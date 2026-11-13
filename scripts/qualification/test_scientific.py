import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("scientific", Path(__file__).with_name("scientific.py"))
scientific = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scientific)


class ScientificEvidenceTests(unittest.TestCase):
    def fixture(self):
        inputs = {"sha256": "a" * 64, "files": 3, "encoding": "git-ls-tree-r-z-mode-type-blob-path-v1"}
        cohort = {"id": "sample", "required": True, "expected_passed": 2, "command": ["cargo", "test"]}
        receipt = {"schema_version": 1, "cohort": "sample", "policy_sha256": "b" * 64,
                   "measured": {"source_revision": "c" * 40, "target": "linux-x86_64",
                                "toolchain": "1.90.0", "command": ["cargo", "test"],
                                "log_sha256": "d" * 64, "log_location": "retained/sample.log",
                                "artifacts": [{"sha256": "e" * 64, "location": "retained/test-binary"}],
                                "exit_code": 0, "passed": 2, "failed": 0, "ignored": 0,
                                "case_names": ["case_one", "case_two"],
                                "filtered_out": 0, "inputs": inputs}}
        return inputs, cohort, receipt

    def test_missing_cases_wrong_policy_and_changed_inputs_are_rejected(self):
        inputs, cohort, receipt = self.fixture()
        scientific.verify_receipt(Path.cwd(), cohort, receipt, inputs, "b" * 64)
        for key, value in [("passed", 1), ("failed", 1), ("ignored", 1), ("filtered_out", 1), ("exit_code", 101)]:
            changed = copy.deepcopy(receipt)
            changed["measured"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                scientific.verify_receipt(Path.cwd(), cohort, changed, inputs, "b" * 64)
        for current, policy_hash in [({**inputs, "sha256": "f" * 64}, "b" * 64), (inputs, "f" * 64)]:
            with self.assertRaises(ValueError):
                scientific.verify_receipt(Path.cwd(), cohort, receipt, current, policy_hash)

    def test_wrong_identity_and_incomplete_evidence_are_rejected(self):
        inputs, cohort, receipt = self.fixture()
        for field, value in [("schema_version", 2), ("cohort", "another-cohort")]:
            changed = copy.deepcopy(receipt)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                scientific.verify_receipt(Path.cwd(), cohort, changed, inputs, "b" * 64)
        for field, value in [("source_revision", "HEAD"), ("command", "cargo test"),
                             ("log_sha256", "bad"), ("artifacts", []),
                             ("case_names", ["case_one", "case_one"]), ("target", "")]:
            changed = copy.deepcopy(receipt)
            changed["measured"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                scientific.verify_receipt(Path.cwd(), cohort, changed, inputs, "b" * 64)

    def test_command_must_match_the_policy_argument_vector(self):
        inputs, cohort, receipt = self.fixture()
        for command in (["true"], ["cargo", "test", "--release"], ["test", "cargo"]):
            changed = copy.deepcopy(receipt)
            changed["measured"]["command"] = command
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "command differs"):
                scientific.verify_receipt(Path.cwd(), cohort, changed, inputs, "b" * 64)

    def test_historical_import_is_pinned_to_approval_cohort_and_complete_measurement(self):
        inputs, cohort, receipt = self.fixture()
        receipt["measured"]["command"] = ["cargo", "test", "--all-targets"]
        receipt["measured"]["profile"] = {"debug": 0, "release": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "historical.json"
            record = {"schema_version": 1, "cohort": cohort["id"], "measured": receipt["measured"]}
            path.write_text(json.dumps(record))
            reference = {"path": path.name, "sha256": scientific.digest(path.read_bytes())}
            receipt["historical_measurement"] = reference
            cohort["approved_historical_measurements"] = [reference]
            scientific.verify_receipt(root, cohort, receipt, inputs, "b" * 64)
            unapproved = {**cohort, "approved_historical_measurements": []}
            with self.assertRaisesRegex(ValueError, "not explicitly approved"):
                scientific.verify_receipt(root, unapproved, receipt, inputs, "b" * 64)
            for field, value in [("toolchain", "another-toolchain"), ("profile", {"debug": False, "release": False})]:
                changed = copy.deepcopy(receipt)
                changed["measured"][field] = value
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "changes the approved"):
                    scientific.verify_receipt(root, cohort, changed, inputs, "b" * 64)
            path.write_text(json.dumps({**record, "cohort": "another-cohort"}))
            with self.assertRaisesRegex(ValueError, "missing or changed"):
                scientific.verify_receipt(root, cohort, receipt, inputs, "b" * 64)
            reference["sha256"] = scientific.digest(path.read_bytes())
            with self.assertRaisesRegex(ValueError, "wrong cohort"):
                scientific.verify_receipt(root, cohort, receipt, inputs, "b" * 64)

    def test_historical_approval_does_not_bypass_case_counts_or_source_conservation(self):
        inputs, cohort, receipt = self.fixture()
        receipt["measured"]["command"] = ["cargo", "test", "--all-targets"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "historical.json"
            def approve():
                path.write_text(json.dumps({"schema_version": 1, "cohort": cohort["id"], "measured": receipt["measured"]}))
                reference = {"path": path.name, "sha256": scientific.digest(path.read_bytes())}
                receipt["historical_measurement"] = reference
                cohort["approved_historical_measurements"] = [reference]
            approve()
            with self.assertRaisesRegex(ValueError, "without reviewed conservation"):
                scientific.verify_receipt(root, cohort, receipt, {**inputs, "sha256": "f" * 64}, "b" * 64)
            receipt["measured"].update(passed=1, case_names=["case_one"])
            approve()
            with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
                scientific.verify_receipt(root, cohort, receipt, inputs, "b" * 64)

    def test_result_counts_are_explicit_nonnegative_integers(self):
        inputs, cohort, receipt = self.fixture()
        for field in ("passed", "failed", "ignored", "filtered_out", "exit_code"):
            for value in (None, -1, False, True, 0.0, "0"):
                changed = copy.deepcopy(receipt)
                if value is None:
                    del changed["measured"][field]
                else:
                    changed["measured"][field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, "integer"):
                    scientific.verify_receipt(Path.cwd(), cohort, changed, inputs, "b" * 64)

    def test_selected_filter_matches_even_when_no_cases_are_filtered_out(self):
        inputs, cohort, receipt = self.fixture()
        cohort["selected_filter"] = "case_"
        for filtered_out in (0, 5):
            for selected_filter in (None, "another_"):
                changed = copy.deepcopy(receipt)
                changed["measured"].update(filtered_out=filtered_out, selected_filter=selected_filter)
                with self.subTest(count=filtered_out, selection=selected_filter), self.assertRaisesRegex(ValueError, "filtering differs"):
                    scientific.verify_receipt(Path.cwd(), cohort, changed, inputs, "b" * 64)
            receipt["measured"].update(filtered_out=filtered_out, selected_filter="case_")
            scientific.verify_receipt(Path.cwd(), cohort, receipt, inputs, "b" * 64)
        del cohort["selected_filter"]
        receipt["measured"]["filtered_out"] = 0
        with self.assertRaisesRegex(ValueError, "filtering differs"):
            scientific.verify_receipt(Path.cwd(), cohort, receipt, inputs, "b" * 64)

    def test_routed_names_and_target_must_match_the_cohort(self):
        inputs, cohort, receipt = self.fixture()
        cohort.update(ordinary_skip=["case_one", "case_two"], required_target="linux-x86_64")
        scientific.verify_receipt(Path.cwd(), cohort, receipt, inputs, "b" * 64)
        for field, value in [("case_names", ["case_one", "unrelated"]), ("target", "macos-arm64")]:
            changed = copy.deepcopy(receipt)
            changed["measured"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                scientific.verify_receipt(Path.cwd(), cohort, changed, inputs, "b" * 64)

    def test_evidence_paths_reject_traversal_and_external_symlinks(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            outside = Path(external) / "evidence.json"
            outside.write_text("{}")
            (root / "linked.json").symlink_to(outside)
            for relative in ("", "../evidence.json", str(outside), "linked.json"):
                with self.subTest(path=relative), self.assertRaises(ValueError):
                    scientific.safe_file(root, relative)

    def test_conservation_binds_both_snapshots_and_its_qualification(self):
        inputs, cohort, receipt = self.fixture()
        before = ["100644", "blob", "a" * 40]
        after = ["100644", "blob", "f" * 40]
        inputs = scientific.inventory_identity({"crates/plot/camera.rs": before})
        accepted = scientific.inventory_identity({"crates/plot/camera.rs": after})
        receipt["measured"]["inputs"] = inputs
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proofs = root / "qualification/scientific"
            proofs.mkdir(parents=True)
            witness = proofs / "affected-check.json"
            witness.write_text('{"passed": 3, "failed": 0}')
            review = proofs / "review.json"
            record = {"measured_inputs": inputs, "accepted_inputs": accepted, "cohorts": ["sample"],
                      "changed_inputs": [{"path": "crates/plot/camera.rs", "before": before, "after": after, "reason": "Camera-only change"}],
                      "qualification_evidence": [{"path": "qualification/scientific/affected-check.json",
                                                  "sha256": scientific.digest(witness.read_bytes())}]}
            review.write_text(json.dumps(record))
            receipt["conservation"] = {"measured_inputs": inputs, "accepted_inputs": accepted,
                                       "reason": "Reviewed source-only transition",
                                       "review": {"path": "qualification/scientific/review.json",
                                                  "sha256": scientific.digest(review.read_bytes())}}
            with patch.object(scientific, "source_entries", return_value={"crates/plot/camera.rs": after}):
                scientific.verify_receipt(root, cohort, receipt, accepted, "b" * 64)
                witness.write_text('{"passed": 0, "failed": 1}')
                with self.assertRaisesRegex(ValueError, "qualification evidence"):
                    scientific.verify_receipt(root, cohort, receipt, accepted, "b" * 64)

    def test_source_inventory_detects_dirty_and_added_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, stderr=subprocess.DEVNULL)
            git("init")
            git("config", "user.name", "Qualification fixture")
            git("config", "user.email", "fixture@invalid.example")
            (root / "qualification/scientific").mkdir(parents=True)
            (root / "qualification/scientific/policy.json").write_text(json.dumps({
                "schema_version": 1, "input_roots": ["crates"],
                "cohorts": [{"id": "sample", "required": True}]}))
            (root / "crates").mkdir()
            source = root / "crates/solver.rs"
            source.write_text("const X: u32 = 1;\n")
            git("add", ".")
            git("-c", "commit.gpgsign=false", "commit", "-m", "Fixture")
            before = scientific.input_identity(root)
            source.write_text("const X: u32 = 2;\n")
            with self.assertRaises(ValueError):
                scientific.input_identity(root)
            git("add", ".")
            git("-c", "commit.gpgsign=false", "commit", "-m", "Changed fixture")
            self.assertNotEqual(before, scientific.input_identity(root))
            (root / "crates/new.rs").write_text("// New source\n")
            with self.assertRaises(ValueError):
                scientific.input_identity(root)


if __name__ == "__main__":
    unittest.main()
