"""Check scientific routing without invoking Cargo or numerical workloads."""

import importlib.util
import io
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "workspace_test_groups", Path(__file__).with_name("run-workspace-test-group.py")
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class ScientificRoutingTests(unittest.TestCase):
    def test_solver_feature_variants_reuse_distinct_rust_archives(self):
        manifest = Path(__file__).resolve().parents[2] / "crates/runmat-clarabel/Cargo.toml"
        import tomllib
        self.assertEqual(tomllib.loads(manifest.read_text())["lib"]["crate-type"], ["lib"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text('[workspace]\nmembers=["shared", "app"]\nresolver="2"\n')
            for name in ("shared", "app"):
                (root / name / "src").mkdir(parents=True)
            (root / "shared/Cargo.toml").write_text(
                '[package]\nname="cache-shared"\nversion="0.1.0"\nedition="2021"\n'
                '[lib]\ncrate-type=["lib"]\n[features]\nextra=[]\n')
            (root / "shared/src/lib.rs").write_text('pub fn value() -> bool { cfg!(feature="extra") }\n')
            (root / "app/Cargo.toml").write_text(
                '[package]\nname="cache-app"\nversion="0.1.0"\nedition="2021"\n'
                '[dependencies]\ncache-shared={path="../shared"}\n')
            (root / "app/src/lib.rs").write_text('pub fn value() -> bool { cache_shared::value() }\n')
            environment = {**os.environ, "CARGO_TARGET_DIR": str(root / "target"), "CARGO_NET_OFFLINE": "true"}
            observations = []
            for features in (["--features", "cache-shared/extra"], [], ["--features", "cache-shared/extra"]):
                result = subprocess.run(
                    ["cargo", "test", "-p", "cache-app", "--lib", "--message-format=json", *features],
                    cwd=root, env=environment, text=True, capture_output=True, check=True)
                artifacts = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
                artifact = next(row for row in artifacts if row.get("reason") == "compiler-artifact"
                                and row["target"]["name"] == "cache_shared")
                observations.append(artifact)
            self.assertFalse(observations[0]["fresh"])
            self.assertFalse(observations[1]["fresh"])
            self.assertTrue(observations[2]["fresh"])
            self.assertNotEqual(observations[0]["filenames"], observations[1]["filenames"])

    def test_linux_runtime_directory_is_private_live_and_replaces_stale_input(self):
        with patch.object(runner.sys, "platform", "linux"), patch.dict(os.environ, {"XDG_RUNTIME_DIR": "/stale-runtime"}):
            with runner.test_runtime_environment() as environment:
                directory = Path(environment["XDG_RUNTIME_DIR"])
                self.assertEqual(environment["CARGO_PROFILE_DEV_LTO"],
                                 os.environ.get("CARGO_PROFILE_DEV_LTO", "off"))
                self.assertTrue(directory.is_dir())
                if os.name != "nt":
                    self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
                    self.assertEqual(directory.stat().st_uid, os.getuid())
                self.assertEqual(os.environ["XDG_RUNTIME_DIR"], "/stale-runtime")
            self.assertFalse(directory.exists())

    def test_windows_archive_compression_converts_existing_ntfs_archives(self):
        self.assertEqual(runner.WINDOWS_ARCHIVE_COMPRESSION_FLAGS,
                         ("/C", "/F", "/EXE:LZX", "/I", "/Q"))

    def test_cached_workspace_roots_are_reclaimed_without_libraries_or_build_scripts(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            metadata = {"packages": [{"targets": [{"name": "my-bin"}, {"name": "integration"}]}]}
            candidates = ["debug/my-bin.exe", "debug/deps/my_bin-0123456789abcdef.exe",
                          "debug/examples/integration-0123456789abcdef.exe"]
            retained = ["debug/deps/my_bin-0123456789abcdef.rlib",
                        "debug/deps/unrelated-0123456789abcdef.exe",
                        "debug/build/my-bin/build-script-build.exe",
                        "debug/deps/my_bin-0123456789abcdef.dll"]
            for name in candidates + retained:
                file = target / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"test")
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}):
                with self.assertRaisesRegex(RuntimeError, "isolated"):
                    runner.reclaim_cached_root_binaries(target, metadata)
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
                self.assertEqual(runner.reclaim_cached_root_binaries(target, metadata), 3)
            self.assertTrue(all(not (target / name).exists() for name in candidates))
            self.assertTrue(all((target / name).exists() for name in retained))

    def test_batches_cover_roots_once_and_keep_package_feature_union(self):
        members = runner.GROUPS["language"]
        metadata = {"packages": [{"name": name, "targets": [
            {"name": name, "kind": ["lib"]},
            *[{"name": f"integration_{index}", "kind": ["test"]} for index in range(11)],
            {"name": "demo", "kind": ["example"]},
        ]} for name in members]}
        commands = runner.test_commands("language", True, metadata)
        for count in (2, 4):
            shards = [runner.shard_commands(commands, index, count) for index in range(count)]
            self.assertEqual(Counter(tuple(command) for command in commands),
                             Counter(tuple(command) for shard in shards for command in shard))
            for index, shard in enumerate(shards):
                for other in shards[index + 1:]:
                    self.assertFalse(set(map(tuple, shard)) & set(map(tuple, other)))
        self.assertEqual(runner.shard_commands(commands, 0, 1), commands)
        for index, count in ((-1, 2), (2, 2), (0, 0), (0, len(commands) + 1)):
            with self.assertRaises(ValueError):
                runner.shard_commands(commands, index, count)
        roots = [(flag, command[index + 1]) for command in commands
                 for index, flag in enumerate(command) if flag in ("--test", "--example")]
        self.assertEqual(len(roots), 12)
        self.assertEqual(len(set(roots)), 12)
        self.assertEqual(sum("--lib" in command for command in commands), 1)
        for command in commands:
            self.assertIn("--all-features", command)
            self.assertEqual({command[index + 1] for index, token in enumerate(command)
                              if token == "--package"}, set(members))
            self.assertLessEqual(command.count("--test"), 4)

    def test_unicode_child_output_survives_redirected_legacy_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = root / "unicode_child.py"
            child.write_text('import sys\nsys.stdout.buffer.write("\\u03f5\\n".encode("utf-8"))\n', encoding="utf-8")
            output_bytes, error_bytes = io.BytesIO(), io.BytesIO()
            output = io.TextIOWrapper(output_bytes, encoding="cp1252", newline="\n")
            errors = io.TextIOWrapper(error_bytes, encoding="cp1252", newline="\n")
            with patch.object(runner.sys, "stdout", output), patch.object(runner.sys, "stderr", errors):
                runner.configure_output_streams()
                code = runner.run_batch([sys.executable, str(child)], root, root / "target", False)
                print("\u03f5", file=runner.sys.stderr, flush=True)
            output.flush()
            errors.flush()
            self.assertEqual(code, 0)
            self.assertIn("\u03f5".encode("utf-8"), output_bytes.getvalue())
            self.assertIn(b"Batch exit: 0", output_bytes.getvalue())
            self.assertEqual(error_bytes.getvalue(), "\u03f5\n".encode("utf-8"))

    def test_cargo_batch_selection_and_reclamation(self):
        # Exercise Cargo's actual mixed-package root selection and JSON stream.
        # A package without a selected target must keep contributing features.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").write_text('[workspace]\nmembers=["a", "b", "shared"]\nresolver="2"\n')
            (root / "shared/src").mkdir(parents=True)
            (root / "shared/Cargo.toml").write_text('[package]\nname="shared"\nversion="0.1.0"\nedition="2021"\n[features]\nfrom_b=[]\n')
            (root / "shared/src/lib.rs").write_text('pub fn value() -> u8 { if cfg!(feature="from_b") { 42 } else { 0 } }\n')
            for name in ("a", "b"):
                member = root / name
                (member / "src").mkdir(parents=True)
                features = ', features=["from_b"]' if name == "b" else ''
                dependency_kind = "dev-dependencies" if name == "b" else "dependencies"
                (member / "Cargo.toml").write_text(f'[package]\nname="{name}"\nversion="0.1.0"\nedition="2021"\n[{dependency_kind}]\nshared={{path="../shared"{features}}}\n')
                (member / "src/lib.rs").write_text('pub fn value() -> u8 { 42 }\n')
            (root / "a/tests").mkdir()
            (root / "a/src/main.rs").write_text('fn main() { println!("42"); }\n')
            (root / "a/tests/only_a.rs").write_text('#[test] fn smoke() { assert_eq!(a::value(), 42); assert_eq!(shared::value(), 42); }\n')
            target = root / "target"
            with patch.dict(os.environ, {"CARGO_TARGET_DIR": str(target), "CARGO_NET_OFFLINE": "true"}):
                result = runner.run_batch(["cargo", "test", "-p", "a", "-p", "b", "--test", "only_a"],
                                          root, target, True)
            self.assertEqual(result, 0)
            self.assertFalse(any(path.is_file() and os.access(path, os.X_OK)
                                 and path.suffix != ".d" for path in (target / "debug/deps").glob("only_a-*")))
            self.assertTrue(list((target / "debug/deps").glob("liba-*.rlib")))
            self.assertTrue((target / "debug" / ("a.exe" if os.name == "nt" else "a")).is_file())
            if os.name == "nt":
                libraries = {path: path.read_bytes() for path in (target / "debug/deps").glob("*.rlib")}
                usage = runner.shutil.disk_usage(target)
                with patch.object(runner, "WINDOWS_ARCHIVE_MIN_BYTES", 1), \
                        patch.object(runner.shutil, "disk_usage", return_value=usage._replace(free=0)):
                    self.assertGreater(runner.compact_windows_archives(target), 0)
                    self.assertEqual(runner.compact_windows_archives(target), 0)
                for path, contents in libraries.items():
                    self.assertEqual(path.read_bytes(), contents)
                    self.assertTrue(path.stat().st_file_attributes & runner.stat.FILE_ATTRIBUTE_COMPRESSED)
                # Relink the reclaimed test root against the retained compressed
                # libraries. Cargo must reuse those exact dependency archives.
                with patch.dict(os.environ, {"CARGO_TARGET_DIR": str(target), "CARGO_NET_OFFLINE": "true"}):
                    self.assertEqual(runner.run_batch(
                        ["cargo", "test", "-p", "a", "-p", "b", "--test", "only_a"], root, target, True), 0)
                for path, contents in libraries.items():
                    self.assertEqual(path.read_bytes(), contents)
                    self.assertTrue(path.stat().st_file_attributes & runner.stat.FILE_ATTRIBUTE_COMPRESSED)

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

    def test_full_gear_witness_routes_to_manual_scientific_qualification(self):
        name = "induction_gear_example_runs_its_three_frequency_harmonic_study"
        self.assertIn(name, runner.scientific_filters("tooling-distributed"))
        self.assertNotIn(name, runner.scientific_filters("geometry-analysis"))
        policy_path = Path(__file__).resolve().parents[2] / "qualification/scientific/policy.json"
        cohort = next(c for c in json.loads(policy_path.read_text())["cohorts"]
                      if c["id"] == "induction-gear-full-model")
        self.assertIn(name, cohort["command"])
        self.assertFalse(cohort["required"])
        self.assertIn("outstanding beta resource limitation", cohort["exception"])

    def test_extended_commands_do_not_depend_on_skip_policy(self):
        with patch.object(runner, "scientific_filters", side_effect=AssertionError("read skip policy")):
            for group in runner.GROUPS:
                with self.subTest(group=group):
                    self.assertTrue(runner.test_commands(group, include_extended=True))

    def test_boolean_step_corpus_routes_to_manual_without_skipping_bounded_box_checks(self):
        policy_path = Path(__file__).resolve().parents[2] / "qualification/scientific/policy.json"
        cohort = next(c for c in json.loads(policy_path.read_text())["cohorts"]
                      if c["id"] == "occt-boolean-step-corpus")
        self.assertFalse(cohort["required"])
        self.assertIn("failed constrained-Delaunay facet recovery", cohort["exception"])
        self.assertIn("--include-ignored", cohort["command"])
        self.assertIn("occt_boolean_step_meshing", cohort["command"])
        self.assertEqual(len(cohort["ordinary_skip"]), 6)
        self.assertTrue(set(cohort["ordinary_skip"]).issubset(
            runner.scientific_filters("tooling-distributed")))
        for name in ("occt_box_executes_the_complete_serial_exact_dag",
                     "runmat_and_independent_meshers_agree_on_box_invariants",
                     "occt_box_artifact_is_independent_of_legal_partition_layout",
                     "disconnected_cad_solids_retain_distinct_region_identity_through_meshing"):
            self.assertFalse(any(pattern in name for pattern in
                                 runner.scientific_filters("tooling-distributed")))

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
