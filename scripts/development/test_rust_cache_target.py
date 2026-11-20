"""Check the Windows Rust target cache seal and restore check without Cargo."""

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location(
    "rust_cache_target", Path(__file__).with_name("rust-cache-target.py")
)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)

USED = "0123456789abcdef"
STALE = "fedcba9876543210"


def write(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class RustCacheTargetTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)
        self.target = self.root / "target"
        self.manifest_dir = self.root / "manifest"
        self.cargo_home = self.root / "cargo"
        self.ledger = self.root / "ledger.txt"
        deps = self.target / "debug" / "deps"
        write(deps / f"libserde-{USED}.rlib", b"rlib")
        write(deps / f"libserde-{USED}.rmeta", b"meta")
        write(deps / f"serde-{USED}.d", b"d")
        write(deps / f"libserde-{STALE}.rlib", b"stale rlib")
        write(deps / f"serde-{STALE}.d", b"stale d")
        write(deps / "dxcompiler.dll", b"dll")
        write(self.target / "debug" / ".fingerprint" / f"serde-{USED}" / "lib-serde", b"fp")
        write(self.target / "debug" / "build" / f"zstd-sys-{USED}" / "out" / "zstd.lib", b"lib")
        write(self.cargo_home / "registry" / "cache" / "serde.crate", b"crate")
        self.ledger.write_text(
            f"C:\\a\\target\\debug\\deps\\libserde-{USED}.rlib\n"
            f"C:\\a\\target\\debug\\build\\zstd-sys-{USED}\\out\n"
        )

    def tearDown(self):
        self._directory.cleanup()

    def run_quietly(self, *args):
        with redirect_stdout(io.StringIO()) as output:
            code = cache.main([str(arg) for arg in args])
        return code, output.getvalue()

    def seal(self, cap_gib=48):
        return self.run_quietly("seal", "--target", self.target, "--manifest-dir", self.manifest_dir,
                                "--ledger", self.ledger, "--cap-gib", cap_gib)

    def verify(self):
        return self.run_quietly("verify", "--target", self.target, "--manifest-dir", self.manifest_dir,
                                "--cargo-home", self.cargo_home)

    def test_seal_prunes_only_unused_hashed_deps(self):
        self.assertEqual(self.seal()[0], 0)
        deps = self.target / "debug" / "deps"
        self.assertTrue((deps / f"libserde-{USED}.rlib").exists())
        self.assertTrue((deps / f"libserde-{USED}.rmeta").exists())
        self.assertTrue((deps / f"serde-{USED}.d").exists())
        self.assertFalse((deps / f"libserde-{STALE}.rlib").exists())
        self.assertFalse((deps / f"serde-{STALE}.d").exists())
        self.assertTrue((deps / "dxcompiler.dll").exists())

    def test_missing_ledger_keeps_every_output(self):
        self.ledger.unlink()
        self.seal()
        self.assertTrue((self.target / "debug" / "deps" / f"libserde-{STALE}.rlib").exists())

    def test_manifest_lists_hashed_outputs_and_skips_unhashed_files(self):
        self.seal()
        manifest = (self.manifest_dir / cache.MANIFEST_NAME).read_text()
        self.assertIn(f"debug/deps/libserde-{USED}.rlib\t4\n", manifest)
        self.assertIn(f"debug/build/zstd-sys-{USED}/out/zstd.lib\t3\n", manifest)
        self.assertIn(f"debug/.fingerprint/serde-{USED}/lib-serde\t2\n", manifest)
        self.assertNotIn("dxcompiler.dll", manifest)

    def test_complete_restore_is_kept(self):
        self.seal()
        code, output = self.verify()
        self.assertEqual(code, 0)
        self.assertIn("complete", output)
        self.assertTrue((self.target / "debug" / "deps" / f"libserde-{USED}.rlib").exists())
        self.assertTrue((self.cargo_home / "registry").exists())

    def test_missing_output_clears_target_and_registry(self):
        self.seal()
        (self.target / "debug" / "deps" / f"libserde-{USED}.rmeta").unlink()
        code, output = self.verify()
        self.assertEqual(code, 0)
        self.assertIn("incomplete", output)
        self.assertTrue(self.target.is_dir())
        self.assertEqual(list(self.target.iterdir()), [])
        self.assertFalse((self.cargo_home / "registry").exists())

    def test_truncated_output_clears_target(self):
        self.seal()
        write(self.target / "debug" / "deps" / f"libserde-{USED}.rlib", b"")
        self.verify()
        self.assertEqual(list(self.target.iterdir()), [])

    def test_target_without_manifest_is_cleared(self):
        code, output = self.verify()
        self.assertEqual(code, 0)
        self.assertIn("no readable completion manifest", output)
        self.assertEqual(list(self.target.iterdir()), [])

    def test_no_restore_builds_cold_without_touching_registry(self):
        empty = self.root / "empty-target"
        empty.mkdir()
        code, output = self.run_quietly("verify", "--target", empty, "--manifest-dir", self.manifest_dir,
                                        "--cargo-home", self.cargo_home)
        self.assertEqual(code, 0)
        self.assertIn("No Rust target cache was restored", output)
        self.assertTrue((self.cargo_home / "registry").exists())

    def test_target_over_budget_is_not_cached(self):
        code, output = self.seal(cap_gib=1e-9)
        self.assertEqual(code, 0)
        self.assertIn("exceeds", output)
        self.assertTrue(self.target.is_dir())
        self.assertEqual(list(self.target.iterdir()), [])
        # The empty, sealed target restores as complete.
        self.assertIn("complete", self.verify()[1])


if __name__ == "__main__":
    unittest.main()
