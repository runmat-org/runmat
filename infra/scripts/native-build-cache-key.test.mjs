import assert from "node:assert/strict";
import test from "node:test";
import { nativeBuildCacheKey } from "./native-build-cache-key.mjs";
import { captureSource, restoreSource } from "./native-build-cache-source.mjs";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

test("native checkpoints conserve a compatible family and separate incompatible inputs", () => {
  const identity = { compiler: "rustc 1.90.0\nhost: aarch64-apple-darwin\n",
    target: "x86_64-apple-darwin", inputs: "a".repeat(64), platform: "darwin", architecture: "arm64" };
  const key = nativeBuildCacheKey(identity);
  assert.equal(nativeBuildCacheKey({ ...identity }), key);
  assert.notEqual(nativeBuildCacheKey({ ...identity, environment: { RUSTFLAGS: "-C target-cpu=native" } }), key);
  for (const field of ["compiler", "target", "platform", "architecture", "inputs"]) {
    const value = field === "inputs" ? "b".repeat(64) : `${identity[field]}-changed`;
    assert.notEqual(nativeBuildCacheKey({ ...identity, [field]: value }), key);
    assert.throws(() => nativeBuildCacheKey({ ...identity, [field]: "" }));
  }
});

test("a same-commit checkpoint reuses identical inputs and rebuilds changed bytes", () => {
  const root = mkdtempSync(join(tmpdir(), "runmat-cache-proof-"));
  try {
    mkdirSync(join(root, "src"));
    writeFileSync(join(root, "Cargo.toml"), '[package]\nname="cache-proof"\nversion="0.1.0"\nedition="2021"\n');
    const filename = join(root, "src/main.rs");
    writeFileSync(filename, 'fn main() { println!("1"); }\n');
    const cargo = args => {
      const result = spawnSync("cargo", args, { cwd: root, encoding: "utf8",
        env: { ...process.env, CARGO_TARGET_DIR: join(root, "target"), CARGO_INCREMENTAL: "0" } });
      assert.equal(result.status, 0, result.stderr);
      return result.stdout;
    };
    cargo(["build", "--offline"]);
    const paths = ["Cargo.toml", "src/main.rs"], snapshot = captureSource(root, "same-commit", paths);
    const later = new Date(Date.now() + 10_000);
    for (const name of paths) utimesSync(join(root, name), later, later);
    assert.deepEqual(restoreSource(root, "same-commit", snapshot, paths), { restored: 2, changed: 0 });
    const artifacts = cargo(["build", "--offline", "--message-format=json"]).trim().split("\n").map(JSON.parse);
    assert(artifacts.some(row => row.reason === "compiler-artifact" && row.fresh));
    assert.deepEqual(restoreSource(root, "same-commit", {
      ...snapshot, files: [{ ...snapshot.files[1], mode: snapshot.files[1].mode ^ 0o111 }],
    }, paths), { restored: 0, changed: 1 });
    writeFileSync(filename, 'fn main() { println!("2"); }\n');
    utimesSync(filename, later, later);
    assert.deepEqual(restoreSource(root, "same-commit", snapshot, paths), { restored: 1, changed: 1 });
    assert.equal(cargo(["run", "--offline", "--quiet"]).trim(), "2");
    assert.equal(readFileSync(filename, "utf8"), 'fn main() { println!("2"); }\n');
    assert.equal(restoreSource(root, "different-commit", snapshot, paths).restored, 0);
    assert.equal(restoreSource(root, "same-commit", { ...snapshot, root: "other-checkout" }, paths).restored, 0);
    assert.throws(() => restoreSource(root, "same-commit", {
      ...snapshot, files: [{ ...snapshot.files[0], path: "../outside" }],
    }, paths));
  } finally { rmSync(root, { recursive: true, force: true }); }
});
