import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { BASELINE, CI_RUN, validateDelta, validateRun, requiredJobs } from "./release-ci-checkpoint.mjs";

const ci = readFileSync(new URL("../../.github/workflows/ci.yml", import.meta.url), "utf8");
const names = requiredJobs(ci);
const run = () => ({ id: CI_RUN, head_sha: BASELINE, head_repository: { full_name: "runmat-org/runmat" },
  head_branch: "main", event: "push", path: ".github/workflows/ci.yml", status: "completed", conclusion: "success" });
const jobs = () => names.map(name => ({ name, status: "completed", conclusion: "success" }));
const changed = path => ({ path, beforeMode: "100644", afterMode: "100644" });

test("requires the complete successful cross-platform checkpoint", () => {
  assert.equal(names.length, 26);
  validateRun(run(), jobs(), names);
  for (const change of [{ status: "in_progress" }, { conclusion: "failure" }, { head_sha: "a".repeat(40) },
    { head_branch: "dev" }, { event: "pull_request" }, { head_repository: { full_name: "other/runmat" } },
    { path: ".github/workflows/other.yml" }, { id: CI_RUN + 1 }]) {
    assert.throws(() => validateRun({ ...run(), ...change }, jobs(), names));
  }
});
test("rejects missing, duplicate, skipped and failed jobs", () => {
  assert.throws(() => validateRun(run(), jobs().slice(1), names));
  assert.throws(() => validateRun(run(), [...jobs().slice(1), jobs()[1]], names));
  for (const conclusion of ["failure", "skipped", "cancelled", null]) {
    const input = jobs(); input[0].conclusion = conclusion;
    assert.throws(() => validateRun(run(), input, names));
  }
});
test("accepts only the exact publish=false removal from required manifests", () => {
  const file = "crates/runmat-design-tools/Cargo.toml";
  const original = '[package]\nname = "runmat-design-tools"\npublish = false\n';
  validateDelta([changed(file)], () => original, () => original.replace("publish = false\n", ""));
  for (const after of [original + 'version = "0.7.2"\n', original.replace("false", "true")]) {
    assert.throws(() => validateDelta([changed(file)], () => original, () => after));
  }
});
test("rejects runtime, lockfile, compiler, fixture and unknown changes", () => {
  for (const path of ["Cargo.lock", "Cargo.toml", "rust-toolchain.toml", "crates/runmat-runtime/src/lib.rs",
    "scripts/build-runmat-with-aot-runtime.sh", "verification/meshing/input.json", "docs/CHANGELOG.md"]) {
    assert.throws(() => validateDelta([changed(path)], () => "", () => ""));
  }
  assert.throws(() => validateDelta([{ ...changed("scripts/publish-crates.mjs"), afterMode: "120000" }], () => "", () => ""));
  assert.throws(() => validateDelta([{ ...changed("scripts/publish-crates.mjs"), afterMode: null }], () => "", () => ""));
  validateDelta([{ ...changed("scripts/verify-release-version.mjs"), beforeMode: "100755", afterMode: "100755" }], () => "", () => "");
  assert.throws(() => validateDelta([{ ...changed("scripts/verify-release-version.mjs"), beforeMode: "100755" }], () => "", () => ""));
});
test("ordinary CI can only add the two publication test files", () => {
  const file = ".github/workflows/ci.yml", base = "node --test\n          old.test.mjs\n";
  const after = base + "          scripts/publish-crates.test.mjs\n          scripts/distribution/release-ci-checkpoint.test.mjs\n";
  validateDelta([changed(file)], () => base, () => after);
  assert.throws(() => validateDelta([changed(file)], () => base, () => after.replace("old.test.mjs", "different.test.mjs")));
});
test("release publication changes preserve the complete native build job", () => {
  const file = ".github/workflows/release.yml";
  const base = "  build-release:\n    command: original\n  create-release:\n    name: Release\n";
  validateDelta([changed(file)], () => base, () => base + "  publish-crates:\n    runs-on: ubuntu-latest\n");
  assert.throws(() => validateDelta([changed(file)], () => base, () => base.replace("command: original", "command: changed")));
});
