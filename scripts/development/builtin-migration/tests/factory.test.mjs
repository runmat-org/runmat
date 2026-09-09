import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { auditInventory, parseBatch } from "../audit.mjs";
import { compareCodePoint } from "../constants.mjs";
import { buildDispositionSeed, buildInventory, validateDispositionInput } from "../inventory.mjs";
import { prepareIdentity } from "../prepare.mjs";
import { buildQueue } from "../queue.mjs";

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-migration-factory-"));
  const write = (relative, contents) => {
    const target = path.join(root, relative);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, contents);
  };
  write("crates/runmat-runtime/src/builtins/math/basic/foo.rs", `
#[runtime_builtin(
  name = "Foo",
  category = "math/basic",
  type_resolver(foo_type),
  builtin_path = "crate::builtins::math::basic::foo"
)]
async fn foo_builtin() {}
#[cfg(test)] mod tests {}
#[runmat_macros::register_gpu_spec(builtin_path = "foo")] const GPU_SPEC: () = ();
`);
  write("crates/runmat-runtime/src/builtins/io/files/hidden.rs", `
#[runtime_builtin(name = "hidden", category = "io/files")]
fn hidden_builtin() { let _ = std::fs::read("x"); }
`);
  write("crates/runmat-runtime/src/builtins/generated_wasm_registry.rs", `
crate :: builtins :: math :: basic :: foo :: __runmat_wasm_register_builtin_foo_builtin();
`);
  write("crates/runmat-builtins/src/catalog/entries/math/basic/foo/mod.rs", `
pub const ENTRY: BuiltinCatalogEntry = BuiltinCatalogEntry {
 identity: BuiltinCatalogIdentity { name: "foo" },
 link: BuiltinLinkContract {},
 contract: BuiltinContractDeclaration { inference_rule: BuiltinInferenceRule::Math(Rule::Foo) },
};
`);
  write("docs/builtins/reference/foo.json", JSON.stringify({ name: "foo", summary: "Foo", examples: [{ input: "foo(1)" }] }));
  write("crates/runmat-runtime/src/builtins/builtins-json/foo.json", JSON.stringify({ name: "foo", summary: "old" }));
  return root;
}

function disposition(identities) {
  return { schema_version: 1, kind: "runmat-builtin-dispositions", identities };
}

function reviewed(values) {
  return { review: { status: "reviewed", evidence: ["ticket review"] }, ...values };
}

test("the shared comparator uses explicit code-point order", () => {
  assert.deepEqual(["a", "Z", "A", "z"].sort(compareCodePoint), ["A", "Z", "a", "z"]);
});

test("inventory joins source surfaces without inferring unresolved intent", () => {
  const inventory = buildInventory(fixture(), disposition({ hidden: reviewed({ disposition: "internal", reason: "reviewed helper" }) }));
  assert.deepEqual(inventory.identities.map((entry) => entry.identity), ["foo", "hidden"]);
  assert.equal(inventory.summary.runtime_binding_records, 2);
  assert.deepEqual(inventory.summary.runtime_binding_provenance, { "literal-attribute": 2 });
  const foo = inventory.identities[0];
  assert.equal(foo.disposition.kind, "canonical");
  assert.equal(foo.disposition.source, "catalog-entry");
  assert.equal(foo.domain, "math");
  assert.equal(foo.family, "basic");
  assert.equal(foo.registrations.runtime[0].function, "foo_builtin");
  assert.deepEqual(foo.registrations.wasm, ["foo_builtin"]);
  assert.equal(foo.registrations.native_link.runtime_binding_inputs[0].builtin_path, "crate::builtins::math::basic::foo");
  assert.equal(foo.provider.gpu_or_wgpu_paths.length, 1);
  assert.equal(foo.tests.strength, "medium");
  assert.equal(foo.ownership.runtime_documentation_shadows.length, 1);
  assert.equal(foo.examples.discovered_count, 1);
  assert.deepEqual(foo.unresolved, ["case-spelling-conflict"]);
  const hidden = inventory.identities[1];
  assert.equal(hidden.disposition.kind, "internal");
  assert.deepEqual(hidden.host_capability_hints.map((hint) => hint.kind), ["filesystem"]);
});

test("queue is deterministic, weighted, and exposes collisions", () => {
  const inventory = buildInventory(fixture());
  const first = buildQueue(inventory);
  const second = buildQueue(inventory);
  assert.deepEqual(first, second);
  for (let index = 1; index < first.rows.length; index += 1) {
    assert.ok(first.rows[index - 1].complexity.score >= first.rows[index].complexity.score);
  }
  const hidden = first.rows.find((row) => row.identity === "hidden");
  assert.equal(hidden.migration_state, "classification-required");
  assert.ok(hidden.complexity.evidence.some((entry) => entry.factor === "unresolved-fields"));
  const foo = first.rows.find((row) => row.identity === "foo");
  assert.ok(foo.write_set_collision_keys.includes("family:math/basic"));
  assert.ok(foo.write_set_collision_keys.includes("generated-registry:wasm"));
  assert.equal(first.summary.identities, 2);
});

test("reviewed alias and internal inputs require their semantic evidence", () => {
  assert.throws(
    () => validateDispositionInput(disposition({ old: reviewed({ disposition: "alias" }) })),
    /aliases require a canonical target/,
  );
  assert.throws(
    () => validateDispositionInput(disposition({ helper: reviewed({ disposition: "internal" }) })),
    /require a reviewed reason/,
  );
});

test("inventory diagnoses dangling and contradictory reviewed dispositions", () => {
  const inventory = buildInventory(fixture(), disposition({
    foo: reviewed({ disposition: "alias", canonical: "missing" }),
  }));
  assert.deepEqual(inventory.diagnostics.map((entry) => entry.code), [
    "dangling-alias",
    "disposition-contradicts-catalog",
  ]);
});

test("seed format is explicit and accepts only unreviewed empty rows", () => {
  const seed = buildDispositionSeed(buildInventory(fixture()));
  validateDispositionInput(seed);
  assert.equal(seed.identities.foo.review.status, "unreviewed");
  assert.equal(seed.identities.foo.disposition, null);
  seed.identities.foo.domain = "math";
  assert.throws(() => validateDispositionInput(seed), /unreviewed seed rows/);
});

test("strict audit reports legacy debt and passes a reviewed internal identity", () => {
  const inventory = buildInventory(fixture(), disposition({ hidden: reviewed({ disposition: "internal", reason: "reviewed helper" }) }));
  const report = auditInventory(inventory, ["foo", "hidden"], { source: "git:test", artifact: "fixture-audit" });
  assert.equal(report.schema_version, 2);
  assert.equal(report.metadata.source, "git:test");
  assert.match(report.metadata.inventory.digest, /^sha256:[a-f0-9]{64}$/);
  assert.equal(report.result, "fail");
  assert.equal(report.identities.find((entry) => entry.identity === "hidden").result, "pass");
  const fooCodes = report.identities.find((entry) => entry.identity === "foo").failures.map((entry) => entry.code);
  assert.ok(fooCodes.includes("legacy-sidecar-count"));
  assert.ok(fooCodes.includes("runtime-shadow-count"));
  assert.throws(() => parseBatch({ schema_version: 1, kind: "runmat-builtin-migration-batch", identities: ["foo", "Foo"] }), /unique ignoring case/);
});

test("prepare creates an isolated deterministic review workspace", () => {
  const repository = fixture();
  const inventory = buildInventory(repository);
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-migration-review-"));
  const first = prepareIdentity(repository, inventory, "foo", output);
  const copied = path.join(first.workspace, "legacy-documents", "sidecar-foo.json");
  assert.equal(fs.readFileSync(copied, "utf8"), fs.readFileSync(path.join(repository, "docs/builtins/reference/foo.json"), "utf8"));
  const firstResult = fs.readFileSync(path.join(first.workspace, "prepare-result.json"), "utf8");
  prepareIdentity(repository, inventory, "foo", output);
  assert.equal(fs.readFileSync(path.join(first.workspace, "prepare-result.json"), "utf8"), firstResult);
  fs.writeFileSync(path.join(first.workspace, "catalog", "mod.rs.template"), "review edit\n");
  assert.throws(() => prepareIdentity(repository, inventory, "foo", output), /refuses to overwrite modified review file/);
  assert.throws(() => prepareIdentity(repository, inventory, "foo", path.join(repository, "generated")), /outside the canonical repository path/);
});

test("prepare rejects an output-root symlink into the repository", () => {
  const repository = fixture();
  const inventory = buildInventory(repository);
  const outside = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-migration-output-link-"));
  const linkedOutput = path.join(outside, "review");
  fs.symlinkSync(repository, linkedOutput, "dir");
  const sourceBefore = fs.readFileSync(path.join(repository, "docs/builtins/reference/foo.json"), "utf8");
  assert.throws(() => prepareIdentity(repository, inventory, "foo", linkedOutput), /canonical repository path/);
  assert.equal(fs.readFileSync(path.join(repository, "docs/builtins/reference/foo.json"), "utf8"), sourceBefore);
  assert.equal(fs.existsSync(path.join(repository, "foo")), false);
});

test("prepare rejects a pre-existing identity-workspace symlink into the repository", () => {
  const repository = fixture();
  const inventory = buildInventory(repository);
  const output = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-migration-workspace-link-"));
  fs.symlinkSync(repository, path.join(output, "foo"), "dir");
  const sourceBefore = fs.readFileSync(path.join(repository, "docs/builtins/reference/foo.json"), "utf8");
  assert.throws(() => prepareIdentity(repository, inventory, "foo", output), /canonical repository path/);
  assert.equal(fs.readFileSync(path.join(repository, "docs/builtins/reference/foo.json"), "utf8"), sourceBefore);
  assert.equal(fs.existsSync(path.join(repository, "catalog")), false);
});

test("case variants join while preserving contradictory spelling evidence", () => {
  const inventory = buildInventory(fixture());
  const foo = inventory.identities.find((entry) => entry.identity === "foo");
  assert.deepEqual(foo.spellings, ["Foo", "foo"]);
  assert.ok(foo.unresolved.includes("case-spelling-conflict"));
});
