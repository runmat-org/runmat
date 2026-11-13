// Modeled parser fixtures only, never compiled Rust evidence.
import assert from "node:assert/strict";
import test from "node:test";
import { catalogEntry } from "../compiled-schema.mjs";
import { gpuSpec } from "../compiled-runtime-schema.mjs";
import { parseCompiledInventory } from "../compiled-inventory.mjs";
import { contentDigest } from "../evidence.mjs";
import { compiledInventoryFixture } from "./helpers.mjs";

function renewDigest(fixture) {
  fixture.digest.value = contentDigest(Buffer.from(JSON.stringify(fixture.snapshot))).slice("sha256:".length);
  return fixture;
}
const entryFixture = () => compiledInventoryFixture().snapshot.declared.catalog_entries[0];

test("v4 inventory requires exactly Rust catalog v10, rejecting stale and future versions", () => {
  const current = compiledInventoryFixture();
  assert.equal(current.schema_version, 4);
  assert.equal(current.snapshot.declared.catalog_schema_version, 10);
  assert.doesNotThrow(() => parseCompiledInventory(current));
  for (const version of [0, 7, 8, 9, 11, "10", null]) {
    const fixture = structuredClone(current);
    fixture.snapshot.declared.catalog_schema_version = version;
    assert.throws(() => parseCompiledInventory(renewDigest(fixture)), /catalog schema version must be 10/);
  }
});

test("binding worker policy is mandatory, closed and retained by parser roundtrip", () => {
  for (const policy of ["Forbidden", "PortableHostCpu"]) {
    const fixture = compiledInventoryFixture();
    fixture.snapshot.declared.catalog_entries[0].bindings[0].owned_worker = policy;
    const parsed = parseCompiledInventory(renewDigest(fixture));
    assert.equal(parsed.snapshot.declared.catalog_entries[0].bindings[0].owned_worker, policy);
  }
  for (const mutation of [
    binding => { delete binding.owned_worker; },
    binding => { binding.owned_worker = null; },
    binding => { binding.owned_worker = true; },
    binding => { binding.owned_worker = "portable_host_cpu"; },
    binding => { binding.owned_worker = "Guessed"; },
    binding => { binding.owned_worker = { PortableHostCpu: {} }; },
    binding => { binding.worker_purity = "Pure"; },
  ]) {
    const fixture = compiledInventoryFixture();
    mutation(fixture.snapshot.declared.catalog_entries[0].bindings[0]);
    assert.throws(() => parseCompiledInventory(renewDigest(fixture)), /binding declaration|owned worker policy/);
  }
});

test("identity inference exposes only its mandatory structural-dimension capability", () => {
  const digests = [];
  for (const flag of [false, true]) {
    const fixture = compiledInventoryFixture();
    fixture.snapshot.declared.catalog_entries[0].contract.inference_rule.Identity.structural_dimension_constructor = flag;
    const parsed = parseCompiledInventory(renewDigest(fixture));
    assert.equal(parsed.snapshot.declared.catalog_entries[0].contract.inference_rule.Identity.structural_dimension_constructor, flag);
    digests.push(parsed.digest);
  }
  assert.notEqual(digests[0], digests[1]);
  for (const identity of [{}, { structural_dimension_constructor: null }, { structural_dimension_constructor: 1 },
    { structural_dimension_constructor: "false" }, { structural_dimension_constructor: false, handler: "pointer" },
    { structural_dimension_constructor: false, fusion_fact: "pointer" }]) {
    const entry = entryFixture();
    entry.contract.inference_rule = { Identity: identity };
    assert.throws(() => catalogEntry(entry), /identity inference capability|structural dimension constructor/);
  }
});

test("closed current inference and placement variants do not admit invented variants", () => {
  const entry = entryFixture();
  for (const rule of [{ Math: "BesselJ" }, { Math: { NumericLimit: { Floating: "SpacingAtOne" } } }, { Logical: { MetadataPredicate: "Struct" } }]) {
    entry.contract.inference_rule = rule;
    assert.doesNotThrow(() => catalogEntry(entry));
  }
  for (const rule of [{ Aggregate: "Struct" }, { Math: "BesselY" }, { Math: { NumericLimit: { Floating: "Spacing" } } }]) {
    entry.contract.inference_rule = rule;
    assert.throws(() => catalogEntry(entry), /inference rule|unsupported variant/);
  }
  entry.contract.inference_rule = { Identity: { structural_dimension_constructor: false } };
  for (const transform of ["Flip", "Rotate90", "CircularShift", "ShiftDimensions", "Diagonal", "Kronecker", "BlockDiagonal", "RepeatElements", { Triangular: "Lower" }, { Triangular: "Upper" }]) {
    entry.contract.semantic_kind = { ShapeTransform: transform };
    assert.doesNotThrow(() => catalogEntry(entry));
  }
  entry.contract.semantic_kind = { ShapeTransform: { Triangular: "Diagonal" } };
  assert.throws(() => catalogEntry(entry), /semantic kind/);
  entry.contract.semantic_kind = "General";
  entry.placement.distributed = "MapUnaryOrScalarLikePrototype";
  assert.doesNotThrow(() => catalogEntry(entry));
  entry.placement.distributed = "MapEverything";
  assert.throws(() => catalogEntry(entry), /distributed policy/);
});

test("producer errors, preparation limits and u32 geometry keep exact shapes", () => {
  const entry = entryFixture();
  entry.descriptor.errors = [{ code: "RM.FIXTURE.ERROR", identifier: null, when: "Invalid fixture input.", message: "fixture error" }];
  assert.doesNotThrow(() => catalogEntry(entry));
  for (const mutate of [error => { error.cause = "guessed"; }, error => { delete error.identifier; }, error => { error.identifier = 1; }]) {
    const invalid = structuredClone(entry);
    mutate(invalid.descriptor.errors[0]);
    assert.throws(() => catalogEntry(invalid), /error descriptor|error identifier/);
  }
  entry.placement.preparation = { Forms: [{ id: 1, binding_variant: "default", signatures: [0], data_arguments: [0], operation: { FloatingCast: "Int32" }, provider_requirement: "F64" }] };
  assert.throws(() => catalogEntry(entry), /floating cast class/);
  entry.placement.preparation.Forms[0].operation = { ElementwiseNary: { data_arity: 1 } };
  assert.throws(() => catalogEntry(entry), /at least two/);
  const spec = compiledInventoryFixture("foo", { legacyGroup: true }).snapshot.observed.gpu_specs[0];
  for (const value of [null, 0, 0xffff_ffff]) { spec.workgroup_size = value; assert.doesNotThrow(() => gpuSpec(spec)); }
  spec.workgroup_size = 0x1_0000_0000;
  assert.throws(() => gpuSpec(spec), /fit u32/);
  spec.workgroup_size = null;
  assert.throws(() => gpuSpec({ ...spec, guessed_geometry: 64 }), /fields must be exactly/);
});
