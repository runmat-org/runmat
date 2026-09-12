import assert from "node:assert/strict";
import test from "node:test";

import {
  gpuSpec, implementationProvenance, registrationManifestEntry, runtimeConstant,
} from "../compiled-runtime-schema.mjs";
import { parseCompiledInventory } from "../compiled-inventory.mjs";
import { contentDigest } from "../evidence.mjs";
import { compiledInventoryFixture } from "./helpers.mjs";

test("compiled provenance requires canonical Rust and repository paths", () => {
  const constant = {
    name: "pi", source_file: "crates/runmat-runtime/src/builtins/constants/mod.rs",
    module_path: "runmat_runtime::builtins::constants", builtin_path: "crate::builtins::constants",
  };
  assert.doesNotThrow(() => runtimeConstant(constant));
  assert.throws(() => runtimeConstant({ ...constant, source_file: "..\\constants.rs" }), /repository-relative path/);
  assert.throws(() => runtimeConstant({ ...constant, builtin_path: "not a path" }), /Rust module path/);
  assert.throws(() => runtimeConstant({ ...constant, builtin_path: "crate::builtins::other" }), /differs/);

  const provenance = {
    name: "foo", binding_variant: "default",
    source_file: "crates/runmat-runtime/src/builtins/foo.rs",
    module_path: "runmat_runtime::builtins::foo", function: "foo_builtin",
    builtin_path: "crate::builtins::foo", authority: "canonical_binding",
  };
  assert.doesNotThrow(() => implementationProvenance(provenance));
  assert.doesNotThrow(() => implementationProvenance({
    ...provenance,
    module_path: "runmat_runtime::builtins::foo::conversions",
  }));
  assert.throws(() => implementationProvenance({ ...provenance, function: "foo()" }), /Rust identifier/);
  assert.throws(() => implementationProvenance({
    ...provenance,
    module_path: "runmat_runtime::builtins::foobar",
  }), /differs/);
  assert.throws(() => implementationProvenance({
    ...provenance,
    module_path: "runmat_runtime::builtins::other::foo",
  }), /differs/);
});

test("typed registration and spec declarations reject widened provenance", () => {
  assert.doesNotThrow(() => registrationManifestEntry({
    kind: "gpu_spec", declaration: "FOO_GPU_SPEC", variant: null,
    builtin_path: "crate::builtins::foo",
  }));
  assert.doesNotThrow(() => registrationManifestEntry({
    kind: "builtin", declaration: "legacy_builtin", variant: null,
    builtin_path: "crate::builtins::legacy",
  }));
  assert.doesNotThrow(() => registrationManifestEntry({
    kind: "builtin", declaration: "canonical", variant: "default",
    builtin_path: "crate::builtins::canonical",
  }));
  assert.doesNotThrow(() => registrationManifestEntry({
    kind: "builtin", declaration: "dash_variant", variant: "-",
    builtin_path: "crate::builtins::dash_variant",
  }));
  assert.throws(() => registrationManifestEntry({
    kind: "constant", declaration: "pi", variant: "default",
    builtin_path: "crate::builtins::constants",
  }), /only builtin/);
  const spec = {
    key: "foo", declaration: "FOO_GPU_SPEC", builtin_path: "crate::builtins::foo",
    source_file: "crates/runmat-runtime/src/builtins/foo.rs", module_path: "runmat_runtime::builtins::foo",
    owner: { kind: "exact_builtin", identity: { name: "foo" } }, operation: "elementwise",
    supported_precisions: [], broadcast: "none", provider_hooks: [],
    constant_strategy: "inline_literal", residency: "inherit_inputs", nan_mode: "include",
    two_pass_threshold: null, workgroup_size: null, accepts_nan_mode: false, notes: "fixture",
  };
  assert.doesNotThrow(() => gpuSpec(spec));
  assert.throws(() => gpuSpec({ ...spec, declaration: "foo-spec" }), /Rust identifier/);
  assert.throws(() => gpuSpec({ ...spec, module_path: "runmat_runtime::builtins::other" }), /differs/);
});

test("compiled evidence rejects forged manifest rows even after digest recomposition", () => {
  const forged = compiledInventoryFixture();
  forged.snapshot.observed.registration_manifest.entries[0].builtin_path = "crate::builtins::other";
  reseal(forged);
  assert.throws(() => parseCompiledInventory(forged), /manifest differs/);

  const malformed = compiledInventoryFixture();
  malformed.snapshot.observed.implementation_provenance[0].source_file = "/tmp/foo.rs";
  reseal(malformed);
  assert.throws(() => parseCompiledInventory(malformed), /repository-relative path/);
});

test("compiled evidence accepts a provider declaration below its implementation owner", () => {
  const nested = compiledInventoryFixture();
  nested.snapshot.observed.gpu_specs.push({
    key: "foo", declaration: "FOO_GPU_SPEC",
    source_file: "crates/runmat-runtime/src/builtins/foo/specification.rs",
    module_path: "runmat_runtime::builtins::foo::specification",
    builtin_path: "crate::builtins::foo::specification",
    owner: { kind: "exact_builtin", identity: { name: "foo" } },
    operation: "elementwise", supported_precisions: [], broadcast: "none",
    provider_hooks: [], constant_strategy: "inline_literal", residency: "inherit_inputs",
    nan_mode: "include", two_pass_threshold: null, workgroup_size: null,
    accepts_nan_mode: false, notes: "fixture",
  });
  nested.snapshot.observed.registration_manifest.entries.push({
    kind: "gpu_spec", declaration: "FOO_GPU_SPEC", variant: null,
    builtin_path: "crate::builtins::foo::specification",
  });
  nested.snapshot.observed.registration_manifest.entries.sort((left, right) =>
    `${left.kind}\0${left.declaration}\0${left.variant ?? ""}\0${left.builtin_path}`
      .localeCompare(`${right.kind}\0${right.declaration}\0${right.variant ?? ""}\0${right.builtin_path}`, "en", { sensitivity: "variant" }));
  nested.snapshot.observed.registration_manifest.counts.gpu_spec = 1;
  reseal(nested);
  assert.doesNotThrow(() => parseCompiledInventory(nested));
});

function reseal(value) {
  value.snapshot.observed.registration_manifest.digest = contentDigest(Buffer.from(JSON.stringify(
    value.snapshot.observed.registration_manifest.entries,
  ))).slice("sha256:".length);
  value.digest.value = contentDigest(Buffer.from(JSON.stringify(value.snapshot))).slice("sha256:".length);
}
