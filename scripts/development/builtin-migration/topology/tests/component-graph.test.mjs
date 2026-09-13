import assert from "node:assert/strict";
import test from "node:test";
import { identityAtomicAuthorityPaths } from "../../baseline-evidence.mjs";
import { buildAuthorityComponentGraph } from "../components.mjs";
import { parseAuthorityComponentGraph } from "../component-graph.mjs";

function inventory() {
  return {
    digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    identities: [
      row("alpha", ["runtime/shared.rs"]),
      row("beta", ["runtime/shared.rs"]),
      row("gamma", ["runtime/gamma.rs"]),
      row("constant_alpha", [], ["runtime/constants.rs"]),
      row("constant_beta", [], ["runtime/constants.rs"]),
    ],
  };
}

function row(identity, runtime, runtimeConstants = []) {
  return {
    identity,
    domain: "test",
    family: "core",
    ownership: { catalog: [], runtime },
    semantic_authority: {
      catalog_entries: [],
      legacy_functions: [],
      runtime_constants: runtimeConstants.map((source_file) => ({ source_file })),
    },
    registrations: { runtime: [] },
  };
}

test("component graph binds callable and constant implementation owners atomically", () => {
  const baseline = inventory();
  const graph = buildAuthorityComponentGraph(baseline);
  const parsed = parseAuthorityComponentGraph(graph, baseline);
  assert.deepEqual([...parsed.index], [
    ["component-alpha", ["alpha", "beta"]],
    ["component-constant_alpha", ["constant_alpha", "constant_beta"]],
    ["component-gamma", ["gamma"]],
  ]);
  assert.deepEqual(
    graph.candidates.find((candidate) => candidate.candidate_id === "component-constant_alpha")
      .shared_sources,
    ["runtime/constants.rs"],
  );
  assert.equal(parsed.digest.startsWith("sha256:"), true);
});

test("component graph parser rejects relabeled or edited evidence", () => {
  const baseline = inventory();
  const graph = buildAuthorityComponentGraph(baseline);
  assert.throws(
    () => parseAuthorityComponentGraph({ ...graph, baseline_inventory_digest: "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" }, baseline),
    /differs from the deterministic reviewed-baseline projection/,
  );
  const edited = structuredClone(graph);
  edited.candidates[0].identities.pop();
  assert.throws(() => parseAuthorityComponentGraph(edited, baseline), /differs from the deterministic reviewed-baseline projection/);
});

test("atomic authority paths include writable identity authority but exclude tests and integration products", () => {
  const value = row("alpha", ["atomic/05-runtime.rs"]);
  value.ownership = {
    catalog: ["atomic/01-catalog.rs"],
    catalog_documentation: ["atomic/07-catalog-documentation.rs"],
    runtime: ["atomic/05-runtime.rs"],
    sidecars: ["atomic/08-sidecar.json"],
    runtime_documentation_shadows: ["atomic/09-runtime-documentation.rs"],
  };
  value.semantic_authority = {
    catalog_aliases: [{ provenance: { source_file: "atomic/02-alias.rs" } }],
    constants: [{ provenance: { source_file: "atomic/03-catalog-constant.rs" } }],
    catalog_provenance: [{ provenance: { source_file: "atomic/04-catalog-provenance.rs" } }],
    implementation_provenance: [{ source_file: "atomic/06-implementation.rs" }],
    runtime_constants: [{ source_file: "atomic/12-runtime-constant.rs" }],
    gpu_specs: [{ source_file: "atomic/19-canonical-provider.rs" }],
    fusion_specs: [{ source_file: "atomic/20-canonical-fusion.rs" }],
  };
  value.documentation = { sources: [{ path: "atomic/10-documentation.md" }] };
  value.tests = { paths: ["excluded/test.rs"] };
  value.registrations = {
    runtime: [{ path: "atomic/11-registration.rs" }],
    native_link: {
      catalog_contract_paths: ["atomic/13-native-catalog.rs"],
      runtime_binding_inputs: [{ path: "atomic/14-native-runtime.rs" }],
    },
  };
  value.dependencies = {
    legacy_resolver_paths: ["atomic/15-legacy-resolver.rs"],
    catalog_resolver_paths: ["atomic/16-catalog-resolver.rs"],
    generated_registry: ["excluded/generated.rs"],
  };
  value.provider = {
    gpu_or_wgpu_paths: ["atomic/17-provider.rs"],
    fusion_paths: ["atomic/18-fusion.rs"],
  };
  assert.deepEqual(identityAtomicAuthorityPaths(value), [
    "atomic/01-catalog.rs",
    "atomic/02-alias.rs",
    "atomic/03-catalog-constant.rs",
    "atomic/04-catalog-provenance.rs",
    "atomic/05-runtime.rs",
    "atomic/06-implementation.rs",
    "atomic/07-catalog-documentation.rs",
    "atomic/08-sidecar.json",
    "atomic/09-runtime-documentation.rs",
    "atomic/10-documentation.md",
    "atomic/11-registration.rs",
    "atomic/12-runtime-constant.rs",
    "atomic/13-native-catalog.rs",
    "atomic/14-native-runtime.rs",
    "atomic/15-legacy-resolver.rs",
    "atomic/16-catalog-resolver.rs",
    "atomic/17-provider.rs",
    "atomic/18-fusion.rs",
    "atomic/19-canonical-provider.rs",
    "atomic/20-canonical-fusion.rs",
  ]);
});

test("component graph joins cross-kind owners by physical path and ignores tests and registries", () => {
  const runtimeOwner = row("alpha", ["shared/cross-kind.rs"]);
  runtimeOwner.tests = { paths: ["shared/test.rs"] };
  runtimeOwner.dependencies = { generated_registry: ["shared/generated.rs"] };
  const providerOwner = row("beta", []);
  providerOwner.provider = { gpu_or_wgpu_paths: ["shared/cross-kind.rs"], fusion_paths: [] };
  providerOwner.tests = { paths: ["shared/test.rs"] };
  providerOwner.dependencies = { generated_registry: ["shared/generated.rs"] };
  const unrelated = row("gamma", ["runtime/gamma.rs"]);
  unrelated.tests = { paths: ["shared/test.rs"] };
  unrelated.dependencies = { generated_registry: ["shared/generated.rs"] };
  const graph = buildAuthorityComponentGraph({
    digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    identities: [runtimeOwner, providerOwner, unrelated],
  });
  assert.deepEqual(graph.candidates.map((candidate) => candidate.identities), [
    ["alpha", "beta"],
    ["gamma"],
  ]);
  assert.deepEqual(graph.candidates[0].shared_sources, ["shared/cross-kind.rs"]);
});
