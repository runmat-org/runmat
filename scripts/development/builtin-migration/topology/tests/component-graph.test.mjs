import assert from "node:assert/strict";
import test from "node:test";
import { buildAuthorityComponentGraph } from "../components.mjs";
import { parseAuthorityComponentGraph } from "../component-graph.mjs";

function inventory() {
  return {
    digest: "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    identities: [
      row("alpha", ["runtime/shared.rs"]),
      row("beta", ["runtime/shared.rs"]),
      row("gamma", ["runtime/gamma.rs"]),
    ],
  };
}

function row(identity, runtime) {
  return {
    identity,
    domain: "test",
    family: "core",
    ownership: { catalog: [], runtime },
    semantic_authority: { catalog_entries: [], legacy_functions: [] },
    registrations: { runtime: [] },
  };
}

test("component graph parser accepts only the deterministic inventory projection", () => {
  const baseline = inventory();
  const graph = buildAuthorityComponentGraph(baseline);
  const parsed = parseAuthorityComponentGraph(graph, baseline);
  assert.deepEqual([...parsed.index], [
    ["component-alpha", ["alpha", "beta"]],
    ["component-gamma", ["gamma"]],
  ]);
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
