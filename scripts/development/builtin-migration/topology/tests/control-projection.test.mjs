import assert from "node:assert/strict";
import test from "node:test";

import { materializeTopologyControl } from "../control-projection.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;

test("materializes topology-owned facts with control-only execution policy", () => {
  const { topology, overlay } = fixture();
  const result = materializeTopologyControl(topology, overlay);
  assert.deepEqual(result.bundles.get("c01-math-basic"), {
    id: "c01-math-basic",
    identities: ["foo"],
    atomic_reason: "One frozen authority component",
    prerequisites: [],
    integration_product_refs: [],
    module_composition_transition: null,
    integration_outputs: [],
    authored_write_set: [
      { kind: "file", path: "crates/runmat-runtime/src/builtins/math/basic/foo.rs" },
      { kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic" },
    ],
  });
  assert.deepEqual(result.identities.get("foo"), {
    identity: "foo",
    cohort: "C01",
    bundle_id: "c01-math-basic",
    domain: "math",
    family: "basic",
    public_identity: {
      kind: "primary", primary_spelling: { identity: "foo", spelling: "foo" },
    },
  });
});

test("materializes typed public identities that agree with topology", () => {
  const { topology, overlay } = fixture({ dispositions: true });
  const result = materializeTopologyControl(topology, overlay);
  assert.deepEqual(result.identities.get("foo").public_identity, {
    kind: "primary", primary_spelling: { identity: "foo", spelling: "foo" },
  });
  assert.deepEqual(result.identities.get("foalias").public_identity, {
    kind: "alias", alias_spelling: { identity: "foalias", spelling: "foalias" },
    canonical_identity: "foo",
  });
  assert.deepEqual(result.identities.get("__helper").public_identity, {
    kind: "internal",
    reason: "Runtime-only helper",
    evidence: ["reviewed-topology:__helper"],
  });
});

test("rejects topology drift and incomplete overlay key sets", () => {
  const digestMismatch = fixture();
  digestMismatch.overlay.topology_digest = `sha256:${"b".repeat(64)}`;
  assert.throws(() => materializeTopologyControl(digestMismatch.topology, digestMismatch.overlay), /digest does not match/);

  const missingBundle = fixture();
  missingBundle.overlay.bundleControls.delete("c01-math-basic");
  assert.throws(() => materializeTopologyControl(missingBundle.topology, missingBundle.overlay), /bundle policy set differs/);

  const missingIdentity = fixture();
  missingIdentity.overlay.identityControls.delete("foo");
  assert.throws(() => materializeTopologyControl(missingIdentity.topology, missingIdentity.overlay), /identity policy set differs/);
});

test("rejects invalid topology dispositions instead of accepting an overlay substitute", () => {
  const { topology, overlay } = fixture();
  topology.identities.get("foo").disposition = { kind: "alias", canonical: null };
  assert.throws(() => materializeTopologyControl(topology, overlay), /differs from topology disposition/);
});

function fixture({ dispositions = false } = {}) {
  const identityRows = [["foo", topologyIdentity("foo", { kind: "canonical", canonical: null, reason: null, source: "reviewed-input" })]];
  if (dispositions) {
    identityRows.push(["foalias", topologyIdentity("foalias", { kind: "alias", canonical: "foo", reason: null, source: "reviewed-input" })]);
    identityRows.push(["__helper", topologyIdentity("__helper", { kind: "internal", canonical: null, reason: "Runtime-only helper", source: "reviewed-input" })]);
    identityRows.sort(([left], [right]) => left.localeCompare(right));
  }
  const identities = identityRows.map(([id]) => id);
  const topology = {
    digest: DIGEST,
    bundles: new Map([["c01-math-basic", {
      id: "c01-math-basic",
      cohort: "C01",
      authority_components: ["component-foo"],
      identities,
      atomic_reason: "One frozen authority component",
      composition: {
        kind: "single-component",
        target_packages: [{ domain: "math", family: "basic" }],
        authored_write_set: [{ kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic" }],
        shared_authority_sources: [],
        evidence: ["fixture topology review"],
      },
    }]]),
    identities: new Map(identityRows),
  };
  const overlay = {
    topology_digest: DIGEST,
    bundleControls: new Map([["c01-math-basic", {
      prerequisites: [],
      additional_authored_write_set: [{ kind: "file", path: "crates/runmat-runtime/src/builtins/math/basic/foo.rs" }],
      integration_product_refs: [],
      module_composition_transition: null,
    }]]),
    identityControls: new Map(identityRows.map(([id, row]) => [id, {
      public_identity: row.disposition.kind === "canonical"
        ? { kind: "primary", primary_spelling: { identity: id, spelling: id } }
        : row.disposition.kind === "alias"
          ? { kind: "alias", alias_spelling: { identity: id, spelling: id }, canonical_identity: row.disposition.canonical }
          : {
            kind: "internal", reason: row.disposition.reason,
            evidence: [`reviewed-topology:${id}`],
          },
    }])),
    integrationProducts: new Map(),
  };
  return { topology, overlay };
}

function topologyIdentity(identity, disposition) {
  return {
    identity,
    bundle_id: "c01-math-basic",
    cohort: "C01",
    domain: "math",
    family: "basic",
    disposition,
  };
}
