import assert from "node:assert/strict";
import test from "node:test";

import { validateTopologyControlProjection } from "../control-projection.mjs";

const DIGEST = `sha256:${"a".repeat(64)}`;

test("accepts an exact single-family projection with control-only execution fields", () => {
  const { topology, control } = fixture();
  assert.equal(validateTopologyControlProjection(topology, control), control);
});

test("accepts a family bundle with a shared authored package write set", () => {
  const { topology, control } = fixture();
  control.bundles.get("c01-math-basic").authored_write_set = [
    { kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic" },
  ];
  assert.doesNotThrow(() => validateTopologyControlProjection(topology, control));
});

test("accepts an atomic component whose identities have different target families", () => {
  const { topology, control } = fixture({ multiFamily: true });
  assert.doesNotThrow(() => validateTopologyControlProjection(topology, control));
});

test("rejects a topology digest mismatch", () => {
  const { topology, control } = fixture();
  control.topology_digest = `sha256:${"b".repeat(64)}`;
  assert.throws(() => validateTopologyControlProjection(topology, control), /topology digest does not match/);
});

test("rejects added, removed, or renamed bundles", () => {
  const { topology, control } = fixture();
  control.bundles.set("c01-extra", control.bundles.get("c01-math-basic"));
  assert.throws(() => validateTopologyControlProjection(topology, control), /bundle set differs/);
});

test("rejects component regrouping", () => {
  const { topology, control } = fixture();
  control.bundles.get("c01-math-basic").authority_components = ["component-bar"];
  assert.throws(() => validateTopologyControlProjection(topology, control), /authority components differ/);
});

test("rejects identity regrouping", () => {
  const { topology, control } = fixture({ multiFamily: true });
  control.bundles.get("c01-math-basic").identities = ["foo"];
  assert.throws(() => validateTopologyControlProjection(topology, control), /control identities differ/);
});

test("rejects bundle cohort and atomic-reason changes", () => {
  const cohort = fixture();
  cohort.control.bundles.get("c01-math-basic").cohort = "C02";
  assert.throws(() => validateTopologyControlProjection(cohort.topology, cohort.control), /control cohort differs/);

  const reason = fixture();
  reason.control.bundles.get("c01-math-basic").atomic_reason = "A different reason";
  assert.throws(() => validateTopologyControlProjection(reason.topology, reason.control), /atomic reason differs/);
});

test("rejects identity set, bundle, cohort, domain, family, and disposition changes", () => {
  const removed = fixture();
  removed.control.identities.delete("foo");
  assert.throws(() => validateTopologyControlProjection(removed.topology, removed.control), /identity set differs/);

  for (const [field, value, pattern] of [
    ["bundle_id", "c01-other", /control bundle differs/],
    ["cohort", "C02", /control cohort differs/],
    ["domain", "logical", /control domain differs/],
    ["family", "tests", /control family differs/],
  ]) {
    const changed = fixture();
    changed.control.identities.get("foo")[field] = value;
    assert.throws(() => validateTopologyControlProjection(changed.topology, changed.control), pattern);
  }

  const disposition = fixture();
  disposition.control.identities.get("foo").disposition = { kind: "alias", target: "bar" };
  assert.throws(() => validateTopologyControlProjection(disposition.topology, disposition.control), /control disposition differs/);
});

test("rejects noncanonical or duplicate component and identity membership", () => {
  const components = fixture();
  components.control.bundles.get("c01-math-basic").authority_components = ["component-foo", "component-foo"];
  assert.throws(() => validateTopologyControlProjection(components.topology, components.control), /authority components must be unique/);

  const identities = fixture({ multiFamily: true });
  identities.control.bundles.get("c01-math-basic").identities = ["foo", "bar"];
  assert.throws(() => validateTopologyControlProjection(identities.topology, identities.control), /control identities must use canonical order/);

  const caseChanged = fixture();
  caseChanged.control.bundles.get("c01-math-basic").identities = ["Foo"];
  assert.throws(() => validateTopologyControlProjection(caseChanged.topology, caseChanged.control), /normalized lowercase identities/);
});

function fixture({ multiFamily = false } = {}) {
  const identityRows = [identityRow("foo", "math", "basic")];
  if (multiFamily) identityRows.push(identityRow("bar", "logical", "tests"));
  identityRows.sort(([left], [right]) => left.localeCompare(right));
  const identities = identityRows.map(([name]) => name);
  const bundle = {
    id: "c01-math-basic",
    cohort: "C01",
    authority_components: ["component-foo"],
    identities,
    atomic_reason: multiFamily
      ? "One frozen shared owner spans two identity-local target families"
      : "One frozen authority component",
  };
  const topology = {
    digest: DIGEST,
    bundles: new Map([[bundle.id, structuredClone(bundle)]]),
    identities: new Map(identityRows.map(([name, row]) => [name, structuredClone(row)])),
  };
  const controlBundle = {
    ...structuredClone(bundle),
    authored_write_set: [],
    integration_outputs: [],
    prerequisites: [],
  };
  const control = {
    topology_digest: DIGEST,
    bundles: new Map([[bundle.id, controlBundle]]),
    identities: new Map(identityRows.map(([name, row]) => [name, {
      ...structuredClone(row),
      public_spelling: name,
      runtime_owner: `crates/runmat-runtime/src/builtins/${name}.rs`,
    }])),
  };
  return { topology, control };
}

function identityRow(name, domain, family) {
  return [name, {
    identity: name,
    bundle_id: "c01-math-basic",
    cohort: "C01",
    domain,
    family,
    disposition: { kind: "canonical", target: name },
  }];
}
