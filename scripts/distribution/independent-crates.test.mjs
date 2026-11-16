import assert from "node:assert/strict";
import test from "node:test";
import { independentPlan, publishPlan } from "./independent-crates.mjs";

const pkg = (name, dependencies = [], extra = {}) => ({ id: name, name, version: "0.7.1", publish: null,
  dependencies: dependencies.map(dep => typeof dep === "string" ? { name: dep, kind: null } : dep), ...extra });
const metadata = packages => ({ workspace_members: packages.map(pkg => pkg.id), packages });
const plan = (packages, published = [], blocked = ["runtime"]) => independentPlan(metadata(packages),
  new Set(published), "0.7.1", new Set(blocked));

test("publishes independent dependency chains in order while retaining the blocked family", () => {
  const result = plan([pkg("consumer", ["leaf"]), pkg("runtime"), pkg("dependent", ["runtime"]), pkg("leaf")]);
  assert.deepEqual(result.plan.map(pkg => pkg.name), ["leaf", "consumer"]);
  assert.deepEqual(result.deferred, [{ name: "runtime", blocked: true, dependencies: [] },
    { name: "dependent", blocked: false, dependencies: ["runtime"] }]);
});
test("includes optional, build, development, renamed and target-specific dependencies", () => {
  for (const dep of [{ name: "runtime", optional: true }, { name: "runtime", kind: "dev" },
    { name: "runtime", kind: "build" }, { name: "runtime", rename: "alias" },
    { name: "runtime", target: "cfg(windows)" }]) {
    assert.deepEqual(plan([pkg("consumer", [dep]), pkg("runtime")]).plan, []);
  }
});
test("keeps test dependency cycles intact and defers both packages", () => {
  const result = plan([pkg("a", [{ name: "b", kind: "dev" }]), pkg("b", ["a"])]);
  assert.deepEqual(result.plan, []); assert.equal(result.deferred.length, 2);
});
test("skips existing versions and releases their consumers", () => {
  assert.deepEqual(plan([pkg("a"), pkg("b", ["a"])], ["a"]).plan.map(pkg => pkg.name), ["b"]);
});
test("does not attempt private packages or consumers of unpublished private packages", () => {
  const result = plan([pkg("private", [], { publish: [] }), pkg("consumer", ["private"]), pkg("leaf")]);
  assert.deepEqual(result.plan.map(pkg => pkg.name), ["leaf"]);
  assert.deepEqual(result.deferred[0].dependencies, ["private"]);
});
test("external dependencies do not enter the workspace publication order", () => {
  assert.equal(plan([pkg("a", ["serde"])]).plan.length, 1);
});
test("rejects mixed versions, duplicate names and empty public inventories", () => {
  assert.throws(() => plan([pkg("a", [], { version: "0.7.2" })]));
  assert.throws(() => plan([pkg("a"), pkg("a", [], { id: "other" })]));
  assert.throws(() => plan([pkg("private", [], { publish: [] })]));
});

test("publishes in order, skips existing versions, and records only visible versions", async () => {
  const visible = new Set(["existing"]), calls = [], recorded = [];
  await publishPlan([pkg("existing"), pkg("a"), pkg("b")], {
    exists: async name => visible.has(name),
    publish: async name => {
      if (name === "b") assert.ok(visible.has("a"));
      calls.push(name); visible.add(name); return { code: 0, output: "" };
    },
    onPublished: name => recorded.push(name),
  });
  assert.deepEqual(calls, ["a", "b"]); assert.deepEqual(recorded, calls);
});
test("a packaging failure stops before subsequent packages", async () => {
  const calls = [];
  await assert.rejects(publishPlan([pkg("a"), pkg("b")], {
    exists: async () => false,
    publish: async name => { calls.push(name); return { code: 101, output: "HTTP 413" }; },
  }), /publication stopped/);
  assert.deepEqual(calls, ["a"]);
});
test("does not certify a successful Cargo exit while its registry version is absent", async () => {
  const recorded = [], waits = [];
  await assert.rejects(publishPlan([pkg("a")], {
    exists: async () => false, publish: async () => ({ code: 0, output: "" }),
    wait: async delay => waits.push(delay), onPublished: name => recorded.push(name),
  }), /registry version remains absent/);
  assert.equal(waits.length, 45); assert.deepEqual(recorded, []);
});
test("waits for registry propagation before recording completion", async () => {
  let queries = 0; const waits = [], recorded = [];
  await publishPlan([pkg("a")], {
    exists: async () => ++queries >= 4, publish: async () => ({ code: 0, output: "" }),
    wait: async delay => waits.push(delay), onPublished: name => recorded.push(name),
  });
  assert.deepEqual(waits, [2_000, 2_000]); assert.deepEqual(recorded, ["a"]);
});
