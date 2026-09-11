import assert from "node:assert/strict";
import test from "node:test";

import {
  canonicalComponentIds,
  parseBundleComposition,
  parseIdentityTarget,
  parseReviewBaseline,
  parseReviewedEvidence,
} from "../schema.mjs";

const SHA = `sha256:${"a".repeat(64)}`;

test("topology schema accepts only exact v2 review primitives", () => {
  assert.doesNotThrow(() => parseReviewBaseline({
    revision: `git:${"b".repeat(40)}`,
    inventory_digest: SHA,
    component_graph_digest: SHA,
    control_draft_digest: SHA,
  }));
  assert.doesNotThrow(() => parseIdentityTarget({
    identity: "gpuArray",
    domain: "acceleration",
    family: "device/transfer",
    classification: "normalization",
    evidence: ["review:gpu-transfer-boundary"],
  }, "target"));
  assert.doesNotThrow(() => parseIdentityTarget({
    identity: "newbuiltin",
    domain: "math",
    family: "basic",
    classification: "newly-classified",
    evidence: ["review:first explicit target classification"],
  }, "target"));
  assert.doesNotThrow(() => parseReviewedEvidence({ status: "reviewed", evidence: ["review:root"] }, "review"));
});

test("topology schema rejects unknown fields, invalid package paths, and implicit review", () => {
  assert.throws(() => parseIdentityTarget({
    identity: "alpha", domain: "math", family: "basic", classification: "preserved", evidence: ["review:alpha"], selector: "math/*",
  }, "target"), /fields must be exactly/);
  assert.throws(() => parseIdentityTarget({
    identity: "alpha", domain: "Math", family: "basic", classification: "preserved", evidence: ["review:alpha"],
  }, "target"), /lowercase package path/);
  assert.throws(() => parseIdentityTarget({
    identity: "alpha", domain: "math/core", family: "basic", classification: "preserved", evidence: ["review:alpha"],
  }, "target"), /lowercase package path/);
  assert.throws(() => parseIdentityTarget({
    identity: "alpha", domain: "math", family: "basic", classification: "unresolved-baseline", evidence: ["review:alpha"],
  }, "target"), /classification must be one of/);
  assert.throws(() => parseReviewedEvidence({ status: "unreviewed", evidence: [] }, "review"), /status must be reviewed/);
});

test("composition evidence and component identifiers use canonical exact sets", () => {
  assert.doesNotThrow(() => parseBundleComposition({
    kind: "shared-target-package",
    target_packages: [{ domain: "math", family: "basic" }],
    authored_write_set: [
      { kind: "file", path: "crates/runmat-runtime/src/builtins/math/basic.rs" },
      { kind: "tree", path: "crates/runmat-builtins/src/catalog/entries/math/basic" },
    ],
    shared_authority_sources: [],
    evidence: ["review:shared-package-composition"],
  }, "composition"));
  assert.throws(() => parseBundleComposition({
    kind: "single-component",
    target_packages: [
      { domain: "math", family: "zeta" },
      { domain: "math", family: "alpha" },
    ],
    authored_write_set: [], shared_authority_sources: [], evidence: ["review:ordering"],
  }, "composition"), /canonical order/);
  assert.throws(() => canonicalComponentIds(["component-beta", "component-alpha"], "components"), /canonical order/);
  assert.throws(() => canonicalComponentIds(["alpha"], "components"), /start with component-/);
});
