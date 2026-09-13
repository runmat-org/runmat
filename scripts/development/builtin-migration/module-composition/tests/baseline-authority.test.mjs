import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  deriveModuleCompositionBaselineCandidate,
  parseModuleCompositionBaselineCandidate,
} from "../baseline-candidate.mjs";
import {
  freezeReviewedModuleCompositionBaseline,
  validateReviewedModuleCompositionBaseline,
} from "../baseline-authority.mjs";
import {
  buildModuleCompositionBaselineReviewTemplate,
  sealModuleCompositionBaselineReview,
} from "../baseline-review.mjs";
import { bootstrapModuleComposition } from "../bootstrap.mjs";

const SIGNER = "0123456789ABCDEF0123456789ABCDEF01234567";
const REVISION = "a".repeat(40);
const TREE = "b".repeat(40);

test("candidate deterministically observes fixed products and leaves semantic roles unresolved", () => withRepository((root) => {
  const first = candidate(root);
  const second = candidate(root);
  assert.deepEqual(first, second);
  assert.equal(first.products.length, 99);
  const math = first.products.find((entry) => entry.product_id === "runtime-math");
  assert.equal(math.parent_evidence.path, "crates/runmat-runtime/src/builtins/math/mod.rs");
  assert.deepEqual(math.children.map((entry) => [entry.module, entry.source_kind, entry.role]), [["alpha", "file", null]]);
  assert.match(math.children[0].source_evidence.content_digest, /^sha256:/);
  assert.equal(math.parent_evidence.git_mode, "100644");
  assert.equal(Object.hasOwn(math.parent_evidence, "mode"), false);
  assert.equal(Object.isFrozen(first), true);
  assert.equal(parseModuleCompositionBaselineCandidate(structuredClone(first), root, SIGNER, sourceOptions()).digest, first.digest);
}));

test("candidate stores children canonically without losing runtime declaration order", () => withRepository((root) => {
  write(root, "crates/runmat-runtime/src/builtins/mod.rs", "#[macro_use]\npub mod common;\npub mod acceleration;\n");
  write(root, "crates/runmat-runtime/src/builtins/common.rs", "macro_rules! shared { () => {} }\n");
  write(root, "crates/runmat-runtime/src/builtins/acceleration.rs", "shared!();\n");
  const observed = candidate(root);
  const runtimeRoot = observed.products.find((entry) => entry.product_id === "runtime-root");
  assert.deepEqual(
    runtimeRoot.children.map((entry) => [entry.module, entry.declaration_order, entry.macro_use]),
    [["acceleration", 1, false], ["common", 0, true]],
  );
  assert.deepEqual(
    buildModuleCompositionBaselineReviewTemplate(observed).roles
      .filter((entry) => entry.product_id === "runtime-root")
      .map((entry) => entry.module),
    ["acceleration", "common"],
  );
}));

test("candidate preserves aggregation cfg independently from its child declaration", () => withRepository((root) => {
  write(root, "crates/runmat-builtins/src/catalog/entries/math/mod.rs", `
pub mod alpha;
pub(super) fn extend_entries(values: &mut Vec<&'static crate::BuiltinCatalogEntry>) {
    #[cfg(test)]
    values.extend(alpha::ENTRIES.iter().copied());
}
`);
  write(root, "crates/runmat-builtins/src/catalog/entries/math/alpha.rs", "pub const ENTRIES: &[&crate::BuiltinCatalogEntry] = &[];\n");
  const observed = candidate(root);
  const math = observed.products.find((entry) => entry.product_id === "catalog-math");
  assert.deepEqual(math.children[0].declaration_condition, { kind: "always" });
  assert.deepEqual(math.children[0].aggregation_sources, [{
    role: "entries", kind: "slice", order: 0, condition: { kind: "test" },
  }]);
}));

test("reviewed authority preserves an empty present parent as distinct from absence", () => withRepository((root) => {
  write(root, "crates/runmat-builtins/src/catalog/aliases/mod.rs", "pub(super) fn extend_aliases(_aliases: &mut Vec<&'static crate::BuiltinCatalogAlias>) {}\n");
  const observed = candidate(root);
  const candidateAliases = observed.products.find((entry) => entry.product_id === "catalog-aliases");
  assert.equal(candidateAliases.state, "present");
  assert.deepEqual(candidateAliases.children, []);

  const baseline = freezeReviewedModuleCompositionBaseline(observed, completedReview(observed));
  const reviewedAliases = baseline.projection.products.find((entry) => entry.product_id === "catalog-aliases");
  assert.equal(reviewedAliases.state, "present");
  assert.deepEqual(reviewedAliases.children, []);
}));

test("candidate rejects unsigned, untrusted, dirty, ambiguous, and unsupported source", () => {
  for (const [label, override, message] of [
    ["unsigned", { signature: "N\0\0" }, /unsigned commit|valid signature/],
    ["wrong signer", { signature: `G\0${"F".repeat(40)}\0` }, /trusted signer/],
    ["dirty", { status: " M crates/runmat-runtime/src/builtins/math/mod.rs\n" }, /clean source HEAD/],
  ]) withRepository((root) => assert.throws(
    () => deriveModuleCompositionBaselineCandidate(root, SIGNER, sourceOptions(override)), message, label,
  ));
  withRepository((root) => {
    write(root, "crates/runmat-runtime/src/builtins/math/alpha/mod.rs", "// collision\n");
    assert.throws(() => candidate(root), /resolve exactly once/);
  });
  withRepository((root) => {
    write(root, "crates/runmat-runtime/src/builtins/math/mod.rs", "pub mod alpha;\nfn policy() {}\n");
    assert.throws(() => candidate(root), /unsupported handwritten Rust syntax/);
  });
  withRepository((root) => {
    const headContents = new Map([["crates/runmat-runtime/src/builtins/math/alpha.rs", "pub fn committed() {}\n"]]);
    assert.throws(
      () => deriveModuleCompositionBaselineCandidate(root, SIGNER, sourceOptions({ headContents })),
      /do not match the signed source HEAD/,
    );
  });
  for (const [treeEntry, message] of [
    [{ mode: "100755", type: "blob" }, /working mode does not match/],
    [{ mode: "040000", type: "tree" }, /must be a Git blob/],
  ]) withRepository((root) => {
    const sourcePath = "crates/runmat-runtime/src/builtins/math/alpha.rs";
    assert.throws(
      () => deriveModuleCompositionBaselineCandidate(root, SIGNER, sourceOptions({
        treeEntries: new Map([[sourcePath, treeEntry]]),
      })),
      message,
    );
  });
});

test("candidate records the signed Git mode independently of host permission bits on Windows", () => withRepository((root) => {
  const sourcePath = "crates/runmat-runtime/src/builtins/math/alpha.rs";
  const observed = deriveModuleCompositionBaselineCandidate(root, SIGNER, sourceOptions({
    platform: "win32",
    treeEntries: new Map([[sourcePath, { mode: "100755", type: "blob" }]]),
  }));
  const math = observed.products.find((entry) => entry.product_id === "runtime-math");
  assert.equal(math.children[0].source_evidence.git_mode, "100755");
}));

test("review owns exact child roles and cannot alter observed source facts", () => withRepository((root) => {
  const observed = candidate(root);
  const template = buildModuleCompositionBaselineReviewTemplate(observed);
  assert.equal(template.roles.every((entry) => entry.role === null), true);
  assert.throws(() => sealModuleCompositionBaselineReview(template, observed), /role must be one of/);
  const reviewed = completedReview(observed);
  assert.throws(() => sealModuleCompositionBaselineReview({ ...reviewed }, observed), /must not supply its own digest/);
  const missing = structuredClone(reviewed);
  missing.roles.pop(); delete missing.digest;
  assert.throws(() => sealModuleCompositionBaselineReview(missing, observed), /exactly cover/);
  const invented = structuredClone(reviewed);
  invented.roles[0].source_path = "invented.rs"; delete invented.digest;
  assert.throws(() => sealModuleCompositionBaselineReview(invented, observed), /fields must be exactly/);
}));

test("reviewed baseline reconstructs exactly and rejects every stale trust boundary", () => withRepository((root) => {
  const observed = candidate(root);
  const baseline = freezeReviewedModuleCompositionBaseline(observed, completedReview(observed));
  const parsed = validateReviewedModuleCompositionBaseline(
    structuredClone(baseline), baseline.digest, root, SIGNER, sourceOptions(),
  );
  assert.equal(parsed.projection.products.length, 99);

  const wrongTrust = `sha256:${"0".repeat(64)}`;
  assert.throws(() => validateReviewedModuleCompositionBaseline(baseline, wrongTrust, root, SIGNER, sourceOptions()), /trusted digest/);
  const forged = structuredClone(baseline);
  forged.projection.products.find((entry) => entry.product_id === "runtime-math").children[0].visibility = "private";
  assert.throws(() => validateReviewedModuleCompositionBaseline(forged, forged.digest, root, SIGNER, sourceOptions()), /digest mismatch/);
  write(root, "crates/runmat-runtime/src/builtins/math/alpha.rs", "pub fn changed() {}\n");
  assert.throws(() => validateReviewedModuleCompositionBaseline(baseline, baseline.digest, root, SIGNER, sourceOptions()), /deterministic reconstruction/);
}));

test("bootstrap accepts only reviewed authority and atomically renders its present products", () => withRepository((root) => {
  const observed = candidate(root);
  const baseline = freezeReviewedModuleCompositionBaseline(observed, completedReview(observed));
  const result = bootstrapModuleComposition({
    repository: root,
    reviewedBaseline: baseline,
    trustedReviewedBaselineDigest: baseline.digest,
    trustedSignerFingerprint: SIGNER,
  }, { sourceObservation: sourceOptions({
    status: "?? .runmat-module-composition.lock/owner.json\n",
  }) });
  assert.deepEqual(result.installed.map((entry) => entry.product_id), ["runtime-math", "runtime-root"]);
  assert.match(fs.readFileSync(path.join(root, "crates/runmat-runtime/src/builtins/math/mod.rs"), "utf8"), /^\/\/ @generated/);
}));

test("bootstrap permits only its own active lock in the otherwise clean signed worktree", () => withRepository((root) => {
  const observed = candidate(root);
  const baseline = freezeReviewedModuleCompositionBaseline(observed, completedReview(observed));
  assert.throws(() => bootstrapModuleComposition({
    repository: root,
    reviewedBaseline: baseline,
    trustedReviewedBaselineDigest: baseline.digest,
    trustedSignerFingerprint: SIGNER,
  }, { sourceObservation: sourceOptions({
    status: "?? .runmat-module-composition.lock/owner.json\n?? unrelated.txt\n",
  }) }), /clean source HEAD/);
  assert.equal(
    fs.readFileSync(path.join(root, "crates/runmat-runtime/src/builtins/math/mod.rs"), "utf8"),
    "pub mod alpha;\n",
  );
}));

test("bootstrap final authority permits its exact live transaction artifacts", () => withRepository((root) => {
  const observed = candidate(root);
  const baseline = freezeReviewedModuleCompositionBaseline(observed, completedReview(observed));
  const statuses = [];
  const result = bootstrapModuleComposition({
    repository: root,
    reviewedBaseline: baseline,
    trustedReviewedBaselineDigest: baseline.digest,
    trustedSignerFingerprint: SIGNER,
  }, { sourceObservation: sourceOptions({
    status: () => liveTransactionStatus(root, statuses),
  }) });
  assert.deepEqual(result.installed.map((entry) => entry.product_id), ["runtime-math", "runtime-root"]);
  const finalAuthorityStatus = statuses.find((lines) => lines.some((line) => line === "?? .runmat-module-composition.transaction.json"));
  assert.ok(finalAuthorityStatus);
  assert.equal(finalAuthorityStatus.filter((line) => line.includes(".runmat-stage-")).length, 2);
  assert.equal(finalAuthorityStatus.every((line) => line === "?? .runmat-module-composition.lock/owner.json"
    || line === "?? .runmat-module-composition.transaction.json"
    || line.includes(".runmat-stage-")), true);
}));

test("bootstrap validates the trusted envelope before lock acquisition or recovery", () => withRepository((root) => {
  const observed = candidate(root);
  const baseline = freezeReviewedModuleCompositionBaseline(observed, completedReview(observed));
  const forged = structuredClone(baseline);
  forged.authority = "raw-projection-bypass";
  assert.throws(() => bootstrapModuleComposition({
    repository: root,
    reviewedBaseline: forged,
    trustedReviewedBaselineDigest: baseline.digest,
    trustedSignerFingerprint: SIGNER,
  }, { sourceObservation: { git() { throw new Error("Git must not run before integrity validation"); } } }), /invalid authority/);
  assert.equal(fs.existsSync(path.join(root, ".runmat-module-composition.lock")), false);
}));

test("bootstrap revalidates every signed source file after audit and before writing", () => withRepository((root) => {
  const observed = candidate(root);
  const baseline = freezeReviewedModuleCompositionBaseline(observed, completedReview(observed));
  assert.throws(() => bootstrapModuleComposition({
    repository: root,
    reviewedBaseline: baseline,
    trustedReviewedBaselineDigest: baseline.digest,
    trustedSignerFingerprint: SIGNER,
  }, {
    sourceObservation: sourceOptions(),
    afterAudit() {
      write(root, "crates/runmat-runtime/src/builtins/math/alpha.rs", "pub fn changed() {}\n");
    },
  }), /deterministic reconstruction|clean source HEAD|signed source HEAD/);
  assert.equal(
    fs.readFileSync(path.join(root, "crates/runmat-runtime/src/builtins/math/mod.rs"), "utf8"),
    "pub mod alpha;\n",
  );
}));

function candidate(root) {
  return deriveModuleCompositionBaselineCandidate(root, SIGNER, sourceOptions());
}

function completedReview(observed) {
  const value = buildModuleCompositionBaselineReviewTemplate(observed);
  for (const row of value.roles) row.role = "identity";
  value.review = { status: "reviewed", evidence: ["reviewed exact module responsibilities"] };
  return sealModuleCompositionBaselineReview(value, observed);
}

function sourceOptions(overrides = {}) {
  const values = {
    revision: REVISION, tree: TREE, status: "",
    signature: `G\0${SIGNER}\0${SIGNER}`,
    ...overrides,
  };
  return { platform: values.platform, git(_root, arguments_) {
    const command = arguments_.join(" ");
    if (command === "rev-parse --show-toplevel") return _root;
    if (command === "rev-parse HEAD") return `${values.revision}\n`;
    if (command === "rev-parse HEAD^{tree}") return `${values.tree}\n`;
    if (command.startsWith("status ")) return typeof values.status === "function" ? values.status(_root) : values.status;
    if (command === "verify-commit HEAD") {
      if (values.signature.startsWith("N")) throw new Error("unsigned commit");
      return "";
    }
    if (command.startsWith(`ls-tree -z ${values.revision} -- `)) {
      const sourcePath = command.slice(`ls-tree -z ${values.revision} -- `.length);
      const override = values.treeEntries?.get(sourcePath);
      const mode = override?.mode ?? (fs.statSync(path.join(_root, sourcePath)).mode & 0o111 ? "100755" : "100644");
      const type = override?.type ?? "blob";
      return `${mode} ${type} ${"c".repeat(40)}\t${sourcePath}\0`;
    }
    if (command.startsWith(`show ${values.revision}:`)) {
      const sourcePath = command.slice(`show ${values.revision}:`.length);
      return values.headContents?.get(sourcePath) ?? fs.readFileSync(path.join(_root, sourcePath), "utf8");
    }
    if (command.startsWith("log -1 ")) return `${values.signature}\n`;
    throw new Error(`unexpected git command ${command}`);
  } };
}

function liveTransactionStatus(root, observations) {
  const lines = [];
  if (fs.existsSync(path.join(root, ".runmat-module-composition.lock/owner.json"))) {
    lines.push("?? .runmat-module-composition.lock/owner.json");
  }
  if (fs.existsSync(path.join(root, ".runmat-module-composition.transaction.json"))) {
    lines.push("?? .runmat-module-composition.transaction.json");
  }
  for (const product of [
    "crates/runmat-runtime/src/builtins/mod.rs",
    "crates/runmat-runtime/src/builtins/math/mod.rs",
  ]) {
    const directory = path.dirname(path.join(root, product));
    const prefix = `${path.basename(product)}.runmat-stage-`;
    for (const name of fs.readdirSync(directory)) {
      if (name.startsWith(prefix)) lines.push(`?? ${path.relative(root, path.join(directory, name)).split(path.sep).join("/")}`);
    }
  }
  observations.push(lines);
  return lines.length ? `${lines.join("\n")}\n` : "";
}

function withRepository(callback) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-module-baseline-"));
  try {
    write(root, "crates/runmat-runtime/src/builtins/mod.rs", "pub mod math;\n");
    write(root, "crates/runmat-runtime/src/builtins/math/mod.rs", "pub mod alpha;\n");
    write(root, "crates/runmat-runtime/src/builtins/math/alpha.rs", "pub fn value() {}\n");
    return callback(root);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
}

function write(root, relative, contents) {
  const target = path.join(root, relative);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, contents);
}
