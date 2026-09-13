import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { readExampleGateInput, stageExampleGateInput } from "../example-gate-input.mjs";

const REVISION = `git:${"a".repeat(40)}`;

test("adapter stages a canonical example manifest at its fixed admitted-temp path", () => {
  withTemporaryDirectory((temporary) => {
    const fixture = fixtureInputs(temporary);
    const expected = expectation(temporary, ["beta", "alpha"]);
    const staged = stageExampleGateInput(fixture, temporary, expected);
    const decoded = readExampleGateInput(staged.stdin);
    assert.equal(decoded.manifest_evidence.path, path.join(fs.realpathSync(temporary), "example-gate-manifest.json"));
    assert.deepEqual(decoded.manifest.identities, ["alpha", "beta"]);
    assert.deepEqual(decoded.manifest_evidence, staged.manifest_evidence);
    assert.equal(fs.readFileSync(decoded.manifest_evidence.path, "utf8").endsWith("\n"), true);

    assert.throws(
      () => stageExampleGateInput(fixture, temporary, expectation(temporary, ["alpha", "beta"])),
      /target already exists/,
    );
  });
});

test("example input rejects caller command authority and stale provenance", () => {
  withTemporaryDirectory((temporary) => {
    const extra = fixtureInputs(temporary);
    extra.arguments = ["--manifest", "/tmp/other.json"];
    assert.throws(
      () => stageExampleGateInput(extra, temporary, expectation(temporary, ["alpha", "beta"])),
      /fields must be exactly/,
    );
    assert.throws(
      () => stageExampleGateInput(fixtureInputs(temporary), temporary, expectation(temporary, ["alpha"])),
      /stale or mismatched bundle provenance/,
    );
  });
});

test("example input rejects changed bytes, changed digests, and noncanonical manifest paths", () => {
  withTemporaryDirectory((temporary) => {
    const staged = stageExampleGateInput(fixtureInputs(temporary), temporary, expectation(temporary, ["alpha", "beta"]));
    const envelope = JSON.parse(staged.stdin);
    envelope.manifest.content_digest = `sha256:${"0".repeat(64)}`;
    assert.throws(() => readExampleGateInput(JSON.stringify(envelope)), /bytes differ/);

    envelope.manifest = staged.manifest_evidence;
    fs.appendFileSync(staged.manifest_evidence.path, " ");
    assert.throws(() => readExampleGateInput(JSON.stringify(envelope)), /bytes differ/);

    const alias = path.join(temporary, "manifest-alias.json");
    fs.symlinkSync(staged.manifest_evidence.path, alias);
    envelope.manifest = { ...staged.manifest_evidence, path: alias };
    assert.throws(() => readExampleGateInput(JSON.stringify(envelope)), /canonical regular file/);
  });
});

test("example input rejects symlinked and out-of-root evidence files", () => {
  withTemporaryDirectory((temporary) => {
    const outside = path.join(temporary, "outside.json");
    fs.writeFileSync(outside, "{}\n");
    const outOfRoot = fixtureInputs(temporary);
    outOfRoot.example_manifest.inventory = fs.realpathSync(outside);
    assert.throws(
      () => stageExampleGateInput(outOfRoot, temporary, expectation(temporary, ["alpha", "beta"])),
      /outside the admitted evidence root/,
    );

    const symlinked = fixtureInputs(temporary);
    const alias = path.join(symlinked.evidence_root, "inventory-alias.json");
    fs.symlinkSync(symlinked.example_manifest.inventory, alias);
    symlinked.example_manifest.inventory = alias;
    assert.throws(
      () => stageExampleGateInput(symlinked, temporary, expectation(temporary, ["alpha", "beta"])),
      /canonical regular file/,
    );
  });
});

function fixtureInputs(temporary) {
  const requestedRoot = path.join(temporary, `evidence-${Math.random().toString(16).slice(2)}`);
  fs.mkdirSync(requestedRoot);
  const root = fs.realpathSync(requestedRoot);
  const evidencePath = (name) => {
    const target = path.join(root, name);
    fs.writeFileSync(target, "{}\n");
    return target;
  };
  return {
    artifact_output: "/tmp/example-reconciliation.json",
    evidence_root: fs.realpathSync(root),
    example_manifest: {
      schema_version: 1,
      kind: "runmat-builtin-example-gate-manifest",
      source_revision: REVISION,
      identities: ["alpha", "beta"],
      inventory: evidencePath("inventory.json"),
      plan: evidencePath("plan.json"),
      reconciliation: evidencePath("reconciliation.json"),
      shard_results: [evidencePath("shard.json")],
      artifact_manifests: [evidencePath("artifact.json")],
      product_probes: [],
    },
  };
}

function expectation(root, identities) {
  const stat = fs.statSync(root, { bigint: true });
  const filesystemId = process.platform === "win32"
    ? `windows-volume:${stat.dev.toString(16).padStart(8, "0")}`
    : `posix-dev:${stat.dev}`;
  return { source_revision: REVISION, identities, evidence_storage: { evidence_path: fs.realpathSync(root), filesystem_id: filesystemId } };
}

function withTemporaryDirectory(callback) {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "runmat-example-input-"));
  try { callback(temporary); } finally { fs.rmSync(temporary, { recursive: true, force: true }); }
}
