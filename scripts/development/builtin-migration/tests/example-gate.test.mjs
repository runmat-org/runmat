import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import { buildArtifactManifestFromFiles, buildArtifactManifestFromTree } from "../../../runtime/builtin-example-verifier/artifacts.mjs";
import { buildInventory } from "../../../runtime/builtin-example-verifier/inventory.mjs";
import { sha256 } from "../../../runtime/builtin-example-verifier/identity.mjs";
import { buildPlan } from "../../../runtime/builtin-example-verifier/plan.mjs";
import { reconcileShardResults } from "../../../runtime/builtin-example-verifier/reconcile.mjs";
import { buildShardResult } from "../../../runtime/builtin-example-verifier/result-manifest.mjs";
import { buildExampleGateProof, parseExampleGateProof } from "../example-gate.mjs";
import { contentDigest } from "../evidence.mjs";

const SOURCE = "0123456789abcdef0123456789abcdef01234567";

test("example gate proof binds an exact multi-identity scope to every declared product lane", () => {
  withEvidence((manifest, records) => {
    const manifestEvidence = writeManifestEvidence(records.root, manifest);
    const rootBinding = evidenceRoot(records.root);
    const proof = buildExampleGateProof(manifest, manifestEvidence, rootBinding);
    assert.equal(proof.result, "pass");
    assert.deepEqual(proof.identities, ["alpha", "beta"]);
    assert.equal(proof.rows.find((entry) => entry.identity === "alpha").status, "passed");
    assert.equal(proof.rows.find((entry) => entry.identity === "beta").status, "absent");
    assert.doesNotThrow(() => parseExampleGateProof(proof, { source_revision: `git:${SOURCE}`, identities: ["alpha", "beta"] }));

    const omittedIdentity = structuredClone(manifest);
    omittedIdentity.identities = ["alpha"];
    const omittedIdentityEvidence = writeManifestEvidence(records.root, omittedIdentity, "omitted-identity-manifest.json");
    assert.throws(() => buildExampleGateProof(omittedIdentity, omittedIdentityEvidence, rootBinding), /exact reviewed bundle identity set/);

    const omittedShard = structuredClone(manifest);
    omittedShard.shard_results.pop();
    const omittedShardEvidence = writeManifestEvidence(records.root, omittedShard, "omitted-shard-manifest.json");
    assert.throws(() => buildExampleGateProof(omittedShard, omittedShardEvidence, rootBinding), /Missing shard results/);

    fs.appendFileSync(records.nativeBinary, "changed");
    assert.throws(() => buildExampleGateProof(manifest, manifestEvidence, rootBinding), /does not match|digest|tree/i);
    fs.writeFileSync(records.nativeBinary, "native");
    fs.appendFileSync(records.inventoryPath, " ");
    assert.throws(
      () => parseExampleGateProof(proof, { source_revision: `git:${SOURCE}`, identities: ["alpha", "beta"] }),
      /content-bound proof/,
    );
  });
});

function withEvidence(callback) {
  const root = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), "runmat-example-gate-")));
  try {
    const inventory = buildInventory({
      schema_version: 1,
      builtins: [
        { key: "alpha", authority: "catalog", category: "math", examples: [{ id: "portable", input: "disp(1)", output: "1", harness: "Portable", compatibility: "RunMat", verification: "Succeeds" }] },
        { key: "beta", authority: "catalog", category: "math", examples: [] },
      ],
    }, { sourceRevision: SOURCE, sourceState: "clean", runnerDigest: sha256("runner"), scope: { builtins: ["beta", "alpha"] } });
    const plan = buildPlan(inventory);
    const nativeRoot = path.join(root, "native");
    fs.mkdirSync(nativeRoot);
    const nativeBinary = path.join(nativeRoot, "runmat");
    fs.writeFileSync(nativeBinary, "native");
    const nativeManifestPath = path.join(root, "native-artifact.json");
    const native = buildArtifactManifestFromTree({ product: "native-cli", artifactProfile: "embedded-aot", sourceRevision: SOURCE, manifestPath: nativeManifestPath, root: nativeRoot, entrypoints: { "runmat-binary": "runmat" } });
    const browserJs = path.join(root, "runmat.js");
    const browserWasm = path.join(root, "runmat.wasm");
    fs.writeFileSync(browserJs, "js");
    fs.writeFileSync(browserWasm, "wasm");
    const browserManifestPath = path.join(root, "browser-artifact.json");
    const browser = buildArtifactManifestFromFiles({ product: "browser-wasm", artifactProfile: "web", sourceRevision: SOURCE, manifestPath: browserManifestPath, files: { "wasm-js": browserJs, "wasm-binary": browserWasm } });
    write(nativeManifestPath, native);
    write(browserManifestPath, browser);
    const artifacts = new Map([["native-cli", native], ["browser-wasm", browser]]);
    const shards = plan.lanes.flatMap((lane) => lane.shards.map((shard) => {
      const units = shard.executionIdentities.map((id) => inventory.executionUnits.find((entry) => entry.executionIdentity === id));
      return buildShardResult({
        sourceRevision: SOURCE, sourceState: "clean", inventoryDigest: inventory.inventoryDigest, planDigest: plan.planDigest, runnerDigest: plan.runnerDigest,
        lane: lane.lane, shardIndex: shard.index, shardCount: lane.shardCount, assignmentDigest: shard.assignmentDigest,
        product: lane.product, artifactProfile: plan.products.find((entry) => entry.kind === lane.product).artifactProfile,
        artifactManifestDigest: artifacts.get(lane.product).artifactManifestDigest,
        environment: { platform: "test", architecture: "test", node: "test" }, limits: lane.limits,
        adapter: { available: true, reason: "" },
        results: units.map((unit) => ({ executionIdentity: unit.executionIdentity, exampleIdentity: unit.exampleIdentity, definitionDigest: unit.definitionDigest, builtinKey: unit.builtinKey, exampleId: unit.exampleId, lane: unit.lane, status: "passed", errorIdentifier: "", errorText: "", normalizedExpected: "1", normalizedActual: "1", imageRelativePath: null })),
      });
    }));
    const artifactManifests = [{ manifest: native, manifestPath: nativeManifestPath }, { manifest: browser, manifestPath: browserManifestPath }];
    const reconciliation = reconcileShardResults(inventory, plan, shards, { artifactManifests });
    const inventoryPath = path.join(root, "inventory.json");
    const planPath = path.join(root, "plan.json");
    const reconciliationPath = path.join(root, "reconciliation.json");
    write(inventoryPath, inventory); write(planPath, plan); write(reconciliationPath, reconciliation);
    const shardPaths = shards.map((value, index) => { const target = path.join(root, `${index}.shard-result.json`); write(target, value); return target; });
    callback({ schema_version: 1, kind: "runmat-builtin-example-gate-manifest", source_revision: `git:${SOURCE}`, identities: ["alpha", "beta"], inventory: inventoryPath, plan: planPath, reconciliation: reconciliationPath, shard_results: shardPaths, artifact_manifests: [nativeManifestPath, browserManifestPath], product_probes: [] }, { inventoryPath, nativeBinary, root });
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
}

function write(target, value) { fs.writeFileSync(target, `${JSON.stringify(value, null, 2)}\n`); }

function writeManifestEvidence(root, value, filename = "gate-manifest.json") {
  const manifestPath = path.join(root, filename);
  write(manifestPath, value);
  const bytes = fs.readFileSync(manifestPath);
  return { path: manifestPath, byte_length: bytes.length, content_digest: contentDigest(bytes) };
}

function evidenceRoot(root) {
  const stat = fs.statSync(root, { bigint: true });
  const filesystemId = process.platform === "win32"
    ? `windows-volume:${stat.dev.toString(16).padStart(8, "0")}`
    : `posix-dev:${stat.dev}`;
  return { path: fs.realpathSync(root), filesystem_id: filesystemId };
}
