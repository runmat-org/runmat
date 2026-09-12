import fs from "node:fs";
import path from "node:path";

import { validateArtifactManifest } from "../../runtime/builtin-example-verifier/artifacts.mjs";
import { validateInventory } from "../../runtime/builtin-example-verifier/inventory.mjs";
import { validatePlan } from "../../runtime/builtin-example-verifier/plan.mjs";
import { validateProductProbe } from "../../runtime/builtin-example-verifier/product-probe.mjs";
import { reconcileShardResults, validateReconciliation } from "../../runtime/builtin-example-verifier/reconcile.mjs";
import { validateShardResult } from "../../runtime/builtin-example-verifier/result-manifest.mjs";
import { compareCodePoint } from "./constants.mjs";
import { canonicalJson, contentDigest, evidenceDigest } from "./evidence.mjs";
import { SAFE_IDENTITY, absolutePath, array, digest, exact, filesystemIdentity, integer, kind, sourceRevision, uniqueStrings } from "./schema.mjs";

export function buildExampleGateProof(manifestValue, manifestEvidence, evidenceRootValue) {
  const evidenceRoot = parseExampleEvidenceRoot(evidenceRootValue);
  const manifest = parseManifest(manifestValue, evidenceRoot);
  const manifestRecord = validateManifestEvidence(manifest, manifestEvidence);
  const inventoryRecord = readJsonEvidence(manifest.inventory, "example inventory", evidenceRoot);
  const planRecord = readJsonEvidence(manifest.plan, "example plan", evidenceRoot);
  const reconciliationRecord = readJsonEvidence(manifest.reconciliation, "example reconciliation", evidenceRoot);
  const shardRecords = manifest.shard_results.map((entry) => readJsonEvidence(entry, "example shard result", evidenceRoot));
  const artifactRecords = manifest.artifact_manifests.map((entry) => readJsonEvidence(entry, "example product artifact", evidenceRoot));
  const probeRecords = manifest.product_probes.map((entry) => readJsonEvidence(entry, "example product probe", evidenceRoot));

  const inventory = validateInventory(inventoryRecord.value);
  const plan = validatePlan(planRecord.value, inventory);
  if (inventory.sourceRevision !== stripRevisionPrefix(manifest.source_revision) || inventory.sourceState !== "clean") throw new Error("example inventory does not describe the clean reviewed subject revision");
  if (inventory.scope.kind !== "development" || inventory.scope.filter !== null || inventory.scope.limit !== null || JSON.stringify(inventory.scope.builtins) !== JSON.stringify(manifest.identities)) {
    throw new Error("example inventory scope must be the exact reviewed bundle identity set");
  }
  if (plan.productScope !== "all") throw new Error("bundle example evidence must execute every product required by its declared harnesses");

  const artifactManifests = artifactRecords.map((record) => {
    validateArtifactManifest(record.value, { sourceRevision: inventory.sourceRevision, verifyFiles: true, manifestPath: record.path });
    return { manifest: record.value, manifestPath: record.path };
  });
  const productProbes = probeRecords.map((record) => {
    validateProductProbe(record.value, { sourceRevision: inventory.sourceRevision });
    return record.value;
  });
  const shards = shardRecords.map((record) => validateShardResult(record.value));
  const recomputed = reconcileShardResults(inventory, plan, shards, { closure: false, artifactManifests, productProbes });
  const supplied = validateReconciliation(reconciliationRecord.value);
  if (JSON.stringify(recomputed) !== JSON.stringify(supplied)) throw new Error("example reconciliation differs from the independently recomputed result");
  if (recomputed.status !== "passed") throw new Error("example reconciliation is not passing");
  const plannedProducts = plan.products.map((entry) => entry.kind).sort(compareCodePoint);
  const manifestedProducts = artifactManifests.map((entry) => entry.manifest.product).sort(compareCodePoint);
  if (JSON.stringify(plannedProducts) !== JSON.stringify(manifestedProducts)) throw new Error("example evidence does not contain exactly one verified artifact for every planned product");
  if (JSON.stringify(Object.keys(recomputed.productArtifactDigests).sort(compareCodePoint)) !== JSON.stringify(plannedProducts)) throw new Error("example shard evidence does not bind every planned product artifact");

  const rows = manifest.identities.map((identity) => {
    const examples = inventory.examples.filter((entry) => entry.builtinKey === identity);
    const executionUnits = inventory.executionUnits.filter((entry) => entry.builtinKey === identity);
    const catalogOnly = examples.every((entry) => entry.authority === "catalog");
    const status = examples.length > 0 && catalogOnly ? "passed" : examples.length === 0 ? "absent" : "failed";
    const row = {
      identity,
      status,
      example_identities: examples.map((entry) => entry.identity).sort(compareCodePoint),
      execution_identities: executionUnits.map((entry) => entry.executionIdentity).sort(compareCodePoint),
    };
    return { ...row, evidence_digest: evidenceDigest(row) };
  });
  const evidence = {
    manifest: manifestRecord,
    inventory: withoutValue(inventoryRecord), plan: withoutValue(planRecord), reconciliation: withoutValue(reconciliationRecord),
    shard_results: shardRecords.map(withoutValue), artifact_manifests: artifactRecords.map(withoutValue), product_probes: probeRecords.map(withoutValue),
  };
  const payload = {
    schema_version: 3, kind: "runmat-builtin-example-gate-proof", authority: "machine-derived-product-evidence",
    source_revision: manifest.source_revision, identities: manifest.identities,
    evidence_root: evidenceRoot,
    example_inventory_digest: inventory.inventoryDigest, plan_digest: plan.planDigest,
    reconciliation_digest: recomputed.reconciliationDigest, product_scope: recomputed.productScope,
    evidence, rows, result: rows.some((entry) => entry.status === "failed") ? "fail" : "pass",
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseExampleGateProof(value, expected) {
  kind(value, 3, "runmat-builtin-example-gate-proof", "example gate proof");
  exact(value, ["schema_version", "kind", "authority", "source_revision", "identities", "evidence_root", "example_inventory_digest", "plan_digest", "reconciliation_digest", "product_scope", "evidence", "rows", "result", "digest"], "example gate proof");
  if (value.authority !== "machine-derived-product-evidence" || value.product_scope !== "all") throw new Error("example gate proof has invalid authority or product scope");
  sourceRevision(value.source_revision, "example gate source revision");
  const identities = uniqueStrings(value.identities, "example gate identities", { pattern: SAFE_IDENTITY, lower: true }).sort(compareCodePoint);
  for (const field of ["example_inventory_digest", "plan_digest", "reconciliation_digest", "digest"]) digest(value[field], `example gate ${field}`);
  if (value.result !== "pass") throw new Error("example gate proof is not passing");
  const evidenceRoot = parseExampleEvidenceRoot(value.evidence_root);
  parseEvidence(value.evidence, evidenceRoot);
  const rows = array(value.rows, "example gate rows").map((entry) => {
    exact(entry, ["identity", "status", "example_identities", "execution_identities", "evidence_digest"], "example gate row");
    if (!identities.includes(entry.identity) || !["passed", "absent", "failed"].includes(entry.status)) throw new Error("example gate row has invalid identity or status");
    for (const field of ["example_identities", "execution_identities"]) {
      const values = array(entry[field], `example row ${field}`, { empty: true });
      values.forEach((id) => digest(id, `example row ${field}`));
      if (new Set(values).size !== values.length || JSON.stringify([...values].sort(compareCodePoint)) !== JSON.stringify(values)) throw new Error(`${entry.identity}: ${field} must be unique and canonically ordered`);
    }
    digest(entry.evidence_digest, "example row evidence digest");
    const { evidence_digest: ignored, ...payload } = entry;
    if (entry.evidence_digest !== evidenceDigest(payload)) throw new Error(`${entry.identity}: example row digest is inconsistent`);
    return entry;
  });
  if (JSON.stringify(rows.map((entry) => entry.identity)) !== JSON.stringify(identities)) throw new Error("example gate rows do not cover the identity set exactly once");
  if ((rows.some((entry) => entry.status === "failed") ? "fail" : "pass") !== value.result) throw new Error("example gate aggregate result conflicts with its rows");
  const { digest: ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("example gate proof digest is inconsistent");
  if (expected && (value.source_revision !== expected.source_revision || JSON.stringify(identities) !== JSON.stringify([...expected.identities].sort(compareCodePoint)))) throw new Error("example gate proof has stale or mismatched provenance");
  if (expected?.manifest_evidence && JSON.stringify(value.evidence.manifest) !== JSON.stringify(expected.manifest_evidence)) {
    throw new Error("example gate proof manifest evidence differs from the adapter-owned input");
  }
  if (expected?.evidence_root && JSON.stringify(evidenceRoot) !== JSON.stringify(expected.evidence_root)) {
    throw new Error("example gate proof evidence root differs from the adapter-admitted root");
  }
  return { ...value, identities, rows };
}

export function parseExampleGateManifest(value, evidenceRoot = null) { return parseManifest(value, evidenceRoot); }

export function parseExampleEvidenceRoot(value, expected = null) {
  exact(value, ["path", "filesystem_id"], "example evidence root");
  const rootPath = absolutePath(value.path, "example evidence root path");
  filesystemIdentity(value.filesystem_id, "example evidence root filesystem id");
  const stat = fs.lstatSync(rootPath);
  if (!stat.isDirectory() || fs.realpathSync(rootPath) !== rootPath) {
    throw new Error("example evidence root must be a canonical directory");
  }
  if (observedFilesystemIdentity(rootPath) !== value.filesystem_id) {
    throw new Error("example evidence root differs from its admitted filesystem");
  }
  if (expected) {
    const admitted = fs.realpathSync(expected.evidence_path);
    if (value.filesystem_id !== expected.filesystem_id || !isWithin(admitted, rootPath)) {
      throw new Error("example evidence root is outside the admitted target-temp storage volume");
    }
  }
  return { path: rootPath, filesystem_id: value.filesystem_id };
}

function parseManifest(value, evidenceRoot = null) {
  kind(value, 1, "runmat-builtin-example-gate-manifest", "example gate manifest");
  exact(value, ["schema_version", "kind", "source_revision", "identities", "inventory", "plan", "reconciliation", "shard_results", "artifact_manifests", "product_probes"], "example gate manifest");
  sourceRevision(value.source_revision, "example gate manifest source revision");
  const identities = uniqueStrings(value.identities, "example gate manifest identities", { pattern: SAFE_IDENTITY, lower: true }).sort(compareCodePoint);
  const single = (entry, label) => absolutePath(entry, label);
  const list = (entries, label) => {
    const values = array(entries, label, { empty: true }).map((entry) => absolutePath(entry, label)).sort(compareCodePoint);
    if (new Set(values).size !== values.length) throw new Error(`${label} contains duplicate paths`);
    return values;
  };
  const result = {
    ...value, identities,
    inventory: single(value.inventory, "example inventory path"), plan: single(value.plan, "example plan path"), reconciliation: single(value.reconciliation, "example reconciliation path"),
    shard_results: list(value.shard_results, "example shard result paths"), artifact_manifests: list(value.artifact_manifests, "example artifact manifest paths"), product_probes: list(value.product_probes, "example product probe paths"),
  };
  if (evidenceRoot) for (const source of manifestEvidencePaths(result)) {
    readCanonicalRegularFile(source, "example manifest evidence", evidenceRoot.path);
  }
  return result;
}

function readJsonEvidence(source, label, evidenceRoot = null) {
  const bytes = readCanonicalRegularFile(source, label, evidenceRoot?.path ?? null);
  let value;
  try { value = JSON.parse(bytes); } catch (error) { throw new Error(`${label} is not valid JSON: ${error.message}`); }
  return { path: path.resolve(source), byte_length: bytes.length, content_digest: contentDigest(bytes), value };
}

function validateManifestEvidence(manifest, evidence) {
  parseEvidenceRecord(evidence, "example gate manifest evidence");
  const record = readJsonEvidence(evidence.path, "example gate manifest");
  if (record.byte_length !== evidence.byte_length || record.content_digest !== evidence.content_digest) {
    throw new Error("example gate manifest evidence differs from its file bytes");
  }
  if (canonicalManifest(record.value) !== canonicalManifest(manifest)) {
    throw new Error("example gate manifest evidence describes a different manifest");
  }
  return withoutValue(record);
}

function canonicalManifest(value) { return canonicalJson(parseManifest(value)); }

function parseEvidence(value, evidenceRoot) {
  exact(value, ["manifest", "inventory", "plan", "reconciliation", "shard_results", "artifact_manifests", "product_probes"], "example gate input evidence");
  parseEvidenceRecord(value.manifest, "example gate manifest evidence");
  parseEvidenceRecord(value.inventory, "example inventory evidence");
  parseEvidenceRecord(value.plan, "example plan evidence");
  parseEvidenceRecord(value.reconciliation, "example reconciliation evidence");
  for (const field of ["shard_results", "artifact_manifests", "product_probes"]) {
    const records = array(value[field], `example gate ${field} evidence`, { empty: field === "product_probes" });
    records.forEach((entry) => parseEvidenceRecord(entry, `example gate ${field} evidence`));
    const paths = records.map((entry) => entry.path);
    if (new Set(paths).size !== paths.length || JSON.stringify([...paths].sort(compareCodePoint)) !== JSON.stringify(paths)) throw new Error(`example gate ${field} evidence must be unique and canonically ordered`);
  }
  validateCurrentEvidence(value.manifest, "example gate manifest evidence");
  for (const record of [value.inventory, value.plan, value.reconciliation, ...value.shard_results, ...value.artifact_manifests, ...value.product_probes]) {
    validateCurrentEvidence(record, "example proof evidence", evidenceRoot.path);
  }
}

function parseEvidenceRecord(value, label) {
  exact(value, ["path", "byte_length", "content_digest"], label);
  absolutePath(value.path, `${label} path`);
  integer(value.byte_length, `${label} byte length`, 0);
  digest(value.content_digest, `${label} content digest`);
}

function withoutValue(record) { const { value: ignored, ...evidence } = record; return evidence; }
function stripRevisionPrefix(value) { return value.slice("git:".length); }

function manifestEvidencePaths(manifest) {
  return [manifest.inventory, manifest.plan, manifest.reconciliation, ...manifest.shard_results, ...manifest.artifact_manifests, ...manifest.product_probes];
}

function readCanonicalRegularFile(source, label, root = null) {
  const candidate = absolutePath(source, `${label} path`);
  const stat = fs.lstatSync(candidate);
  if (!stat.isFile() || fs.realpathSync(candidate) !== candidate) {
    throw new Error(`${label} must be a canonical regular file`);
  }
  if (root && !isWithin(root, candidate)) throw new Error(`${label} is outside the admitted evidence root`);
  if (root && observedFilesystemIdentity(candidate) !== observedFilesystemIdentity(root)) {
    throw new Error(`${label} is outside the admitted evidence filesystem`);
  }
  return fs.readFileSync(candidate);
}

function validateCurrentEvidence(record, label, root = null) {
  const bytes = readCanonicalRegularFile(record.path, label, root);
  if (bytes.length !== record.byte_length || contentDigest(bytes) !== record.content_digest) {
    throw new Error(`${label} bytes differ from the content-bound proof`);
  }
}

function isWithin(root, candidate) {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function observedFilesystemIdentity(candidate) {
  const stats = fs.statSync(candidate, { bigint: true });
  return process.platform === "win32"
    ? `windows-volume:${stats.dev.toString(16).padStart(8, "0")}`
    : `posix-dev:${stats.dev}`;
}
