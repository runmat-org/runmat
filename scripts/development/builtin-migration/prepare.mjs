import fs from "node:fs";
import path from "node:path";
import { rustLeaf, sorted } from "./constants.mjs";
import { assertControlBaseline } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { assertActiveLease } from "./lease.mjs";
import { buildSourceFieldDisposition, sourceFieldBaselineDigest } from "./source-fields.mjs";
import { array, digest, exact, identity, integer, kind, nonempty, repositoryPath, stableId } from "./schema.mjs";
import { parseSourceSnapshot } from "./snapshot.mjs";

export function prepareIdentity(
  repository, inventory, control, lease, identity, outputRoot, clock = Date.now,
) {
  assertControlBaseline(control, inventory);
  assertActiveLease(lease, control, clock);
  const key = identity.toLowerCase();
  const row = inventory.identities.find((entry) => entry.identity === key);
  if (!row) throw new Error(`${identity}: identity is not present in the inventory`);
  const controlled = control.identities.get(key);
  if (!controlled) throw new Error(`${identity}: identity is not present in the reviewed control manifest`);
  if (lease.bundle.id !== controlled.bundle_id) throw new Error(`${identity}: lease does not own the identity bundle`);
  const canonicalRepository = fs.realpathSync(repository);
  const absoluteOutput = path.resolve(outputRoot);
  const canonicalOutput = canonicalPotentialPath(absoluteOutput);
  assertOutsideRepository(canonicalRepository, canonicalOutput, "prepare output");
  const requestedWorkspace = path.join(absoluteOutput, rustLeaf(key));
  const prospectiveWorkspace = canonicalPotentialPath(requestedWorkspace);
  assertOutsideRepository(canonicalRepository, prospectiveWorkspace, "identity workspace");
  assertWithinOutput(canonicalOutput, prospectiveWorkspace);
  fs.mkdirSync(requestedWorkspace, { recursive: true });
  const workspace = fs.realpathSync(requestedWorkspace);
  assertOutsideRepository(canonicalRepository, workspace, "identity workspace");
  assertWithinOutput(canonicalOutput, workspace);
  assertSafeTargets(canonicalRepository, workspace, prepareTargets(workspace, row));
  const documents = copyLegacyDocuments(repository, workspace, row);
  const checklist = buildSourceFieldDisposition(repository, row);
  writeJson(path.join(workspace, "source-field-disposition.json"), checklist);
  writeJson(path.join(workspace, "inventory-evidence.json"), row);
  writeText(path.join(workspace, "catalog", "mod.rs.template"), catalogTemplate(row));
  writeText(path.join(workspace, "catalog", "documentation.rs.template"), documentationTemplate(row, documents));
  writeText(path.join(workspace, "runtime", `${rustLeaf(key)}.rs.template`), runtimeTemplate(row));
  const report = {
    schema_version: 3,
    kind: "runmat-builtin-migration-prepare-result",
    authority: "review-workspace-only",
    identity: key,
    bundle_id: controlled.bundle_id,
    control_manifest_digest: control.digest,
    source: inventory.source,
    inventory_digest: inventory.digest,
    lease_id: lease.value.lease_id,
    lease_digest: lease.value.digest,
    workspace,
    copied_legacy_documents: documents,
    checklist_digest: evidenceDigest(checklist),
    checklist_baseline_digest: sourceFieldBaselineDigest(checklist),
    checklist_entries: checklist.sources.reduce((sum, source) => sum + source.leaves.length, 0),
    source_changes: false,
  };
  writeJson(path.join(workspace, "prepare-result.json"), report);
  return report;
}

export function parsePrepareResult(value, expected) {
  kind(value, 3, "runmat-builtin-migration-prepare-result", "prepare result");
  exact(value, ["schema_version", "kind", "authority", "identity", "bundle_id", "control_manifest_digest", "source", "inventory_digest", "lease_id", "lease_digest", "workspace", "copied_legacy_documents", "checklist_digest", "checklist_baseline_digest", "checklist_entries", "source_changes"], "prepare result");
  if (value.authority !== "review-workspace-only" || value.source_changes !== false) throw new Error("prepare result must remain review-only and source-neutral");
  identity(value.identity, "prepare identity");
  stableId(value.bundle_id, "prepare bundle id");
  digest(value.control_manifest_digest, "prepare control digest");
  parseSourceSnapshot(value.source, "prepare source snapshot");
  digest(value.inventory_digest, "prepare inventory digest");
  digest(value.checklist_digest, "prepare checklist digest");
  digest(value.checklist_baseline_digest, "prepare checklist baseline digest");
  nonempty(value.lease_id, "prepare lease id");
  digest(value.lease_digest, "prepare lease digest");
  nonempty(value.workspace, "prepare workspace");
  array(value.copied_legacy_documents, "copied legacy documents", { empty: true }).forEach((entry) => {
    exact(entry, ["source", "target"], "copied legacy document");
    repositoryPath(entry.source, "copied legacy source");
    repositoryPath(entry.target, "copied legacy target");
  });
  integer(value.checklist_entries, "prepare checklist entries");
  if (expected && ((expected.identity !== undefined && value.identity !== expected.identity) || value.bundle_id !== expected.bundle_id || value.control_manifest_digest !== expected.control_manifest_digest || value.lease_id !== expected.lease_id || value.lease_digest !== expected.lease_digest)) {
    throw new Error("prepare result does not match audit scope");
  }
  return value;
}

function prepareTargets(workspace, row) {
  const targets = [
    "source-field-disposition.json",
    "inventory-evidence.json",
    "catalog/mod.rs.template",
    "catalog/documentation.rs.template",
    `runtime/${rustLeaf(row.identity)}.rs.template`,
    "prepare-result.json",
  ].map((entry) => path.join(workspace, entry));
  for (const sourcePath of [...row.ownership.sidecars, ...row.ownership.runtime_documentation_shadows]) {
    const kind = row.ownership.sidecars.includes(sourcePath) ? "sidecar" : "runtime-shadow";
    targets.push(path.join(workspace, "legacy-documents", `${kind}-${path.posix.basename(sourcePath)}`));
  }
  return targets;
}

function copyLegacyDocuments(repository, workspace, row) {
  const paths = sorted([...row.ownership.sidecars, ...row.ownership.runtime_documentation_shadows]);
  const copied = [];
  for (const sourcePath of paths) {
    const kind = row.ownership.sidecars.includes(sourcePath) ? "sidecar" : "runtime-shadow";
    const target = path.join(workspace, "legacy-documents", `${kind}-${path.posix.basename(sourcePath)}`);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    writeBuffer(target, fs.readFileSync(path.join(repository, sourcePath)));
    copied.push({ source: sourcePath, target: path.relative(workspace, target).split(path.sep).join("/") });
  }
  return copied;
}

function catalogTemplate(row) {
  return `// REVIEW WORKSPACE TEMPLATE — not generated production source.\n` +
    `// Identity: ${row.identity}\n// Expected catalog package: ${displayExpected(row.expected_paths, "catalog_package")}\n` +
    `// Complete source-field-disposition.json before writing the typed contract.\n`;
}

function documentationTemplate(row, documents) {
  return `// REVIEW WORKSPACE TEMPLATE — transfer reviewed prose into typed documentation.\n` +
    `// Identity: ${row.identity}\n// Mechanically copied inputs:\n` +
    `${documents.map((entry) => `// - ${entry.target}`).join("\n") || "// - none"}\n`;
}

function runtimeTemplate(row) {
  return `// REVIEW WORKSPACE TEMPLATE — no implementation was inferred.\n` +
    `// Identity: ${row.identity}\n// Expected runtime path: ${displayExpected(row.expected_paths, "runtime_implementation")}\n` +
    `// Existing runtime owners:\n${row.ownership.runtime.map((entry) => `// - ${entry}`).join("\n") || "// - none"}\n`;
}

function displayExpected(expected, field) { return expected?.status === "unresolved" ? `<unresolved: ${expected.reason}>` : expected[field]; }
function writeJson(target, value) { writeText(target, `${JSON.stringify(value, null, 2)}\n`); }
function writeText(target, value) { writeBuffer(target, Buffer.from(value, "utf8")); }
function writeBuffer(target, value) {
  fs.mkdirSync(path.dirname(target), { recursive: true });
  if (fs.existsSync(target)) {
    if (fs.readFileSync(target).equals(value)) return;
    throw new Error(`prepare refuses to overwrite modified review file ${target}`);
  }
  fs.writeFileSync(target, value);
}

function canonicalPotentialPath(target) {
  const remainder = [];
  let existing = path.resolve(target);
  while (!fs.existsSync(existing)) {
    const parent = path.dirname(existing);
    if (parent === existing) throw new Error(`cannot resolve existing ancestor for ${target}`);
    remainder.unshift(path.basename(existing));
    existing = parent;
  }
  return path.join(fs.realpathSync(existing), ...remainder);
}

function assertOutsideRepository(repository, output, label) {
  if (isWithin(repository, output)) throw new Error(`${label} must be outside the canonical repository path so generated review files cannot become authority`);
}

function assertWithinOutput(output, workspace) {
  if (!isWithin(output, workspace) || output === workspace) throw new Error("identity workspace must remain within the canonical prepare output path");
}

function assertSafeTargets(repository, workspace, targets) {
  for (const target of targets) {
    const canonicalTarget = canonicalPotentialPath(target);
    assertOutsideRepository(repository, canonicalTarget, "prepare target");
    if (!isWithin(workspace, canonicalTarget)) throw new Error(`prepare target escapes the canonical identity workspace: ${target}`);
  }
}

function isWithin(parent, child) {
  const relative = path.relative(parent, child);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}
