import fs from "node:fs";
import path from "node:path";
import { compareCodePoint, rustLeaf, sorted } from "./constants.mjs";

export function prepareIdentity(repository, inventory, identity, outputRoot) {
  const key = identity.toLowerCase();
  const row = inventory.identities.find((entry) => entry.identity === key);
  if (!row) throw new Error(`${identity}: identity is not present in the inventory`);
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
  const checklist = sourceFieldChecklist(repository, row);
  writeJson(path.join(workspace, "source-field-disposition.json"), checklist);
  writeJson(path.join(workspace, "inventory-evidence.json"), row);
  writeText(path.join(workspace, "catalog", "mod.rs.template"), catalogTemplate(row));
  writeText(path.join(workspace, "catalog", "documentation.rs.template"), documentationTemplate(row, documents));
  writeText(path.join(workspace, "runtime", `${rustLeaf(key)}.rs.template`), runtimeTemplate(row));
  const report = {
    schema_version: 1,
    kind: "runmat-builtin-migration-prepare-result",
    authority: "review-workspace-only",
    identity: key,
    workspace,
    copied_legacy_documents: documents,
    checklist_entries: checklist.fields.length,
    source_changes: false,
  };
  writeJson(path.join(workspace, "prepare-result.json"), report);
  return report;
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

function sourceFieldChecklist(repository, row) {
  const fields = [];
  for (const sourcePath of sorted([...row.ownership.sidecars, ...row.ownership.runtime_documentation_shadows])) {
    const document = JSON.parse(fs.readFileSync(path.join(repository, sourcePath), "utf8"));
    for (const [field, value] of Object.entries(document).sort(([a], [b]) => compareCodePoint(a, b))) {
      fields.push({ source: sourcePath, field, status: "review-required", destination: null, value });
    }
  }
  return { schema_version: 1, kind: "runmat-builtin-source-field-disposition", identity: row.identity, fields };
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
