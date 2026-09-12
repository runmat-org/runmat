import { compareCodePoint } from "./constants.mjs";
import { contentDigest, evidenceDigest } from "./evidence.mjs";
import { parseCompletedSourceDisposition } from "./source-fields.mjs";
import { array, digest, enumValue, exact, identity, integer, kind, nonempty, repositoryPath, sourceRevision, stableId, uniqueStrings } from "./schema.mjs";

export function buildDocumentationCutoverArtifact(input) {
  exact(input, ["catalog_export", "catalog_export_bytes", "source_dispositions", "expected_sources", "provenance"], "documentation cutover input");
  const provenance = parseProvenance(input.provenance);
  const encoded = Buffer.from(input.catalog_export_bytes);
  const parsedBytes = JSON.parse(encoded.toString("utf8"));
  if (JSON.stringify(parsedBytes) !== JSON.stringify(input.catalog_export)) throw new Error("catalog export bytes and parsed value differ");
  const exportDigest = contentDigest(encoded);
  parseCatalogExport(input.catalog_export);
  const documents = new Map(input.catalog_export.builtins.map((entry) => [String(entry.key).toLowerCase(), entry]));
  const rows = [];
  const observedIdentities = new Set();
  const observedSources = new Map();
  exact(input.expected_sources, provenance.identities, "documentation expected sources");
  for (const record of array(input.source_dispositions, "documentation source dispositions", { empty: true })) {
    exact(record, ["baseline_digest", "value"], "documentation source disposition input");
    const expectedIdentity = String(record.value?.identity ?? "").toLowerCase();
    const expectedSources = parseExpectedSources(input.expected_sources[expectedIdentity], expectedIdentity);
    const derivedBaselineDigest = evidenceDigest({ identity: expectedIdentity, sources: expectedSources });
    if (record.baseline_digest !== derivedBaselineDigest) throw new Error(`${expectedIdentity}: supplied disposition baseline differs from the frozen source documents`);
    const disposition = parseCompletedSourceDisposition(record.value, expectedIdentity, derivedBaselineDigest);
    if (!provenance.identities.includes(expectedIdentity)) throw new Error(`${expectedIdentity}: documentation disposition is outside the reviewed bundle`);
    if (observedIdentities.has(expectedIdentity)) throw new Error(`${expectedIdentity}: duplicate documentation disposition`);
    observedIdentities.add(expectedIdentity);
    observedSources.set(expectedIdentity, disposition.sources.map((entry) => entry.path).sort(compareCodePoint));
    const document = documents.get(expectedIdentity);
    if (!document || document.authority !== "catalog") throw new Error(`${expectedIdentity}: canonical catalog export document is absent`);
    for (const source of disposition.sources) for (const leaf of source.leaves) rows.push(reconcileLeaf(expectedIdentity, source, leaf, document));
  }
  for (const identityName of provenance.identities) {
    const expectedPaths = parseExpectedSources(input.expected_sources[identityName], identityName).map((entry) => entry.path);
    if (JSON.stringify(expectedPaths) !== JSON.stringify(observedSources.get(identityName) ?? [])) throw new Error(`${identityName}: source dispositions do not exactly cover the baseline legacy documentation sources`);
  }
  rows.sort((left, right) => compareCodePoint(`${left.identity}\0${left.source_path}\0${left.pointer}`, `${right.identity}\0${right.source_path}\0${right.pointer}`));
  const artifactProvenance = { ...provenance, source_dispositions_digest: evidenceDigest(input.source_dispositions) };
  const payload = {
    schema_version: 2, kind: "runmat-builtin-documentation-cutover-evidence", authority: "derived-machine-evidence-only",
    provenance: artifactProvenance,
    catalog_export: {
      schema_version: input.catalog_export.schema_version,
      content_digest: exportDigest,
      evidence_digest: evidenceDigest(input.catalog_export_bytes),
      bytes: input.catalog_export_bytes,
      value: input.catalog_export,
    },
    rows, summary: { identities: observedIdentities.size, leaves: rows.length, passed: rows.filter((entry) => entry.result === "pass").length },
    result: rows.every((entry) => entry.result === "pass") ? "pass" : "fail",
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseDocumentationCutoverArtifact(value, expected = null) {
  kind(value, 2, "runmat-builtin-documentation-cutover-evidence", "documentation cutover evidence");
  exact(value, ["schema_version", "kind", "authority", "provenance", "catalog_export", "rows", "summary", "result", "digest"], "documentation cutover evidence");
  if (value.authority !== "derived-machine-evidence-only") throw new Error("documentation cutover evidence has invalid authority");
  const provenance = parseProvenance(value.provenance, true);
  exact(value.catalog_export, ["schema_version", "content_digest", "evidence_digest", "bytes", "value"], "catalog export evidence");
  if (![1, 2].includes(value.catalog_export.schema_version)) throw new Error("unsupported catalog export schema");
  if (value.catalog_export.schema_version !== value.catalog_export.value?.schema_version) {
    throw new Error("catalog export envelope and value schema versions differ");
  }
  digest(value.catalog_export.content_digest, "catalog export content digest");
  digest(value.catalog_export.evidence_digest, "catalog export evidence digest");
  if (typeof value.catalog_export.bytes !== "string") throw new Error("catalog export bytes must be a string");
  let catalogExport;
  try { catalogExport = JSON.parse(value.catalog_export.bytes); } catch (error) { throw new Error(`catalog export bytes are not valid JSON: ${error.message}`); }
  if (JSON.stringify(catalogExport) !== JSON.stringify(value.catalog_export.value)) throw new Error("catalog export bytes and value differ");
  if (contentDigest(Buffer.from(value.catalog_export.bytes)) !== value.catalog_export.content_digest
    || evidenceDigest(value.catalog_export.bytes) !== value.catalog_export.evidence_digest) throw new Error("catalog export byte identity is inconsistent");
  parseCatalogExport(catalogExport);
  const rows = array(value.rows, "documentation cutover rows", { empty: true }).map(parseRow);
  const keys = rows.map((entry) => `${entry.identity}\0${entry.source_path}\0${entry.pointer}`);
  if (new Set(keys).size !== keys.length) throw new Error("documentation cutover rows must be unique");
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error("documentation cutover rows must use canonical ordering");
  exact(value.summary, ["identities", "leaves", "passed"], "documentation cutover summary");
  integer(value.summary.identities, "documentation identity count"); integer(value.summary.leaves, "documentation leaf count"); integer(value.summary.passed, "documentation passed count");
  const observedIdentities = new Set(rows.map((entry) => entry.identity));
  if (value.summary.identities !== observedIdentities.size || value.summary.leaves !== rows.length || value.summary.passed !== rows.filter((entry) => entry.result === "pass").length) throw new Error("documentation cutover summary is inconsistent");
  const result = rows.every((entry) => entry.result === "pass") ? "pass" : "fail";
  validateObservedDestinations(rows, catalogExport);
  if (value.result !== result) throw new Error("documentation cutover result is inconsistent");
  digest(value.digest, "documentation cutover digest"); const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("documentation cutover artifact digest mismatch");
  if (expected && (provenance.source_revision !== expected.source_revision || provenance.source_digest !== expected.source_digest || provenance.compiled_inventory_digest !== expected.compiled_inventory_digest || provenance.control_manifest_digest !== expected.control_manifest_digest || provenance.bundle_id !== expected.bundle_id || JSON.stringify(provenance.identities) !== JSON.stringify(expected.identities))) throw new Error("documentation cutover provenance is stale or outside the reviewed bundle");
  if (expected?.catalog_export_digest && value.catalog_export.content_digest !== expected.catalog_export_digest) throw new Error("documentation cutover catalog export is stale");
  if (expected?.source_dispositions) {
    const expectedDigest = evidenceDigest(expected.source_dispositions);
    if (provenance.source_dispositions_digest !== expectedDigest) throw new Error("documentation cutover source dispositions are stale");
    validateExpectedRows(rows, expected.source_dispositions);
  }
  return { ...value, provenance, rows };
}

function validateObservedDestinations(rows, catalogExport) {
  const documents = new Map(catalogExport.builtins.map((entry) => [String(entry.key).toLowerCase(), entry]));
  for (const row of rows) {
    if (!row.destination) continue;
    const observed = resolvePointer(documents.get(row.identity.toLowerCase()), row.destination.pointer);
    const observedDigest = observed.found ? evidenceDigest(observed.value) : null;
    if (row.destination.observed_value_digest !== observedDigest) {
      throw new Error(`${row.identity}${row.pointer}: documented destination observation differs from the captured catalog export`);
    }
  }
}

export function documentationCutoverChecks(artifact) {
  const checks = artifact.provenance.identities.map((id) => ({ id: `documentation-cutover:${id}`, result: artifact.result === "pass" ? "pass" : "fail", evidence_digest: artifact.digest }));
  for (const row of artifact.rows) if (row.destination) checks.push({ id: `destination:${row.identity}:${row.source_path}:${row.pointer}:${row.destination.kind}:${row.destination.pointer}`, result: row.result, evidence_digest: row.destination.value_digest });
  return checks;
}

function reconcileLeaf(identityName, source, leaf, document) {
  const base = { identity: identityName, source_path: source.path, source_digest: source.digest, pointer: leaf.pointer, value_digest: leaf.value_digest, disposition: leaf.disposition, reason: leaf.reason, evidence: leaf.evidence };
  if (leaf.disposition === "removed") return { ...base, destination: null, result: "pass" };
  if (leaf.destination.catalog_identity.toLowerCase() !== identityName) throw new Error(`${source.path}${leaf.pointer}: destination identity differs from its source identity`);
  validateDestinationKind(leaf.destination);
  const target = resolvePointer(document, leaf.destination.pointer);
  const observedDigest = target.found ? evidenceDigest(target.value) : null;
  return { ...base, destination: { ...leaf.destination, observed_value_digest: observedDigest }, result: observedDigest === leaf.destination.value_digest ? "pass" : "fail" };
}

function parseRow(value) {
  exact(value, ["identity", "source_path", "source_digest", "pointer", "value_digest", "disposition", "reason", "evidence", "destination", "result"], "documentation cutover row");
  identity(value.identity, "documentation row identity"); repositoryPath(value.source_path, "documentation row source path"); digest(value.source_digest, "documentation row source digest"); pointer(value.pointer, "documentation source pointer"); digest(value.value_digest, "documentation row value digest");
  enumValue(value.disposition, ["preserved", "normalized", "corrected", "removed"], "documentation row disposition"); uniqueStrings(value.evidence, "documentation row evidence", { empty: value.disposition === "preserved" });
  if (value.disposition === "removed") { if (value.destination !== null || !String(value.reason ?? "").trim()) throw new Error("removed documentation row requires an explicit reason and no destination"); }
  else { if (value.disposition === "preserved" && value.reason !== null) throw new Error("preserved documentation row cannot have a reason"); if (value.disposition !== "preserved" && !String(value.reason ?? "").trim()) throw new Error("changed documentation row requires an explicit reason"); parseDestination(value.destination); if (value.destination.catalog_identity.toLowerCase() !== value.identity.toLowerCase()) throw new Error("documentation destination identity differs from its row"); }
  enumValue(value.result, ["pass", "fail"], "documentation row result");
  const expectedResult = value.destination === null || value.destination.observed_value_digest === value.destination.value_digest ? "pass" : "fail";
  if (value.result !== expectedResult) throw new Error("documentation row result conflicts with its destination proof");
  return value;
}

function parseDestination(value) { exact(value, ["kind", "catalog_identity", "pointer", "value_digest", "observed_value_digest"], "documentation destination proof"); enumValue(value.kind, ["catalog-contract", "catalog-documentation", "catalog-evidence"], "documentation destination kind"); identity(value.catalog_identity, "documentation destination identity"); pointer(value.pointer, "documentation destination pointer"); digest(value.value_digest, "documentation destination expected digest"); if (value.observed_value_digest !== null) digest(value.observed_value_digest, "documentation destination observed digest"); }
function parseExpectedSources(value, identityName) {
  const rows = array(value, `${identityName} expected documentation sources`, { empty: true }).map((source) => {
    exact(source, ["path", "digest", "leaves"], `${identityName} expected documentation source`);
    repositoryPath(source.path, `${identityName} expected documentation path`); digest(source.digest, `${identityName} expected documentation digest`);
    const leaves = array(source.leaves, `${source.path} expected leaves`, { empty: true }).map((leaf) => {
      exact(leaf, ["pointer", "value_digest"], `${source.path} expected leaf`); pointer(leaf.pointer, `${source.path} expected pointer`); digest(leaf.value_digest, `${source.path}${leaf.pointer} expected digest`); return leaf;
    });
    const pointers = leaves.map((leaf) => leaf.pointer);
    if (new Set(pointers).size !== pointers.length) throw new Error(`${source.path}: expected documentation leaves must be unique`);
    if (JSON.stringify(pointers) !== JSON.stringify([...pointers].sort(compareCodePoint))) throw new Error(`${source.path}: expected documentation leaves must use canonical ordering`);
    return { path: source.path, digest: source.digest, leaves };
  });
  const paths = rows.map((entry) => entry.path);
  if (new Set(paths).size !== paths.length) throw new Error(`${identityName}: expected documentation sources must be unique`);
  if (JSON.stringify(paths) !== JSON.stringify([...paths].sort(compareCodePoint))) throw new Error(`${identityName}: expected documentation sources must use canonical ordering`);
  return rows;
}
function validateExpectedRows(rows, records) {
  const expected = [];
  for (const record of records) {
    const disposition = parseCompletedSourceDisposition(record.value, String(record.value?.identity ?? "").toLowerCase(), record.baseline_digest);
    for (const source of disposition.sources) for (const leaf of source.leaves) expected.push({
      identity: disposition.identity.toLowerCase(), source_path: source.path, source_digest: source.digest,
      pointer: leaf.pointer, value_digest: leaf.value_digest, disposition: leaf.disposition,
      reason: leaf.reason, evidence: leaf.evidence, destination: leaf.destination,
    });
  }
  expected.sort((left, right) => compareCodePoint(`${left.identity}\0${left.source_path}\0${left.pointer}`, `${right.identity}\0${right.source_path}\0${right.pointer}`));
  const observed = rows.map((row) => ({
    identity: row.identity, source_path: row.source_path, source_digest: row.source_digest,
    pointer: row.pointer, value_digest: row.value_digest, disposition: row.disposition,
    reason: row.reason, evidence: row.evidence,
    destination: row.destination === null ? null : {
      kind: row.destination.kind, catalog_identity: row.destination.catalog_identity,
      pointer: row.destination.pointer, value_digest: row.destination.value_digest,
    },
  }));
  if (JSON.stringify(observed) !== JSON.stringify(expected)) throw new Error("documentation cutover rows do not exactly cover the reviewed source dispositions");
}
function validateDestinationKind(value) { if (value.kind === "catalog-contract" && !value.pointer.startsWith("/catalog/")) throw new Error("catalog-contract destination must be below /catalog"); if (value.kind === "catalog-evidence" && !value.pointer.startsWith("/evidence/")) throw new Error("catalog-evidence destination must be below /evidence"); if (value.kind === "catalog-documentation" && value.pointer.startsWith("/catalog/")) throw new Error("catalog-documentation destination cannot address the catalog contract"); }
function parseCatalogExport(value) { exact(value, ["schema_version", "inventory", "builtins"], "builtin documentation export"); if (![1, 2].includes(value.schema_version)) throw new Error("unsupported builtin documentation export schema"); exact(value.inventory, ["documents", "catalog_identities", "legacy_sidecars", "missing_catalog_documentation"], "documentation export inventory"); integer(value.inventory.documents, "documentation export document count"); integer(value.inventory.catalog_identities, "documentation export catalog count"); integer(value.inventory.legacy_sidecars, "documentation export legacy count"); array(value.inventory.missing_catalog_documentation, "missing catalog documentation", { empty: true }).forEach((entry) => nonempty(entry, "missing catalog identity")); const rows = array(value.builtins, "documentation export builtins", { empty: true }); if (rows.length !== value.inventory.documents) throw new Error("documentation export inventory count differs from its rows"); const keys = rows.map((entry) => { if (!entry || typeof entry !== "object" || Array.isArray(entry)) throw new Error("documentation export row must be an object"); return identity(entry.key, "documentation export key").toLowerCase(); }); if (new Set(keys).size !== keys.length) throw new Error("documentation export keys must be unique"); if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error("documentation export keys must use canonical ordering"); }
function parseProvenance(value, artifact = false) { const fields = ["source_revision", "source_digest", "compiled_inventory_digest", "control_manifest_digest", "bundle_id", "identities"]; if (artifact) fields.push("source_dispositions_digest"); exact(value, fields, "documentation cutover provenance"); const result = { source_revision: sourceRevision(value.source_revision, "documentation source revision"), source_digest: digest(value.source_digest, "documentation source digest"), compiled_inventory_digest: digest(value.compiled_inventory_digest, "documentation compiled inventory digest"), control_manifest_digest: digest(value.control_manifest_digest, "documentation control digest"), bundle_id: stableId(value.bundle_id, "documentation bundle id"), identities: uniqueStrings(value.identities, "documentation identities", { lower: true }).sort(compareCodePoint) }; if (artifact) result.source_dispositions_digest = digest(value.source_dispositions_digest, "documentation source dispositions digest"); return result; }
function pointer(value, label) { if (typeof value !== "string" || (value !== "" && !value.startsWith("/")) || /~(?![01])/u.test(value)) throw new Error(`${label} must be an escaped JSON Pointer`); return value; }
function resolvePointer(value, encoded) { if (encoded === "") return { found: true, value }; let current = value; for (const token of encoded.slice(1).split("/").map((entry) => entry.replaceAll("~1", "/").replaceAll("~0", "~"))) { if (!current || typeof current !== "object" || !Object.prototype.hasOwnProperty.call(current, token)) return { found: false, value: undefined }; current = current[token]; } return { found: true, value: current }; }
