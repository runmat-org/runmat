import { compareCodePoint } from "./constants.mjs";
import { parseControlManifest } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { parseInventoryEvidence } from "./inventory.mjs";
import { array, digest, exact, identity, kind, nonempty, sourceRevision, uniqueStrings } from "./schema.mjs";

const COHORTS = Object.freeze([
  ["C00", "prerequisite"], ["C01", "A"], ["C02", "B"], ["C03", "C"],
  ["C04", "D"], ["C05", "E"], ["C06", "F"], ["C07", "G"],
]);

const UNRESOLVED_CONTROL_FIELDS = Object.freeze([
  "bundles", "exception_manifest", "gate_plans", "identity_controls",
  "migration_finding_dispositions", "storage_policy",
]);

export function buildControlDraft(inventoryValue) {
  const inventory = parseInventoryEvidence(inventoryValue);
  const identityRows = inventory.identities.map((entry) => ({
    identity: entry.identity,
    inventory_row_digest: evidenceDigest(entry),
    compiled_authority_digest: evidenceDigest(entry.semantic_authority),
    discovery_observation_digest: evidenceDigest(entry.lexical_observations),
    unresolved_fields: [
      "public_spelling", "disposition", "cohort", "bundle_id", "domain", "family",
      "runtime_owner", "shared_dependencies", "complexity", "maturity",
      "expected_authorities", "expected_removals", "baseline_evidence", "owner",
    ],
    review: { status: "unreviewed", evidence: [] },
  })).sort((left, right) => compareCodePoint(left.identity, right.identity));
  const findingRows = inventory.migration_findings.map((finding) => ({
    finding_digest: evidenceDigest(finding),
    disposition: null,
    bundle_id: null,
    reason: null,
    evidence: [],
    review: { status: "unreviewed", evidence: [] },
  })).sort((left, right) => compareCodePoint(left.finding_digest, right.finding_digest));
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-draft",
    authority: "unreviewed-scaffold-only",
    program: "RM-1064/C00-C07",
    baseline: baselineFromInventory(inventory),
    cohorts: COHORTS.map(([id, semantic], order) => ({ id, semantic, order })),
    identity_rows: identityRows,
    bundle_drafts: [],
    migration_finding_rows: findingRows,
    unresolved_control_fields: [...UNRESOLVED_CONTROL_FIELDS],
    review: { status: "unreviewed", evidence: [] },
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseControlDraft(value, inventoryValue = null) {
  kind(value, 1, "runmat-builtin-migration-control-draft", "control draft");
  exact(value, ["schema_version", "kind", "authority", "program", "baseline", "cohorts", "identity_rows", "bundle_drafts", "migration_finding_rows", "unresolved_control_fields", "review", "digest"], "control draft");
  if (value.authority !== "unreviewed-scaffold-only" || value.program !== "RM-1064/C00-C07") throw new Error("control draft must remain an unreviewed RM-1064 scaffold");
  parseDraftBaseline(value.baseline);
  parseCohorts(value.cohorts);
  const identities = array(value.identity_rows, "control draft identity rows").map(parseIdentityRow);
  canonicalUnique(identities, (entry) => entry.identity, "control draft identity rows");
  if (array(value.bundle_drafts, "control draft bundles", { empty: true }).length !== 0) throw new Error("control draft cannot infer or pre-populate bundles");
  const findings = array(value.migration_finding_rows, "control draft migration finding rows", { empty: true }).map(parseFindingRow);
  canonicalUnique(findings, (entry) => entry.finding_digest, "control draft migration finding rows");
  const unresolved = uniqueStrings(value.unresolved_control_fields, "control draft unresolved fields");
  if (JSON.stringify(unresolved) !== JSON.stringify(UNRESOLVED_CONTROL_FIELDS)) throw new Error("control draft must retain every global review prompt in canonical order");
  exact(value.review, ["status", "evidence"], "control draft review");
  if (value.review.status !== "unreviewed" || array(value.review.evidence, "control draft review evidence", { empty: true }).length !== 0) throw new Error("control draft cannot claim review");
  digest(value.digest, "control draft digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("control draft digest mismatch");
  if (inventoryValue) validateAgainstInventory(value, parseInventoryEvidence(inventoryValue));
  return value;
}

export function freezeReviewedControl(draftValue, inventoryValue, reviewedTopology, reviewedControl) {
  if (!reviewedTopology || !reviewedControl) throw new Error("reviewed control freeze requires the deterministically validated topology and control review chain");
  const draft = parseControlDraft(draftValue, inventoryValue);
  const inventory = parseInventoryEvidence(inventoryValue);
  if (reviewedTopology.baseline?.control_draft_digest !== draft.digest) {
    throw new Error("reviewed topology does not bind the exact unreviewed control draft");
  }
  const parsed = parseControlManifest(reviewedControl.controlValue, { inventory, reviewedTopology, reviewedControl });
  const draftIdentities = draft.identity_rows.map((entry) => entry.identity);
  if (JSON.stringify([...parsed.identities.keys()].sort(compareCodePoint)) !== JSON.stringify(draftIdentities)) throw new Error("reviewed control identities differ from the frozen draft");
  return parsed;
}

function baselineFromInventory(inventory) {
  return {
    revision: inventory.source.revision,
    source_digest: inventory.source.digest,
    inventory_digest: inventory.digest,
    dispositions_digest: inventory.dispositions_digest,
    migration_findings_digest: inventory.migration_findings_digest,
    compiled_target: {
      operating_system: inventory.compiled_inventory.build.operating_system,
      architecture: inventory.compiled_inventory.build.architecture,
    },
  };
}

function parseDraftBaseline(value) {
  exact(value, ["revision", "source_digest", "inventory_digest", "dispositions_digest", "migration_findings_digest", "compiled_target"], "control draft baseline");
  sourceRevision(value.revision, "control draft revision");
  for (const field of ["source_digest", "inventory_digest", "dispositions_digest", "migration_findings_digest"]) digest(value[field], `control draft ${field}`);
  exact(value.compiled_target, ["operating_system", "architecture"], "control draft target");
  nonempty(value.compiled_target.operating_system, "control draft operating system");
  nonempty(value.compiled_target.architecture, "control draft architecture");
}

function parseCohorts(value) {
  const rows = array(value, "control draft cohorts");
  if (JSON.stringify(rows) !== JSON.stringify(COHORTS.map(([id, semantic], order) => ({ id, semantic, order })))) throw new Error("control draft cohorts must be the fixed C00-C07 sequence");
}

function parseIdentityRow(value) {
  exact(value, ["identity", "inventory_row_digest", "compiled_authority_digest", "discovery_observation_digest", "unresolved_fields", "review"], "control draft identity row");
  identity(value.identity, "control draft identity");
  digest(value.inventory_row_digest, "control draft inventory row digest");
  digest(value.compiled_authority_digest, "control draft compiled authority digest");
  digest(value.discovery_observation_digest, "control draft discovery observation digest");
  const expected = ["public_spelling", "disposition", "cohort", "bundle_id", "domain", "family", "runtime_owner", "shared_dependencies", "complexity", "maturity", "expected_authorities", "expected_removals", "baseline_evidence", "owner"];
  if (JSON.stringify(uniqueStrings(value.unresolved_fields, `${value.identity} unresolved fields`)) !== JSON.stringify(expected)) throw new Error(`${value.identity}: control draft must leave every identity field unresolved`);
  assertUnreviewed(value.review, `${value.identity} draft review`);
  return value;
}

function parseFindingRow(value) {
  exact(value, ["finding_digest", "disposition", "bundle_id", "reason", "evidence", "review"], "control draft finding row");
  digest(value.finding_digest, "control draft finding digest");
  if (value.disposition !== null || value.bundle_id !== null || value.reason !== null || array(value.evidence, "control draft finding evidence", { empty: true }).length !== 0) throw new Error("control draft cannot infer a migration-finding disposition");
  assertUnreviewed(value.review, "control draft finding review");
  return value;
}

function assertUnreviewed(value, label) {
  exact(value, ["status", "evidence"], label);
  if (value.status !== "unreviewed" || array(value.evidence, `${label} evidence`, { empty: true }).length !== 0) throw new Error(`${label} cannot claim review`);
}

function validateAgainstInventory(draft, inventory) {
  if (JSON.stringify(draft.baseline) !== JSON.stringify(baselineFromInventory(inventory))) throw new Error("control draft baseline differs from the inventory");
  const expectedIdentities = inventory.identities.map((entry) => entry.identity).sort(compareCodePoint);
  if (JSON.stringify(draft.identity_rows.map((entry) => entry.identity)) !== JSON.stringify(expectedIdentities)) throw new Error("control draft identities do not exactly cover the inventory");
  const inventoryRows = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  for (const row of draft.identity_rows) {
    const entry = inventoryRows.get(row.identity);
    if (row.inventory_row_digest !== evidenceDigest(entry) || row.compiled_authority_digest !== evidenceDigest(entry.semantic_authority) || row.discovery_observation_digest !== evidenceDigest(entry.lexical_observations)) throw new Error(`${row.identity}: control draft observations differ from the inventory`);
  }
  const findings = inventory.migration_findings.map((entry) => evidenceDigest(entry)).sort(compareCodePoint);
  if (JSON.stringify(draft.migration_finding_rows.map((entry) => entry.finding_digest)) !== JSON.stringify(findings)) throw new Error("control draft findings do not exactly cover the inventory");
}

function canonicalUnique(values, key, label) {
  const keys = values.map(key);
  if (new Set(keys.map((entry) => entry.toLowerCase())).size !== keys.length) throw new Error(`${label} must be unique ignoring case`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) throw new Error(`${label} must use canonical order`);
}
