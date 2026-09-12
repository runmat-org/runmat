import { compareCodePoint } from "./constants.mjs";
import { migrationFinding } from "./compiled-runtime-schema.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { array, digest, enumValue, exact, kind, nonempty, stableId, uniqueStrings } from "./schema.mjs";

export function migrationFindingsDigest(findings) {
  return evidenceDigest(findings);
}

export function parseFindingDispositions(value, bundles, currentFindings = null) {
  kind(value, 1, "runmat-builtin-migration-finding-dispositions", "migration finding dispositions");
  exact(value, ["schema_version", "kind", "rows", "review"], "migration finding dispositions");
  exact(value.review, ["status", "evidence"], "migration finding disposition review");
  if (value.review.status !== "reviewed") throw new Error("migration finding dispositions must be reviewed");
  uniqueStrings(value.review.evidence, "migration finding disposition review evidence");
  const rows = array(value.rows, "migration finding disposition rows", { empty: true }).map((entry) => parseRow(entry, bundles));
  const ordered = [...rows].sort((left, right) => compareCodePoint(left.finding_digest, right.finding_digest));
  if (JSON.stringify(rows) !== JSON.stringify(ordered)) throw new Error("migration finding dispositions must use canonical finding-digest ordering");
  if (new Set(rows.map((entry) => entry.finding_digest)).size !== rows.length) throw new Error("migration finding dispositions must be unique");
  if (currentFindings) {
    const expected = currentFindings.map((finding) => ({ ...finding, finding_digest: evidenceDigest(finding) })).sort((left, right) => compareCodePoint(left.finding_digest, right.finding_digest));
    if (JSON.stringify(rows.map(findingIdentity)) !== JSON.stringify(expected.map(findingIdentity))) throw new Error("migration finding dispositions do not exactly cover the compiled findings");
  }
  return rows;
}

function parseRow(value, bundles) {
  exact(value, ["finding_digest", "code", "source", "affected", "message", "disposition", "bundle_id", "reason", "evidence"], "migration finding disposition");
  digest(value.finding_digest, "migration finding digest");
  migrationFinding({ code: value.code, source: value.source, affected: value.affected, message: value.message });
  enumValue(value.disposition, ["bundle-work", "prerequisite-work", "reviewed-no-action"], "migration finding disposition");
  if (value.disposition === "reviewed-no-action") {
    if (value.bundle_id !== null) throw new Error("reviewed-no-action finding cannot name a bundle");
  } else {
    stableId(value.bundle_id, "migration finding bundle");
    if (!bundles.has(value.bundle_id)) throw new Error(`migration finding references unknown bundle ${value.bundle_id}`);
  }
  nonempty(value.reason, "migration finding disposition reason");
  uniqueStrings(value.evidence, "migration finding disposition evidence");
  return value;
}

function findingIdentity(value) {
  return { finding_digest: value.finding_digest, code: value.code, source: value.source, affected: value.affected, message: value.message };
}
