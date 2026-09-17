import fs from "node:fs";
import path from "node:path";

import { compareCodePoint, sorted } from "./constants.mjs";
import { contentDigest, evidenceDigest } from "./evidence.mjs";
import { array, digest, enumValue, exact, identity, kind, nonempty, repositoryPath, uniqueStrings } from "./schema.mjs";

export function buildSourceFieldDisposition(repository, row) {
  const sources = sorted([...row.ownership.sidecars, ...row.ownership.runtime_documentation_shadows]).map((sourcePath) => {
    const bytes = fs.readFileSync(path.join(repository, sourcePath));
    const value = JSON.parse(bytes.toString("utf8"));
    return {
      path: sourcePath,
      digest: contentDigest(bytes),
      leaves: canonicalLeaves(value).map((leaf) => ({
        pointer: leaf.pointer,
        value_digest: evidenceDigest(leaf.value),
        disposition: "pending",
        destination: null,
        reason: null,
        evidence: [],
      })),
    };
  });
  return {
    schema_version: 2,
    kind: "runmat-builtin-source-field-disposition",
    authority: "review-workspace-only",
    identity: row.identity,
    sources,
    review: { status: "unreviewed", evidence: [] },
  };
}

export function sourceFieldBaselineDigest(value) {
  return evidenceDigest({
    identity: value.identity,
    sources: value.sources.map((source) => ({
      path: source.path,
      digest: source.digest,
      leaves: source.leaves.map((leaf) => ({ pointer: leaf.pointer, value_digest: leaf.value_digest })),
    })),
  });
}

export function sourceFieldBaselineSource(sourcePath, bytes, expectedDigest = null) {
  repositoryPath(sourcePath, "source-field baseline path");
  const encoded = Buffer.from(bytes);
  const observedDigest = contentDigest(encoded);
  if (expectedDigest !== null && observedDigest !== expectedDigest) throw new Error(`${sourcePath}: baseline documentation bytes differ from the frozen source inventory`);
  let value;
  try { value = JSON.parse(encoded.toString("utf8")); }
  catch (error) { throw new Error(`${sourcePath}: baseline documentation is not valid JSON: ${error instanceof Error ? error.message : String(error)}`); }
  return {
    path: sourcePath,
    digest: observedDigest,
    leaves: canonicalLeaves(value).map((leaf) => ({ pointer: leaf.pointer, value_digest: evidenceDigest(leaf.value) })),
  };
}

export function parseCompletedSourceDisposition(value, expectedIdentity, expectedBaselineDigest) {
  kind(value, 2, "runmat-builtin-source-field-disposition", "source-field disposition");
  exact(value, ["schema_version", "kind", "authority", "identity", "sources", "review"], "source-field disposition");
  if (value.authority !== "review-workspace-only") throw new Error("source-field disposition has invalid authority");
  if (identity(value.identity, "source-field identity").toLowerCase() !== expectedIdentity) throw new Error("source-field disposition identity mismatch");
  exact(value.review, ["status", "evidence"], "source-field review");
  if (value.review.status !== "reviewed") throw new Error("source-field disposition is not reviewed");
  uniqueStrings(value.review.evidence, "source-field review evidence");
  const observed = new Set();
  for (const source of array(value.sources, "source-field sources", { empty: true })) {
    exact(source, ["path", "digest", "leaves"], "source-field source");
    repositoryPath(source.path, "source-field source path");
    digest(source.digest, "source-field source digest");
    if (observed.has(source.path)) throw new Error(`duplicate source-field source ${source.path}`);
    observed.add(source.path);
    const actualPointers = new Set();
    const leafRows = array(source.leaves, `${source.path} leaves`, { empty: true });
    for (const leaf of leafRows) {
      parseLeaf(leaf, source.path);
      if (actualPointers.has(leaf.pointer)) throw new Error(`duplicate source-field leaf at ${source.path}${leaf.pointer}`);
      actualPointers.add(leaf.pointer);
    }
    const pointers = leafRows.map((leaf) => leaf.pointer);
    if (JSON.stringify(pointers) !== JSON.stringify([...pointers].sort(compareCodePoint))) throw new Error(`${source.path}: source-field leaves must use canonical ordering`);
  }
  if (sourceFieldBaselineDigest(value) !== expectedBaselineDigest) {
    throw new Error("source-field leaves do not match the prepared baseline inventory");
  }
  return value;
}

function parseLeaf(value, sourcePath) {
  exact(value, ["pointer", "value_digest", "disposition", "destination", "reason", "evidence"], `${sourcePath} leaf`);
  validatePointer(value.pointer, `${sourcePath} leaf`);
  digest(value.value_digest, `${sourcePath}${value.pointer} value digest`);
  enumValue(value.disposition, ["preserved", "normalized", "corrected", "removed"], `${sourcePath}${value.pointer} disposition`);
  const reviewedChange = value.disposition !== "preserved";
  if (value.disposition === "removed") {
    if (value.destination !== null) throw new Error(`${sourcePath}${value.pointer}: removed leaf cannot have a destination`);
  } else {
    exact(value.destination, ["kind", "catalog_identity", "pointer", "value_digest"], `${sourcePath}${value.pointer} destination`);
    enumValue(value.destination.kind, ["catalog-contract", "catalog-documentation", "catalog-evidence"], `${sourcePath}${value.pointer} destination kind`);
    identity(value.destination.catalog_identity, `${sourcePath}${value.pointer} destination identity`);
    validatePointer(value.destination.pointer, `${sourcePath}${value.pointer} destination`, false);
    digest(value.destination.value_digest, `${sourcePath}${value.pointer} destination value digest`);
    if (value.disposition === "preserved" && value.destination.value_digest !== value.value_digest) throw new Error(`${sourcePath}${value.pointer}: preserved destination digest must equal the source value digest`);
  }
  uniqueStrings(value.evidence, `${sourcePath}${value.pointer} evidence`, { empty: !reviewedChange });
  if (reviewedChange && (!String(value.reason ?? "").trim() || value.evidence.length === 0)) throw new Error(`${sourcePath}${value.pointer}: normalized, corrected, or removed leaves require reason and evidence`);
  if (!reviewedChange && value.reason !== null) throw new Error(`${sourcePath}${value.pointer}: preserved leaf cannot have a reason`);
}

function enumerateLeaves(value, pointer = "") {
  if (Array.isArray(value)) {
    if (!value.length) return [{ pointer, value }];
    return value.flatMap((entry, index) => enumerateLeaves(entry, `${pointer}/${index}`));
  }
  if (value && typeof value === "object") {
    const keys = Object.keys(value).sort(compareCodePoint);
    if (!keys.length) return [{ pointer, value }];
    return keys.flatMap((key) => enumerateLeaves(value[key], `${pointer}/${escapePointer(key)}`));
  }
  return [{ pointer, value }];
}

function canonicalLeaves(value) {
  return enumerateLeaves(value).sort((left, right) => compareCodePoint(left.pointer, right.pointer));
}

function escapePointer(value) { return value.replaceAll("~", "~0").replaceAll("/", "~1"); }
function validatePointer(value, label, allowRoot = true) {
  if (typeof value !== "string" || (!allowRoot && value === "") || (value !== "" && !value.startsWith("/")) || /~(?![01])/u.test(value)) throw new Error(`${label} must be an escaped JSON Pointer`);
}
