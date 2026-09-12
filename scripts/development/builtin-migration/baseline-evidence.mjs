import { compareCodePoint } from "./constants.mjs";
import {
  array, digest, enumValue, exact, repositoryPath, SAFE_IDENTITY, uniqueStrings,
} from "./schema.mjs";

export const BASELINE_EVIDENCE_KINDS = Object.freeze([
  "catalog-owner", "catalog-provenance", "runtime-owner", "implementation-provenance",
  "catalog-documentation", "legacy-sidecar", "runtime-documentation-shadow",
  "documentation-source", "test-source", "runtime-registration",
  "native-link-catalog-contract", "native-link-runtime-input", "legacy-resolver",
  "catalog-resolver", "provider", "fusion", "generated-registry",
]);

export function bundleBaselineEvidence(inventory, identities) {
  return bundleTypedPathEvidence(inventory, identities, { includeGeneratedRegistry: false });
}

export function bundleTypedPathEvidence(
  inventory,
  identities,
  { includeGeneratedRegistry = true } = {},
) {
  const rows = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  const sourceFiles = new Map(inventory.source.files.map((entry) => [entry.path, entry.content_digest]));
  const evidence = new Map();
  for (const identity of identities) {
    const row = rows.get(identity);
    if (!row) throw new Error(`${identity}: baseline evidence identity is absent from the inventory`);
    for (const entry of identityTypedBaselineEvidence(row, sourceFiles, {
      includeGeneratedRegistry,
    })) {
      const key = `${entry.kind}\0${entry.path}\0${entry.source_snapshot}\0${entry.content_digest ?? ""}`;
      const prior = evidence.get(key) ?? { ...entry, affected_identities: [] };
      prior.affected_identities.push(identity);
      evidence.set(key, prior);
    }
  }
  return [...evidence.values()]
    .map((entry) => ({ ...entry, affected_identities: [...new Set(entry.affected_identities)].sort(compareCodePoint) }))
    .sort(compareEvidence);
}

export function parseBundleBaselineEvidence(value, bundleId, identities) {
  const allowed = new Set(identities);
  const rows = array(value, `${bundleId} baseline evidence`, { empty: true }).map((entry) => {
    exact(entry, ["kind", "path", "source_snapshot", "content_digest", "affected_identities"], `${bundleId} baseline evidence row`);
    const kind = enumValue(entry.kind, BASELINE_EVIDENCE_KINDS, `${bundleId} baseline evidence kind`);
    const sourcePath = repositoryPath(entry.path, `${bundleId} baseline evidence path`);
    const sourceSnapshot = enumValue(
      entry.source_snapshot,
      ["present", "absent"],
      `${bundleId} baseline evidence source snapshot`,
    );
    const contentDigest = entry.content_digest === null
      ? null
      : digest(entry.content_digest, `${bundleId} baseline evidence digest`);
    if ((sourceSnapshot === "present") !== (contentDigest !== null)) {
      throw new Error(`${bundleId}: baseline evidence presence and content digest disagree`);
    }
    const affected = uniqueStrings(entry.affected_identities, `${bundleId} baseline evidence identities`, {
      pattern: SAFE_IDENTITY, lower: true,
    }).sort(compareCodePoint);
    if (!affected.length || affected.some((identity) => !allowed.has(identity))) {
      throw new Error(`${bundleId}: baseline evidence identities must be a nonempty subset of the bundle`);
    }
    return {
      kind,
      path: sourcePath,
      source_snapshot: sourceSnapshot,
      content_digest: contentDigest,
      affected_identities: affected,
    };
  });
  const keys = rows.map(evidenceKey);
  if (new Set(keys).size !== keys.length || JSON.stringify(rows) !== JSON.stringify([...rows].sort(compareEvidence))) {
    throw new Error(`${bundleId}: baseline evidence must be unique and canonically ordered`);
  }
  return rows;
}

export function parseBundleRemovals(value, bundleId, identities, baselineEvidence) {
  const allowed = new Set(identities);
  const rows = array(value, `${bundleId} expected removals`, { empty: true }).map((entry) => {
    exact(entry, ["kind", "path", "baseline_digest", "affected_identities"], `${bundleId} expected removal`);
    if (entry.kind !== "file") throw new Error(`${bundleId}: expected removal kind must be file`);
    const sourcePath = repositoryPath(entry.path, `${bundleId} expected removal path`);
    const baselineDigest = digest(entry.baseline_digest, `${bundleId} expected removal digest`);
    const affected = uniqueStrings(entry.affected_identities, `${bundleId} expected removal identities`, {
      pattern: SAFE_IDENTITY, lower: true,
    }).sort(compareCodePoint);
    if (!affected.length || affected.some((identity) => !allowed.has(identity))) {
      throw new Error(`${bundleId}: expected removal identities must be a nonempty subset of the bundle`);
    }
    const matchingEvidence = baselineEvidence.filter((candidate) =>
      candidate.path === sourcePath
      && candidate.source_snapshot === "present"
      && candidate.content_digest === baselineDigest);
    const evidencedIdentities = [...new Set(matchingEvidence.flatMap((candidate) => candidate.affected_identities))]
      .sort(compareCodePoint);
    if (JSON.stringify(affected) !== JSON.stringify(evidencedIdentities)) {
      throw new Error(`${bundleId}: expected removal ${sourcePath} does not match its complete typed baseline evidence ownership`);
    }
    return { kind: "file", path: sourcePath, baseline_digest: baselineDigest, affected_identities: affected };
  });
  const keys = rows.map((entry) => entry.path);
  if (new Set(keys).size !== keys.length || JSON.stringify(rows) !== JSON.stringify([...rows].sort(compareRemoval))) {
    throw new Error(`${bundleId}: expected removals must own unique paths in canonical order`);
  }
  return rows;
}

export function validateCompleteBundleBaselineEvidence(value, inventory, identities, bundleId) {
  const parsed = parseBundleBaselineEvidence(value, bundleId, identities);
  const expected = bundleBaselineEvidence(inventory, identities);
  if (JSON.stringify(parsed) !== JSON.stringify(expected)) {
    throw new Error(`${bundleId}: baseline evidence does not exactly preserve the inventory's typed path evidence`);
  }
  return parsed;
}

export function identityTypedBaselineEvidence(
  row,
  sourceFiles,
  { includeGeneratedRegistry = true } = {},
) {
  const records = [];
  const add = (kind, values) => {
    for (const value of values ?? []) {
      const sourcePath = typeof value === "string" ? value : value?.path;
      if (typeof sourcePath !== "string" || sourcePath.length === 0) continue;
      const source = sourceFiles.get(sourcePath) ?? null;
      const contentDigest = typeof source === "string" ? source : source?.content_digest ?? null;
      records.push({
        kind,
        path: sourcePath,
        source_snapshot: contentDigest === null ? "absent" : "present",
        content_digest: contentDigest,
      });
    }
  };
  add("catalog-owner", row.ownership?.catalog);
  add("catalog-provenance", (row.semantic_authority?.catalog_provenance ?? []).map((entry) => entry.provenance?.source_file));
  add("runtime-owner", row.ownership?.runtime);
  add("implementation-provenance", (row.semantic_authority?.implementation_provenance ?? []).map((entry) => entry.source_file));
  add("catalog-documentation", row.ownership?.catalog_documentation);
  add("legacy-sidecar", row.ownership?.sidecars);
  add("runtime-documentation-shadow", row.ownership?.runtime_documentation_shadows);
  add("documentation-source", row.documentation?.sources);
  add("test-source", row.tests?.paths);
  add("runtime-registration", (row.registrations?.runtime ?? []).map((entry) => entry.path));
  add("native-link-catalog-contract", row.registrations?.native_link?.catalog_contract_paths);
  add("native-link-runtime-input", (row.registrations?.native_link?.runtime_binding_inputs ?? []).map((entry) => entry.path));
  add("legacy-resolver", row.dependencies?.legacy_resolver_paths);
  add("catalog-resolver", row.dependencies?.catalog_resolver_paths);
  add("provider", row.provider?.gpu_or_wgpu_paths);
  add("fusion", row.provider?.fusion_paths);
  if (includeGeneratedRegistry) add("generated-registry", row.dependencies?.generated_registry);
  const unique = new Map(records.map((entry) => [evidenceKey(entry), entry]));
  return [...unique.values()].sort(compareEvidence);
}

function evidenceKey(entry) {
  return `${entry.kind}\0${entry.path}\0${entry.source_snapshot}\0${entry.content_digest ?? ""}`;
}
function compareEvidence(left, right) { return compareCodePoint(evidenceKey(left), evidenceKey(right)); }
function compareRemoval(left, right) { return compareCodePoint(left.path, right.path); }
