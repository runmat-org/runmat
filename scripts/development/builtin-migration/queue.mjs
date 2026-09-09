import { compareCodePoint, sorted } from "./constants.mjs";

export function buildQueue(inventory) {
  const rows = inventory.identities.map((identity) => queueRow(identity));
  rows.sort((left, right) => right.complexity.score - left.complexity.score || compareCodePoint(left.identity, right.identity));
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-work-queue",
    authority: "development-scheduling-evidence-only",
    inventory_schema_version: inventory.schema_version,
    ordering: "complexity-score-descending-then-identity",
    weights: WEIGHTS,
    summary: summarize(rows),
    rows,
  };
}

const WEIGHTS = Object.freeze({
  unresolved_field: 3,
  additional_runtime_binding: 2,
  legacy_resolver: 3,
  provider: 3,
  fusion: 2,
  host_capability_kind: 2,
  runtime_shadow: 2,
  weak_or_missing_tests: 2,
  weak_or_missing_documentation: 2,
  file_over_500_lines: 2,
  file_over_1500_lines: 3,
  ownership_file_after_first: 1,
});

function queueRow(identity) {
  const evidence = [];
  add(evidence, "unresolved-fields", identity.unresolved.length * WEIGHTS.unresolved_field, identity.unresolved);
  add(evidence, "additional-runtime-bindings", Math.max(0, identity.registrations.runtime.length - 1) * WEIGHTS.additional_runtime_binding, identity.registrations.runtime.map((entry) => `${entry.path}:${entry.function ?? "?"}`));
  add(evidence, "legacy-resolver", identity.dependencies.legacy_resolver_paths.length * WEIGHTS.legacy_resolver, identity.dependencies.legacy_resolver_paths);
  add(evidence, "provider", identity.provider.gpu_or_wgpu_paths.length ? WEIGHTS.provider : 0, identity.provider.gpu_or_wgpu_paths);
  add(evidence, "fusion", identity.provider.fusion_paths.length ? WEIGHTS.fusion : 0, identity.provider.fusion_paths);
  add(evidence, "host-capabilities", identity.host_capability_hints.length * WEIGHTS.host_capability_kind, identity.host_capability_hints.map((entry) => entry.kind));
  add(evidence, "runtime-shadow", identity.ownership.runtime_documentation_shadows.length * WEIGHTS.runtime_shadow, identity.ownership.runtime_documentation_shadows);
  add(evidence, "test-strength", ["none", "weak"].includes(identity.tests.strength) ? WEIGHTS.weak_or_missing_tests : 0, [identity.tests.strength]);
  add(evidence, "documentation-strength", ["none", "weak"].includes(identity.documentation.strength) ? WEIGHTS.weak_or_missing_documentation : 0, [identity.documentation.strength]);
  const sizeScore = identity.source_metrics.maximum_file_lines > 1500 ? WEIGHTS.file_over_1500_lines : identity.source_metrics.maximum_file_lines > 500 ? WEIGHTS.file_over_500_lines : 0;
  add(evidence, "monolith", sizeScore, identity.source_metrics.files.filter((entry) => entry.lines > 500).map((entry) => `${entry.path}:${entry.lines}`));
  const ownershipPaths = unique([...identity.ownership.catalog, ...identity.ownership.runtime]);
  add(evidence, "ownership-spread", Math.max(0, ownershipPaths.length - 1) * WEIGHTS.ownership_file_after_first, ownershipPaths);
  const score = evidence.reduce((sum, entry) => sum + entry.points, 0);
  const familyKey = typeof identity.domain === "string" && typeof identity.family === "string" ? `${identity.domain}/${identity.family}` : "unresolved";
  return {
    identity: identity.identity,
    disposition: identity.disposition,
    domain: identity.domain,
    family: identity.family,
    family_key: familyKey,
    migration_state: migrationState(identity),
    applicable_maturity_columns: maturityColumns(identity),
    complexity: { score, class: complexityClass(score), evidence },
    write_set_collision_keys: collisionKeys(identity, familyKey),
    expected_paths: identity.expected_paths,
    ownership: identity.ownership,
    provider: identity.provider,
    host_capability_hints: identity.host_capability_hints,
    tests: identity.tests,
    documentation: identity.documentation,
    examples: identity.examples,
    unresolved: identity.unresolved,
  };
}

function migrationState(identity) {
  if (identity.disposition.kind === "unresolved") return "classification-required";
  if (identity.unresolved.length) return "evidence-review-required";
  if (identity.disposition.kind !== "canonical") return "reviewed-noncanonical";
  if (identity.ownership.catalog.length && !identity.ownership.sidecars.length && !identity.ownership.runtime_documentation_shadows.length && !identity.dependencies.legacy_resolver_paths.length) return "catalog-cutover-candidate";
  if (identity.ownership.catalog.length) return "partial-cutover";
  return "legacy";
}

function maturityColumns(identity) {
  const columns = ["identity", "disposition", "runtime-binding", "documentation", "examples", "tests"];
  if (identity.provider.gpu_or_wgpu_paths.length) columns.push("provider");
  if (identity.provider.fusion_paths.length) columns.push("fusion");
  if (identity.host_capability_hints.length) columns.push("host-capabilities");
  if (identity.registrations.native_link.catalog_contract_paths.length || identity.registrations.runtime.length) columns.push("native-link");
  if (identity.dependencies.generated_registry.length) columns.push("wasm-registry");
  return columns;
}

function collisionKeys(identity, familyKey) {
  const keys = [`identity:${identity.identity}`, `family:${familyKey}`];
  for (const sourcePath of unique([
    ...identity.ownership.catalog, ...identity.ownership.runtime, ...identity.ownership.sidecars,
    ...identity.ownership.runtime_documentation_shadows, ...identity.dependencies.legacy_resolver_paths,
    ...identity.dependencies.catalog_resolver_paths,
  ])) keys.push(`path:${sourcePath}`);
  if (identity.dependencies.generated_registry.length) keys.push("generated-registry:wasm");
  return unique(keys);
}

function summarize(rows) {
  const counts = (selector) => Object.fromEntries([...group(rows.map(selector)).entries()].sort((a, b) => compareCodePoint(a[0], b[0])));
  return {
    identities: rows.length,
    complexity_points: rows.reduce((sum, row) => sum + row.complexity.score, 0),
    by_complexity: counts((row) => row.complexity.class),
    by_migration_state: counts((row) => row.migration_state),
    by_disposition: counts((row) => row.disposition.kind),
    unresolved_identities: rows.filter((row) => row.unresolved.length).length,
  };
}

function group(values) { const result = new Map(); for (const value of values) result.set(value, (result.get(value) ?? 0) + 1); return result; }
function add(evidence, factor, points, details) { if (points > 0) evidence.push({ factor, points, details }); }
function complexityClass(score) { return score >= 16 ? "very-high" : score >= 10 ? "high" : score >= 5 ? "medium" : "low"; }
function unique(values) { return sorted(new Set(values)); }

export { WEIGHTS };
