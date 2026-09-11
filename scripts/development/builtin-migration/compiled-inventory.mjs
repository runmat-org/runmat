import { contentDigest } from "./evidence.mjs";
import { compareCodePoint } from "./constants.mjs";
import { catalogEntry, catalogProvenance, constant, legacyDocumentation, legacyFunction, sortedBy, uniqueBy } from "./compiled-schema.mjs";
import { fusionSpec, gpuSpec, implementationProvenance, runtimeBinding, runtimeConstant, validateObservedOrdering, validation } from "./compiled-runtime-schema.mjs";
import { SAFE_IDENTITY, array, enumValue, exact, identity, kind, nonempty, object, uniqueStrings } from "./schema.mjs";

const KNOWN_RUNTIME_FEATURES = Object.freeze([
  "blas-lapack", "blas-only", "gui", "interaction-test-hooks", "occt-native",
  "occt-wasm-host", "plot-core", "plot-web", "test-classes", "wgpu",
]);

export function parseCompiledInventory(value) {
  kind(value, 1, "runmat-compiled-builtin-migration-inventory", "compiled migration inventory");
  exact(value, ["schema_version", "kind", "authority", "digest", "snapshot"], "compiled migration inventory");
  if (value.authority !== "derived-read-only-evidence") throw new Error("compiled migration inventory has invalid authority");
  exact(value.digest, ["algorithm", "value"], "compiled inventory digest");
  if (value.digest.algorithm !== "sha256" || !/^[a-f0-9]{64}$/.test(value.digest.value)) throw new Error("compiled inventory digest must be lowercase sha256");
  const observedDigest = contentDigest(Buffer.from(JSON.stringify(value.snapshot))).slice("sha256:".length);
  if (observedDigest !== value.digest.value) throw new Error("compiled inventory snapshot digest mismatch");
  const snapshot = object(value.snapshot, "compiled inventory snapshot");
  exact(snapshot, ["build", "declared", "observed", "validation"], "compiled inventory snapshot");
  parseBuild(snapshot.build);
  parseDeclared(snapshot.declared);
  parseObserved(snapshot.observed);
  validation(snapshot.validation);
  reconcileSnapshot(snapshot);
  const identities = compiledIdentities(snapshot);
  return { value, snapshot, identities, findings: snapshot.validation.migration_readiness.findings, digest: `sha256:${value.digest.value}` };
}

function parseBuild(value) {
  exact(value, ["architecture", "operating_system", "family", "pointer_width", "endianness", "crate_feature_inventory"], "compiled inventory build");
  exact(value.crate_feature_inventory, ["crate_name", "schema_version", "known_features", "enabled_features"], "compiled feature inventory");
  if (value.crate_feature_inventory.schema_version !== 1 || value.crate_feature_inventory.crate_name !== "runmat-runtime") throw new Error("unsupported compiled feature inventory");
  nonempty(value.architecture, "compiled architecture"); nonempty(value.operating_system, "compiled operating system"); enumValue(value.family, ["unix", "windows", "wasm"], "compiled target family"); enumValue(value.pointer_width, [16, 32, 64], "compiled pointer width"); enumValue(value.endianness, ["little", "big"], "compiled endianness");
  const known = uniqueStrings(value.crate_feature_inventory.known_features, "known runtime features", { empty: true });
  const enabled = uniqueStrings(value.crate_feature_inventory.enabled_features, "enabled runtime features", { empty: true });
  if (JSON.stringify(known) !== JSON.stringify(KNOWN_RUNTIME_FEATURES)) throw new Error("compiled known_features must exactly match the v1 runtime feature inventory");
  if (JSON.stringify([...enabled].sort()) !== JSON.stringify(enabled)) throw new Error("compiled enabled_features must use canonical ordering");
  if (enabled.some((entry) => !known.includes(entry))) throw new Error("enabled runtime feature is not in known_features");
}

function parseDeclared(value) {
  exact(value, ["namespace_scope", "catalog_schema_version", "catalog_fingerprint", "catalog_entries", "catalog_provenance", "constants", "legacy_functions", "legacy_documentation"], "compiled declared inventory");
  if (value.namespace_scope !== "function_callables_and_constants_are_reported_separately") throw new Error("compiled namespace scope is unsupported");
  if (value.catalog_schema_version !== 5) throw new Error("compiled catalog schema version must be 5"); if (!/^[a-f0-9]{64}$/.test(value.catalog_fingerprint)) throw new Error("compiled catalog fingerprint must be lowercase sha256");
  array(value.catalog_entries, "compiled catalog entries", { empty: true }).forEach(catalogEntry);
  array(value.catalog_provenance, "compiled catalog provenance", { empty: true }).forEach(catalogProvenance);
  array(value.constants, "compiled constants", { empty: true }).forEach(constant);
  array(value.legacy_functions, "compiled legacy functions", { empty: true }).forEach(legacyFunction);
  array(value.legacy_documentation, "compiled legacy documentation", { empty: true }).forEach(legacyDocumentation);
  sortedUnique(value.catalog_entries, (entry) => entry.identity.name, "catalog entries");
  sortedUnique(value.catalog_provenance, (entry) => `${entry.identity.builtin.name}\0${entry.identity.variant}`, "catalog provenance");
  sortedUnique(value.constants, (entry) => entry.name, "declared constants"); sortedUnique(value.legacy_functions, (entry) => entry.name, "legacy functions"); sortedUnique(value.legacy_documentation, (entry) => entry.name, "legacy documentation");
  const declaredBindings = value.catalog_entries.flatMap((entry) => entry.bindings.map((binding) => `${entry.identity.name}\0${binding.variant}`)).sort();
  const provenBindings = value.catalog_provenance.map((entry) => `${entry.identity.builtin.name}\0${entry.identity.variant}`).sort();
  if (JSON.stringify(declaredBindings) !== JSON.stringify(provenBindings)) throw new Error("catalog provenance must cover every declared binding exactly once");
}

function parseObserved(value) {
  exact(value, ["runtime_constants", "runtime_bindings", "implementation_provenance", "gpu_specs", "fusion_specs"], "compiled observed inventory");
  array(value.runtime_constants, "compiled runtime constants", { empty: true }).forEach(runtimeConstant);
  array(value.runtime_bindings, "compiled runtime bindings", { empty: true }).forEach(runtimeBinding);
  array(value.implementation_provenance, "compiled implementation provenance", { empty: true }).forEach(implementationProvenance);
  array(value.gpu_specs, "compiled GPU specs", { empty: true }).forEach(gpuSpec); array(value.fusion_specs, "compiled fusion specs", { empty: true }).forEach(fusionSpec);
  validateObservedOrdering(value);
}

function compiledIdentities(snapshot) {
  const result = new Set();
  const add = (value) => {
    if (typeof value !== "string" || !SAFE_IDENTITY.test(value)) return;
    result.add(value.toLowerCase());
  };
  for (const entry of snapshot.declared.catalog_entries) add(entry.identity.name);
  for (const field of ["constants", "legacy_functions", "legacy_documentation"]) for (const entry of snapshot.declared[field]) add(entry.name);
  for (const field of ["runtime_constants", "runtime_bindings", "implementation_provenance"]) for (const entry of snapshot.observed[field]) add(entry.name);
  for (const field of ["gpu_specs", "fusion_specs"]) for (const entry of snapshot.observed[field]) if (entry.owner.kind === "exact_builtin") add(entry.owner.identity.name);
  return [...result].sort();
}

function reconcileSnapshot(snapshot) {
  validateCallableSpellings(snapshot);
  const bindingKey = (name, variant) => `${name}\0${variant}`;
  const declaredBindings = new Map(snapshot.declared.catalog_entries.flatMap((entry) => entry.bindings.map((binding) => [bindingKey(entry.identity.name, binding.variant), binding])));
  const runtimeBindings = new Set(snapshot.observed.runtime_bindings.map((entry) => bindingKey(entry.name, entry.variant)));
  const canonicalProvenance = snapshot.observed.implementation_provenance.filter((entry) => entry.authority === "canonical_binding");
  const provenanceKeys = canonicalProvenance.map((entry) => bindingKey(entry.name, entry.binding_variant));
  if (new Set(provenanceKeys).size !== provenanceKeys.length) throw new Error("compiled canonical implementation provenance must be unique-one per binding");
  for (const key of runtimeBindings) {
    if (!declaredBindings.has(key)) throw new Error(`runtime binding ${key.replace("\0", "#")} has no catalog declaration`);
    if (!provenanceKeys.includes(key)) throw new Error(`runtime binding ${key.replace("\0", "#")} has no canonical implementation provenance`);
  }
  const finding = (code, identity) => snapshot.validation.migration_readiness.findings.some((entry) => entry.code === code && entry.identity === identity);
  for (const [key, declaration] of declaredBindings) if (declaration.availability === "Required" && !runtimeBindings.has(key) && !finding("missing_required_runtime_binding", key.replace("\0", "#"))) throw new Error(`required catalog binding ${key.replace("\0", "#")} is absent without a typed migration finding`);
  for (const key of provenanceKeys) if (!runtimeBindings.has(key) && !finding("missing_required_runtime_binding", key.replace("\0", "#"))) throw new Error(`canonical implementation provenance ${key.replace("\0", "#")} has no runtime binding or typed migration finding`);
  const declaredConstants = snapshot.declared.constants.map((entry) => entry.name);
  const runtimeConstants = snapshot.observed.runtime_constants.map((entry) => entry.name);
  if (JSON.stringify(declaredConstants) !== JSON.stringify(runtimeConstants)) throw new Error("declared and runtime constant identities differ");
  const callables = new Set([...snapshot.declared.catalog_entries.map((entry) => entry.identity.name), ...snapshot.declared.legacy_functions.map((entry) => entry.name)]);
  for (const collection of [snapshot.observed.gpu_specs, snapshot.observed.fusion_specs]) for (const entry of collection) if (entry.owner.kind === "exact_builtin" && !callables.has(entry.owner.identity.name)) throw new Error(`${entry.key}: exact provider owner has no callable identity`);
}

function validateCallableSpellings(snapshot) {
  const spellings = new Map();
  const names = [
    ...snapshot.declared.catalog_entries.map((entry) => entry.identity.name),
    ...snapshot.declared.legacy_functions.map((entry) => entry.name),
    ...snapshot.declared.legacy_documentation.map((entry) => entry.name),
  ];
  for (const name of names) {
    const folded = name.toLowerCase();
    const previous = spellings.get(folded);
    if (previous && previous !== name) throw new Error(`compiled inventory identity spellings collide case-insensitively: ${previous} and ${name}`);
    spellings.set(folded, name);
  }
}

export function authorityFor(compiled, identityName) {
  const id = identityName.toLowerCase();
  const named = (entries, select = (entry) => entry.name) => entries.filter((entry) => select(entry).toLowerCase() === id);
  const ownedSpecs = (entries) => entries.filter((entry) => entry.owner.kind === "exact_builtin" && entry.owner.identity.name.toLowerCase() === id);
  return {
    authority: "compiled-migration-snapshot",
    catalog_entries: named(compiled.snapshot.declared.catalog_entries, (entry) => entry.identity.name),
    catalog_provenance: named(compiled.snapshot.declared.catalog_provenance, (entry) => entry.identity.builtin.name),
    constants: named(compiled.snapshot.declared.constants),
    legacy_functions: named(compiled.snapshot.declared.legacy_functions),
    legacy_documentation: named(compiled.snapshot.declared.legacy_documentation),
    runtime_constants: named(compiled.snapshot.observed.runtime_constants),
    runtime_bindings: named(compiled.snapshot.observed.runtime_bindings),
    implementation_provenance: named(compiled.snapshot.observed.implementation_provenance),
    gpu_specs: ownedSpecs(compiled.snapshot.observed.gpu_specs), fusion_specs: ownedSpecs(compiled.snapshot.observed.fusion_specs),
    migration_findings: compiled.findings.filter((entry) => entry.identity.toLowerCase() === id),
  };
}

export function publicSpellingsFor(compiled, identityName) {
  const authority = authorityFor(compiled, identityName);
  const callable = [
    ...authority.catalog_entries.map((entry) => entry.identity.name),
    ...authority.legacy_functions.map((entry) => entry.name),
    ...authority.legacy_documentation.map((entry) => entry.name),
    ...authority.runtime_bindings.map((entry) => entry.name),
    ...authority.implementation_provenance.map((entry) => entry.name),
  ];
  const candidates = callable.length
    ? callable
    : [...authority.constants, ...authority.runtime_constants].map((entry) => entry.name);
  return [...new Set(candidates)].sort(compareCodePoint);
}

function sortedUnique(value, key, label) { sortedBy(value, key, label); uniqueBy(value, key, label); }
