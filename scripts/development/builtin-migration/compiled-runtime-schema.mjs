import { array, boolean, enumValue, exact, identity, integer, nonempty, repositoryPath } from "./schema.mjs";
import { sortedBy, uniqueBy } from "./compiled-schema.mjs";

const PRECISIONS = ["f32", "f64", "i32", "bool"];
const CONSTANT_STRATEGIES = ["inline_literal", "uniform_buffer", "workgroup_memory"];
const MIGRATION_FINDING_CODES = ["catalog_legacy_authority_overlap", "missing_required_runtime_binding", "legacy_spec_group_requires_disposition", "placement_contract_mismatch"];

export function runtimeConstant(value) { exact(value, ["name"], "compiled runtime constant"); identity(value.name, "compiled runtime constant name"); }

export function runtimeBinding(value) {
  exact(value, ["name", "variant", "native_symbol"], "compiled runtime binding");
  identity(value.name, "compiled runtime binding name"); nonempty(value.variant, "compiled runtime binding variant");
  const expected = `runmat_builtin_binding_v1_${Buffer.from(value.name).toString("hex")}_${Buffer.from(value.variant).toString("hex")}`;
  if (value.native_symbol !== expected) throw new Error("compiled native symbol does not encode its exact binding identity");
}

export function implementationProvenance(value) {
  exact(value, ["name", "binding_variant", "source_file", "module_path", "function", "builtin_path", "authority"], "compiled implementation provenance");
  identity(value.name, "compiled provenance name");
  if (value.binding_variant !== null) nonempty(value.binding_variant, "compiled provenance binding variant");
  repositoryPath(value.source_file, "compiled provenance source file"); nonempty(value.module_path, "compiled provenance module path"); nonempty(value.function, "compiled provenance function"); nonempty(value.builtin_path, "compiled provenance builtin path");
  enumValue(value.authority, ["canonical_binding", "legacy_function"], "compiled provenance authority");
  if ((value.authority === "canonical_binding") !== (value.binding_variant !== null)) throw new Error("compiled provenance authority and binding variant disagree");
}

export function gpuSpec(value) {
  exact(value, ["key", "owner", "operation", "supported_precisions", "broadcast", "provider_hooks", "constant_strategy", "residency", "nan_mode", "two_pass_threshold", "workgroup_size", "accepts_nan_mode", "notes"], "compiled GPU spec");
  nonempty(value.key, "GPU spec key"); specOwner(value.owner, value.key, "GPU spec owner");
  if (!(value.operation === "elementwise" || value.operation === "reduction" || value.operation === "matmul" || value.operation === "transpose" || value.operation === "plot_render" || /^custom:.+/.test(value.operation))) throw new Error("GPU operation is invalid");
  enumArray(value.supported_precisions, PRECISIONS, "GPU precisions"); enumValue(value.broadcast, ["matlab", "scalar_only", "none"], "GPU broadcast");
  array(value.provider_hooks, "provider hooks", { empty: true }).forEach(providerHook); uniqueBy(value.provider_hooks, (entry) => `${entry.kind}:${entry.name}`, "provider hooks");
  enumValue(value.constant_strategy, CONSTANT_STRATEGIES, "GPU constant strategy"); enumValue(value.residency, ["inherit_inputs", "new_handle", "gather_immediately"], "GPU residency"); enumValue(value.nan_mode, ["include", "omit"], "GPU NaN mode");
  optionalInteger(value.two_pass_threshold, "two-pass threshold"); optionalInteger(value.workgroup_size, "workgroup size"); boolean(value.accepts_nan_mode, "accepts NaN mode"); string(value.notes, "GPU notes");
}

export function fusionSpec(value) {
  exact(value, ["key", "owner", "shape", "constant_strategy", "elementwise", "reduction", "emits_nan", "notes"], "compiled fusion spec");
  nonempty(value.key, "fusion spec key"); specOwner(value.owner, value.key, "fusion spec owner"); fusionShape(value.shape); enumValue(value.constant_strategy, CONSTANT_STRATEGIES, "fusion constant strategy"); optional(value.elementwise, fusionTemplate); optional(value.reduction, fusionTemplate); boolean(value.emits_nan, "fusion emits_nan"); string(value.notes, "fusion notes");
}

export function validation(value) {
  exact(value, ["status", "errors", "migration_readiness"], "compiled inventory validation");
  enumValue(value.status, ["valid", "invalid"], "compiled inventory validation status");
  array(value.errors, "compiled inventory errors", { empty: true }).forEach((entry) => { exact(entry, ["source", "identity", "message"], "compiled inventory validation error"); enumValue(entry.source, ["catalog", "catalog_provenance", "callable_identity", "runtime_binding_registry", "implementation_provenance", "gpu_spec_registry", "fusion_spec_registry", "placement_contract", "constant_registry"], "validation error source"); if (entry.identity !== null) nonempty(entry.identity, "validation error identity"); nonempty(entry.message, "validation error message"); });
  if ((value.status === "valid") !== (value.errors.length === 0)) throw new Error("compiled inventory validation status and errors disagree");
  if (value.status !== "valid") throw new Error("compiled migration inventory validation is not clean");
  exact(value.migration_readiness, ["status", "findings"], "compiled migration readiness");
  enumValue(value.migration_readiness.status, ["ready", "incomplete"], "compiled migration readiness status");
  array(value.migration_readiness.findings, "compiled migration findings", { empty: true }).forEach(migrationFinding);
  sortedBy(value.migration_readiness.findings, (entry) => `${MIGRATION_FINDING_CODES.indexOf(entry.code)}\0${entry.source}\0${entry.identity}\0${entry.message}`, "compiled migration findings");
  uniqueBy(value.migration_readiness.findings, (entry) => `${entry.code}\0${entry.source}\0${entry.identity}\0${entry.message}`, "compiled migration findings");
  if ((value.migration_readiness.status === "ready") !== (value.migration_readiness.findings.length === 0)) throw new Error("compiled migration readiness status and findings disagree");
}

export function validateObservedOrdering(observed) {
  sortedUnique(observed.runtime_constants, (entry) => entry.name, "runtime constants");
  sortedUnique(observed.runtime_bindings, (entry) => `${entry.name}\0${entry.variant}`, "runtime bindings");
  sortedUnique(observed.implementation_provenance, (entry) => `${entry.name}\0${entry.binding_variant ?? ""}\0${entry.module_path}\0${entry.function}`, "implementation provenance");
  sortedUnique(observed.gpu_specs, (entry) => entry.key, "GPU specs"); sortedUnique(observed.fusion_specs, (entry) => entry.key, "fusion specs");
  caseFoldUnique(observed.gpu_specs, "GPU specs"); caseFoldUnique(observed.fusion_specs, "fusion specs");
}

function specOwner(value, key, label) {
  if (value?.kind === "exact_builtin") {
    exact(value, ["kind", "identity"], label); exact(value.identity, ["name"], `${label} identity`); identity(value.identity.name, `${label} identity name`);
    if (value.identity.name !== key) throw new Error(`${label} exact identity must equal its key`);
  } else if (value?.kind === "legacy_group") {
    exact(value, ["kind", "raw"], label); nonempty(value.raw, `${label} raw key`);
    if (value.raw !== key) throw new Error(`${label} legacy raw key must equal its key`);
  } else throw new Error(`${label} has an unsupported kind`);
}

export function migrationFinding(value) {
  exact(value, ["code", "source", "identity", "message"], "compiled migration finding");
  enumValue(value.code, MIGRATION_FINDING_CODES, "migration finding code");
  enumValue(value.source, ["catalog", "runtime_binding_registry", "implementation_provenance", "gpu_spec_registry", "fusion_spec_registry", "placement_contract"], "migration finding source");
  nonempty(value.identity, "migration finding identity"); nonempty(value.message, "migration finding message");
}

function providerHook(value) { exact(value, ["kind", "name", "commutative"], "provider hook"); enumValue(value.kind, ["unary", "binary", "reduction", "custom"], "provider hook kind"); nonempty(value.name, "provider hook name"); if (value.commutative !== null) boolean(value.commutative, "provider hook commutative"); if ((value.kind === "binary") !== (value.commutative !== null)) throw new Error("provider hook commutativity is only valid and required for binary hooks"); }
function fusionShape(value) { exact(value, value.kind === "exact" ? ["kind", "dimensions"] : ["kind"], "fusion shape"); enumValue(value.kind, ["broadcast_compatible", "exact", "any"], "fusion shape kind"); if (value.kind === "exact") array(value.dimensions, "fusion dimensions", { empty: true }).forEach((entry) => integer(entry, "fusion dimension")); }
function fusionTemplate(value) { exact(value, ["scalar_precisions", "expression_builder_registered"], "fusion template"); enumArray(value.scalar_precisions, PRECISIONS, "fusion precisions"); boolean(value.expression_builder_registered, "fusion expression builder registration"); }
function enumArray(value, allowed, label) { const entries = array(value, label, { empty: true }); entries.forEach((entry) => enumValue(entry, allowed, label)); uniqueBy(entries, (entry) => entry, label); }
function optional(value, parser) { if (value !== null) parser(value); }
function optionalInteger(value, label) { if (value !== null) integer(value, label); }
function string(value, label) { if (typeof value !== "string") throw new Error(`${label} must be a string`); }
function sortedUnique(value, key, label) { sortedBy(value, key, label); uniqueBy(value, key, label); }
function caseFoldUnique(value, label) { uniqueBy(value, (entry) => entry.key.toLowerCase(), `${label} case-folded keys`); }
