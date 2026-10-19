import { compareCodePoint } from "./constants.mjs";
import { subjectAuthorityPathFailures } from "./authority-paths.mjs";
import { assertControlSubject } from "./control.mjs";
import { evidenceDigest } from "./evidence.mjs";

export function buildInventoryDeltaProof(repository, baseline, current, control, bundleId) {
  assertControlSubject(control, baseline);
  assertControlSubject(control, current);
  const bundle = control.bundles.get(bundleId);
  if (!bundle) throw new Error(`${bundleId}: inventory delta bundle is absent`);
  const failures = [];
  failures.push(...subjectAuthorityPathFailures(control, current, [bundle.id]));
  if (current.compiled_inventory.build.operating_system !== baseline.compiled_inventory.build.operating_system
      || current.compiled_inventory.build.architecture !== baseline.compiled_inventory.build.architecture) {
    failures.push("compiled target differs from the lease base");
  }
  const baselineRows = new Map(baseline.identities.map((entry) => [entry.identity, entry]));
  const currentRows = new Map(current.identities.map((entry) => [entry.identity, entry]));
  if (JSON.stringify([...baselineRows.keys()].sort(compareCodePoint)) !== JSON.stringify([...currentRows.keys()].sort(compareCodePoint))) {
    failures.push("identity set differs from the lease base");
  }
  failures.push(...changedPathFailures(baseline, current, bundle));
  failures.push(...current.diagnostics.filter((entry) => entry.severity === "error").map((entry) => `inventory diagnostic ${entry.code}:${entry.path ?? ""}`));
  failures.push(...findingFailures(baseline, current, control, bundle));

  const identities = bundle.identities.map((identity) => {
    const identityFailures = finalIdentityAuthorityFailures(identity, currentRows.get(identity), control.identities.get(identity));
    return {
      identity,
      baseline_row_digest: evidenceDigest(baselineRows.get(identity)),
      current_row_digest: currentRows.has(identity) ? evidenceDigest(currentRows.get(identity)) : null,
      result: identityFailures.length === 0 ? "pass" : "fail",
      failures: identityFailures,
    };
  });
  for (const [identity, baselineRow] of baselineRows) {
    if (bundle.identities.includes(identity)) continue;
    const currentRow = currentRows.get(identity);
    if (!currentRow || evidenceDigest(baselineRow.semantic_authority) !== evidenceDigest(currentRow.semantic_authority)) {
      failures.push(`${identity}: compiled authority changed outside the reviewed bundle`);
    }
  }
  const payload = {
    schema_version: 2,
    kind: "runmat-builtin-inventory-delta-proof",
    authority: "machine-derived-migration-evidence",
    lease_base_source_revision: baseline.source.revision,
    subject_source_revision: current.source.revision,
    lease_base_inventory_digest: baseline.digest,
    subject_inventory_digest: current.digest,
    subject_source_digest: current.source.digest,
    subject_compiled_inventory_digest: current.compiled_inventory.digest,
    control_manifest_digest: control.digest,
    bundle_id: bundle.id,
    changed_paths: changedPaths(baseline.source.files, current.source.files),
    failures: failures.sort(compareCodePoint),
    identities,
    result: failures.length === 0 && identities.every((entry) => entry.result === "pass") ? "pass" : "fail",
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function inventoryDeltaChecks(proof) {
  return proof.identities.map((entry) => ({
    id: `inventory-delta:${entry.identity}`,
    result: proof.result === "pass" && entry.result === "pass" ? "pass" : "fail",
    evidence_digest: evidenceDigest(entry),
  }));
}

function changedPathFailures(baseline, current, bundle) {
  const scopes = [...bundle.authored_write_set, ...bundle.integration_outputs.map((entry) => ({ kind: "file", path: entry.path }))];
  return changedPaths(baseline.source.files, current.source.files)
    .filter((sourcePath) => !scopes.some((scope) => withinScope(sourcePath, scope)))
    .map((sourcePath) => `${sourcePath}: source changed outside the reviewed bundle scopes`);
}

function changedPaths(baselineFiles, currentFiles) {
  const baseline = new Map(baselineFiles.map((entry) => [entry.path, `${entry.mode}:${entry.content_digest}`]));
  const current = new Map(currentFiles.map((entry) => [entry.path, `${entry.mode}:${entry.content_digest}`]));
  return [...new Set([...baseline.keys(), ...current.keys()])]
    .filter((sourcePath) => baseline.get(sourcePath) !== current.get(sourcePath))
    .sort(compareCodePoint);
}

function withinScope(sourcePath, scope) {
  return scope.kind === "file" ? sourcePath === scope.path : sourcePath === scope.path || sourcePath.startsWith(`${scope.path}/`);
}

function findingFailures(baseline, current, control, bundle) {
  const currentDigests = new Set(current.migration_findings.map(evidenceDigest));
  const reviewed = new Map(control.migrationFindings.map((entry) => [entry.finding_digest, entry]));
  const failures = [];
  for (const finding of baseline.migration_findings) {
    const digest = evidenceDigest(finding);
    const disposition = reviewed.get(digest);
    if (!disposition) {
      failures.push(`${digest}: baseline finding has no reviewed disposition`);
      continue;
    }
    const mustResolve = disposition.disposition === "bundle-work" && disposition.bundle_id === bundle.id;
    if (mustResolve && currentDigests.has(digest)) failures.push(`${digest}: bundle-owned migration finding remains unresolved`);
    if (!mustResolve && !currentDigests.has(digest)) failures.push(`${digest}: migration finding changed outside this bundle`);
  }
  const baselineDigests = new Set(baseline.migration_findings.map(evidenceDigest));
  for (const digest of currentDigests) if (!baselineDigests.has(digest)) failures.push(`${digest}: current inventory introduced an unreviewed migration finding`);
  return failures;
}

export function finalIdentityAuthorityFailures(identity, current, controlled) {
  const failures = [];
  if (!current) failures.push("current inventory row is absent");
  if (!controlled) failures.push("reviewed control row is absent");
  if (!current || !controlled) return failures;
  if (current.unresolved.length) failures.push(`current inventory remains unresolved: ${current.unresolved.join(",")}`);
  const authority = current.semantic_authority;
  if (controlled.public_identity.kind === "primary") {
    if (authority.catalog_aliases.length) failures.push("primary identity retains alias authority");
    if (authority.catalog_entries.length !== controlled.expected_authorities.catalog_entry_count) failures.push("catalog authority count differs from review");
    if (authority.constants.length !== controlled.expected_authorities.catalog_constant_count) failures.push("catalog constant authority count differs from review");
    const actualConstants = authority.runtime_constants
      .map(runtimeConstantKey).sort(compareCodePoint);
    const expectedConstants = reviewedConstantBindings(controlled)
      .map(runtimeConstantKey).sort(compareCodePoint);
    if (JSON.stringify(actualConstants) !== JSON.stringify(expectedConstants)) failures.push("runtime constant provenance differs from review");
    if (authority.legacy_functions.length) failures.push("legacy function authority remains");
    if (authority.legacy_documentation.length) failures.push("legacy documentation authority remains");
    if (current.ownership.sidecars.length) failures.push("legacy documentation sidecar remains");
    if (current.ownership.runtime_documentation_shadows.length) failures.push("runtime documentation shadow remains");
    if (current.dependencies.legacy_resolver_paths.length) failures.push("legacy resolver ownership remains");
    const actualConstantPackages = [...new Set(authority.constants
      .map((entry) => entry.provenance?.source_file).filter(Boolean))].sort(compareCodePoint);
    const expectedConstantPackages = controlled.expected_authorities.catalog_constant_package === null
      ? [] : [controlled.expected_authorities.catalog_constant_package];
    if (JSON.stringify(actualConstantPackages) !== JSON.stringify(expectedConstantPackages)) {
      failures.push("constant catalog provenance differs from review");
    }
    failures.push(...documentationAuthorityFailures(authority, current, controlled));
    failures.push(...catalogPackageFailures(authority, controlled));
    failures.push(...callableAuthorityFailures(authority, controlled));
  } else if (controlled.public_identity.kind === "alias") {
    const actualAliases = authority.catalog_aliases
      .map((entry) => `${entry.alias.name.toLowerCase()}\0${entry.canonical.name.toLowerCase()}`)
      .sort(compareCodePoint);
    const expectedAliases = [`${controlled.public_identity.alias_spelling.spelling.toLowerCase()}\0${controlled.public_identity.canonical_identity.toLowerCase()}`];
    if (JSON.stringify(actualAliases) !== JSON.stringify(expectedAliases)) failures.push("compiled catalog alias edge differs from review");
    const actualAliasPackages = authority.catalog_aliases
      .map((entry) => entry.provenance.source_file).sort(compareCodePoint);
    if (JSON.stringify(actualAliasPackages)
      !== JSON.stringify([controlled.expected_authorities.catalog_alias_package])) {
      failures.push("catalog alias provenance differs from review");
    }
    if (authority.catalog_entries.length || authority.catalog_provenance.length
      || authority.constants.length || authority.legacy_functions.length
      || authority.legacy_documentation.length || authority.runtime_bindings.length
      || authority.runtime_constants.length || authority.implementation_provenance.length
      || authority.gpu_specs.length || authority.fusion_specs.length
      || current.ownership.catalog.length || current.ownership.catalog_documentation.length
      || current.ownership.sidecars.length
      || current.ownership.runtime_documentation_shadows.length || current.dependencies.legacy_resolver_paths.length) {
      failures.push("alias retains copied authority");
    }
  } else {
    if (authority.catalog_entries.length !== controlled.expected_authorities.catalog_entry_count) {
      failures.push("catalog authority count differs from review");
    }
    if (authority.catalog_entries.some((entry) => entry.descriptor.completion_policy !== "HiddenInternal")) {
      failures.push("internal catalog entry is not hidden from public completion");
    }
    if (authority.catalog_aliases.length || authority.constants.length || authority.legacy_functions.length
      || authority.legacy_documentation.length
      || current.ownership.sidecars.length || current.ownership.runtime_documentation_shadows.length
      || current.dependencies.legacy_resolver_paths.length) {
      failures.push("internal identity retains public authority");
    }
    const actualConstants = authority.runtime_constants.map(runtimeConstantKey).sort(compareCodePoint);
    const expectedConstants = reviewedConstantBindings(controlled).map(runtimeConstantKey).sort(compareCodePoint);
    if (JSON.stringify(actualConstants) !== JSON.stringify(expectedConstants)) failures.push("runtime constant provenance differs from review");
    failures.push(...documentationAuthorityFailures(authority, current, controlled));
    failures.push(...catalogPackageFailures(authority, controlled));
    failures.push(...callableAuthorityFailures(authority, controlled));
  }
  const expectedWasm = controlled.expected_authorities.wasm_registry === "required"
    ? [...new Set(reviewedCallableBindings(controlled)
      .filter((entry) => entry.kind === "canonical_binding")
      .map((entry) => entry.function))].sort(compareCodePoint)
    : [];
  const actualWasm = [...current.registrations.wasm].sort(compareCodePoint);
  if (JSON.stringify(actualWasm) !== JSON.stringify(expectedWasm)) {
    failures.push("WASM registration set differs from review");
  }
  return failures.sort(compareCodePoint);
}

function callableAuthorityFailures(authority, controlled) {
  const failures = [];
  const actualBindings = authority.implementation_provenance
    .map(observedCallableKey)
    .sort(compareCodePoint);
  const expectedBindings = reviewedCallableBindings(controlled)
    .map(reviewedCallableKey)
    .sort(compareCodePoint);
  if (JSON.stringify(actualBindings) !== JSON.stringify(expectedBindings)) {
    failures.push("canonical runtime binding provenance differs from review");
  }
  const actualRuntimeBindings = authority.runtime_bindings
    .map((entry) => `${entry.variant}\0${entry.native_symbol}`).sort(compareCodePoint);
  const expectedRuntimeBindings = reviewedCallableBindings(controlled)
    .filter((entry) => entry.kind === "canonical_binding")
    .map((entry) => `${entry.variant}\0${entry.native_symbol}`).sort(compareCodePoint);
  if (JSON.stringify(actualRuntimeBindings) !== JSON.stringify(expectedRuntimeBindings)) {
    failures.push("runtime binding registry differs from review");
  }
  return failures;
}

function documentationAuthorityFailures(authority, current, controlled) {
  const failures = [];
  const catalogDocuments = authority.catalog_entries
    .filter((entry) => entry.documentation.authority === "Catalog"
      && entry.descriptor.completion_policy !== "HiddenInternal");
  if (controlled.expected_authorities.documentation === "catalog") {
    if (catalogDocuments.length !== authority.catalog_entries.length
      || catalogDocuments.length !== controlled.expected_authorities.catalog_entry_count) {
      failures.push("catalog documentation authority differs from review");
    }
  } else if (catalogDocuments.length || current.ownership.catalog_documentation.length) {
    failures.push("unexpected catalog documentation authority remains");
  }
  return failures;
}

function catalogPackageFailures(authority, controlled) {
  const actual = [...new Set(authority.catalog_provenance
    .map((entry) => entry.provenance.source_file))].sort(compareCodePoint);
  const expected = controlled.expected_authorities.catalog_package === null
    ? [] : [controlled.expected_authorities.catalog_package];
  return JSON.stringify(actual) === JSON.stringify(expected)
    ? [] : ["catalog package provenance differs from review"];
}

export function finalAuthorityFailuresForBundles(inventory, control, bundleIds) {
  const rows = new Map(inventory.identities.map((entry) => [entry.identity, entry]));
  return bundleIds.flatMap((bundleId) => {
    const bundle = control.bundles.get(bundleId);
    if (!bundle) return [`${bundleId}: final authority references an unknown bundle`];
    return bundle.identities.flatMap((identity) =>
      finalIdentityAuthorityFailures(identity, rows.get(identity), control.identities.get(identity))
        .map((failure) => `${identity}: ${failure}`));
  }).sort(compareCodePoint);
}

function reviewedCallableBindings(controlled) {
  const authority = controlled.implementation.callable;
  if (authority.kind !== "owned") return [];
  return authority.bindings.map((entry) => ({ ...entry, source_file: authority.owner_path }));
}

function reviewedConstantBindings(controlled) {
  const authority = controlled.implementation.constant;
  if (authority.kind !== "owned") return [];
  return authority.bindings.map((entry) => ({
    name: entry.constant, source_file: authority.owner_path, builtin_path: entry.builtin_path,
  }));
}

function runtimeConstantKey(entry) {
  return `${entry.name}\0${entry.source_file}\0${entry.builtin_path}`;
}

function observedCallableKey(entry) {
  return `${entry.authority}\0${entry.source_file}\0${entry.function}\0${entry.binding_variant ?? ""}\0${entry.builtin_path}`;
}

function reviewedCallableKey(entry) {
  return `${entry.kind}\0${entry.source_file}\0${entry.function}\0${entry.variant ?? ""}\0${entry.builtin_path}`;
}
