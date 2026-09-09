import { CATALOG_ROOT, RUNTIME_ROOT, WASM_REGISTRY, compareCodePoint, rustLeaf, safeRead, sorted, unique } from "./constants.mjs";
import { lineCount, read } from "./source-scan.mjs";

export function finalizeRecords(repository, records) {
  return [...records.values()].map((item) => finalize(repository, item)).sort((a, b) => compareCodePoint(a.identity, b.identity));
}

export function inventorySummary(identities) {
  const runtime = identities.flatMap((entry) => entry.registrations.runtime.map((binding) => ({ identity: entry.identity, ...binding })));
  const provenance = Object.fromEntries([...group(runtime.map((entry) => entry.provenance.kind)).entries()].sort((a, b) => compareCodePoint(a[0], b[0])));
  return {
    identities: identities.length,
    catalog_identities: identities.filter((entry) => entry.ownership.catalog.length).length,
    runtime_name_identities: identities.filter((entry) => entry.ownership.runtime.length).length,
    runtime_binding_records: runtime.length,
    runtime_binding_provenance: provenance,
    sidecar_identities: identities.filter((entry) => entry.ownership.sidecars.length).length,
    runtime_shadow_identities: identities.filter((entry) => entry.ownership.runtime_documentation_shadows.length).length,
    wasm_registered_identities: identities.filter((entry) => entry.registrations.wasm.length).length,
    unresolved_identities: identities.filter((entry) => entry.unresolved.length).length,
    historical_runtime_identity_orientation: {
      count: 1412,
      status: "not-an-assertion",
      reason: "Historical orientation count used different runtime identity/binding provenance; compare typed sets, not the scalar.",
    },
  };
}

function finalize(repository, item) {
  const runtimePaths = unique(item.runtime.map((entry) => entry.path));
  const catalogPaths = sorted(item.catalogPaths);
  const ownershipCandidates = [...item.runtime.map(candidateFromRuntime), ...catalogPaths.map(candidateFromCatalog)].filter(Boolean);
  const domainValues = unique(ownershipCandidates.map((entry) => entry.domain).filter(Boolean));
  const familyValues = unique(ownershipCandidates.map((entry) => entry.family).filter(Boolean));
  const domain = item.input.domain ?? oneOrNull(domainValues);
  const family = item.input.family ?? oneOrNull(familyValues);
  const sourcePaths = unique([
    ...catalogPaths, ...item.catalogDocumentationPaths, ...runtimePaths, ...item.sidecarPaths, ...item.shadowPaths,
    ...item.providerPaths, ...item.fusionPaths, ...item.catalogResolverPaths, ...item.nativeLinkPaths,
  ]);
  const sourceStats = sourcePaths.map((sourcePath) => {
    const source = safeRead(repository, sourcePath, read);
    return { path: sourcePath, lines: lineCount(source), bytes: Buffer.byteLength(source) };
  });
  const disposition = resolveDisposition(item);
  const unresolved = unresolvedFields(item, disposition, domain, family, domainValues, familyValues);
  const documentation = documentationSummary(item.documentationEvidence);
  return {
    identity: item.identity,
    spellings: sorted(item.spellings),
    disposition,
    classification_input: item.input,
    domain: domain ?? unresolvedValue("No single domain is proven by ownership paths or reviewed input"),
    family: family ?? unresolvedValue("No single family is proven by ownership paths or reviewed input"),
    ownership: {
      catalog: catalogPaths,
      catalog_documentation: sorted(item.catalogDocumentationPaths),
      runtime: runtimePaths,
      sidecars: sorted(item.sidecarPaths),
      runtime_documentation_shadows: sorted(item.shadowPaths),
    },
    registrations: {
      runtime: item.runtime.sort(compareBinding),
      wasm: sorted(item.wasmRegistrations),
      native_link: {
        catalog_contract_paths: sorted(item.nativeLinkPaths),
        runtime_binding_inputs: item.runtime.map((entry) => ({ path: entry.path, function: entry.function, binding_variant: entry.binding_variant, builtin_path: entry.builtin_path, provenance: entry.provenance })).sort(compareBinding),
      },
    },
    dependencies: {
      legacy_resolver_paths: unique(item.runtime.filter((entry) => entry.resolver).map((entry) => entry.path)),
      catalog_resolver_paths: sorted(item.catalogResolverPaths),
      generated_registry: item.wasmRegistrations.size ? [WASM_REGISTRY] : [],
    },
    provider: { gpu_or_wgpu_paths: sorted(item.providerPaths), fusion_paths: sorted(item.fusionPaths) },
    host_capability_hints: detectHostMarkers(repository, runtimePaths),
    tests: { paths: sorted(item.testPaths), strength: strength(item.testPaths.size, 1, 3) },
    documentation,
    examples: { source_counts: documentation.sources.filter((entry) => entry.examples > 0).map((entry) => ({ path: entry.path, count: entry.examples })), discovered_count: documentation.sources.reduce((sum, entry) => sum + entry.examples, 0) },
    source_metrics: { files: sourceStats, total_lines: sourceStats.reduce((sum, entry) => sum + entry.lines, 0), maximum_file_lines: Math.max(0, ...sourceStats.map((entry) => entry.lines)), maximum_file_bytes: Math.max(0, ...sourceStats.map((entry) => entry.bytes)) },
    expected_paths: expectedPaths(disposition, domain, family, item.identity),
    unresolved,
  };
}

function documentationSummary(evidence) {
  const sources = evidence.sort((a, b) => compareCodePoint(a.path, b.path));
  const typed = sources.filter((entry) => entry.kind === "catalog-source").length;
  const populated = sources.reduce((sum, entry) => sum + (entry.populated_fields ?? 0), 0);
  return { sources, strength: typed ? "strong" : strength(populated, 8, 20) };
}

function unresolvedFields(item, disposition, domain, family, domains, families) {
  const result = [];
  if (disposition.kind === "unresolved") result.push("disposition");
  if (!domain) result.push("domain");
  if (!family) result.push("family");
  if (domains.length > 1 && !item.input.domain) result.push("domain-conflict");
  if (families.length > 1 && !item.input.family) result.push("family-conflict");
  if (item.spellings.size > 1) result.push("case-spelling-conflict");
  return unique(result);
}

function resolveDisposition(item) {
  if (item.input.review.status === "reviewed") return { kind: item.input.disposition, canonical: item.input.canonical, reason: item.input.reason, source: "reviewed-input" };
  if (item.catalogPaths.size) return { kind: "canonical", canonical: item.identity, reason: null, source: "catalog-entry" };
  return { kind: "unresolved", canonical: null, reason: "Runtime and documentation surfaces do not prove public, alias, or internal intent", source: null };
}

function candidateFromRuntime(entry) {
  const category = entry.declared_category?.split("/").filter(Boolean) ?? [];
  if (category.length) return { domain: category[0], family: category.slice(1).join("/") || null };
  const pathParts = entry.path.split("/");
  const builtinsIndex = pathParts.indexOf("builtins");
  if (builtinsIndex < 0) return null;
  const directories = pathParts.slice(builtinsIndex + 1, -1);
  if (pathParts.at(-1) === "mod.rs") directories.pop();
  return directories.length ? { domain: directories[0], family: directories.slice(1).join("/") || null } : null;
}

function candidateFromCatalog(sourcePath) {
  const parts = sourcePath.slice(`${CATALOG_ROOT}/`.length).split("/");
  if (parts.length < 2) return null;
  const leaf = parts.at(-1).replace(/\.rs$/, "");
  const directories = parts.slice(0, -1);
  if (leaf === "mod") directories.pop();
  else directories.push(leaf);
  return { domain: directories[0], family: directories.slice(1).join("/") || null };
}

function detectHostMarkers(repository, runtimePaths) {
  const patterns = { filesystem: /\b(?:std::fs|tokio::fs|runmat_filesystem)\b/, network: /\b(?:reqwest|TcpStream|UdpSocket|runmat_http)\b/, process: /\b(?:std::process|tokio::process|runmat_process)\b/, graphics: /\b(?:runmat_plot|PlotHandle|FigureHandle)\b/, environment: /\bstd::env\b/ };
  return Object.entries(patterns).flatMap(([kind, pattern]) => {
    const paths = runtimePaths.filter((sourcePath) => pattern.test(safeRead(repository, sourcePath, read)));
    return paths.length ? [{ kind, paths }] : [];
  });
}

function expectedPaths(disposition, domain, family, identity) {
  if (disposition.kind !== "canonical" || !domain || !family) return unresolvedValue("Canonical disposition, domain, and family are required");
  const leaf = rustLeaf(identity);
  return { catalog_package: `${CATALOG_ROOT}/${domain}/${family}/${leaf}/`, runtime_implementation: `${RUNTIME_ROOT}/${domain}/${family}/${leaf}.rs` };
}

function compareBinding(a, b) { return compareCodePoint(`${a.path}:${a.function}:${a.binding_variant}`, `${b.path}:${b.function}:${b.binding_variant}`); }
function unresolvedValue(reason) { return { status: "unresolved", reason }; }
function oneOrNull(values) { return values.length === 1 ? values[0] : null; }
function strength(value, medium, strong) { return value >= strong ? "strong" : value >= medium ? "medium" : value > 0 ? "weak" : "none"; }
function group(values) { const result = new Map(); for (const value of values) result.set(value, (result.get(value) ?? 0) + 1); return result; }
