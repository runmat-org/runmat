import path from "node:path";
import { CATALOG_ROOT, RUNTIME_ROOT, SHADOW_ROOT, SIDECAR_ROOT, WASM_REGISTRY } from "./constants.mjs";
import { entryMacroIdentityLiterals, filesUnder, jsonDocument, read, runtimeAttributes, runtimeMacroInvocations } from "./source-scan.mjs";

export function scanSurfaces(repository, record, diagnostics) {
  scanRuntime(repository, record, diagnostics);
  scanCatalog(repository, record);
  scanJson(repository, SIDECAR_ROOT, "sidecarPaths", record, diagnostics);
  scanJson(repository, SHADOW_ROOT, "shadowPaths", record, diagnostics);
}

export function attachWasmEvidence(repository, records) {
  let wasmSource = "";
  try { wasmSource = read(repository, WASM_REGISTRY); } catch { return; }
  for (const item of records.values()) {
    for (const binding of item.runtime) {
      if (binding.function && wasmSource.includes(`__runmat_wasm_register_builtin_${binding.function}`)) item.wasmRegistrations.add(binding.function);
    }
  }
}

function scanRuntime(repository, record, diagnostics) {
  for (const sourcePath of filesUnder(repository, RUNTIME_ROOT, ".rs")) {
    if (sourcePath === WASM_REGISTRY) continue;
    const source = read(repository, sourcePath);
    for (const attribute of runtimeAttributes(source)) {
      if (attribute.error) {
        diagnostics.push({ severity: "error", code: "malformed-runtime-registration", path: sourcePath, detail: attribute.error });
      } else if (!attribute.name) {
        if (!/\bname\s*=\s*\$[A-Za-z_]/.test(attribute.body)) diagnostics.push({ severity: "error", code: "runtime-registration-without-literal-name", path: sourcePath, detail: `byte ${attribute.offset}` });
      } else {
        addRuntime(record(attribute.name), sourcePath, source, {
          function: attribute.functionName,
          binding_variant: attribute.bindingVariant,
          declared_category: attribute.category,
          builtin_path: attribute.builtinPath,
          legacy_metadata: attribute.hasDescriptor || attribute.hasResolver || attribute.hasCapabilities,
          resolver: attribute.hasResolver,
          provenance: { kind: "literal-attribute" },
        }, attribute.name);
      }
    }
    for (const generated of runtimeMacroInvocations(source)) {
      addRuntime(record(generated.name), sourcePath, source, {
        function: generated.functionName,
        binding_variant: "default",
        declared_category: null,
        builtin_path: null,
        legacy_metadata: true,
        resolver: false,
        provenance: { kind: "macro-invocation", macro: generated.macro },
      }, generated.name);
    }
  }
}

function addRuntime(item, sourcePath, source, binding, spelling) {
  item.spellings.add(spelling);
  item.runtime.push({ path: sourcePath, ...binding });
  if (/register_gpu_spec|register_wgpu|\bGPU_SPEC\b/.test(source)) item.providerPaths.add(sourcePath);
  if (/register_fusion_spec|\bFUSION_SPEC\b/.test(source)) item.fusionPaths.add(sourcePath);
  if (/\#\[cfg\(test\)\]|\bmod\s+tests\b/.test(source)) item.testPaths.add(sourcePath);
}

function scanCatalog(repository, record) {
  for (const sourcePath of filesUnder(repository, CATALOG_ROOT, ".rs")) {
    if (/\/(?:documentation|examples|faqs)(?:\/|\.rs$)/.test(sourcePath)) continue;
    const source = read(repository, sourcePath);
    const names = catalogNames(source);
    for (const name of names) {
      const item = record(name);
      item.spellings.add(name);
      item.catalogPaths.add(sourcePath);
      if (/BuiltinInferenceRule|InferenceRule::/.test(source)) item.catalogResolverPaths.add(sourcePath);
      if (/BuiltinLinkContract|\blink:\s*/.test(source)) item.nativeLinkPaths.add(sourcePath);
      if (/\#\[cfg\(test\)\]|\bmod\s+(?:tests|inference_tests)\b/.test(source)) item.testPaths.add(sourcePath);
      attachCatalogPackage(repository, item, sourcePath);
    }
  }
}

function catalogNames(source) {
  const names = new Set();
  for (const match of source.matchAll(/BuiltinCatalogIdentity\s*\{\s*name:\s*"([^"]+)"/g)) names.add(match[1]);
  for (const name of entryMacroIdentityLiterals(source)) names.add(name);
  for (const match of source.matchAll(/\b[A-Z][A-Z0-9_]*_CATALOG_ENTRY\s*:[^=]+=[\s\S]{0,160}?\bentry\s*\(\s*"([A-Za-z][A-Za-z0-9_.]*)"/g)) names.add(match[1]);
  if (/\bpub\s+use\s+CATALOG_ENTRY\s+as\s+[A-Z][A-Z0-9_]*_CATALOG_ENTRY/.test(source)) {
    for (const match of source.matchAll(/\bdefine_[A-Za-z0-9_]+!\s*\(\s*"([A-Za-z][A-Za-z0-9_.]*)"/g)) names.add(match[1]);
  }
  return names;
}

function attachCatalogPackage(repository, item, sourcePath) {
  const relativeDirectory = path.posix.dirname(sourcePath);
  const basename = path.posix.basename(sourcePath);
  const stem = basename.replace(/\.rs$/, "");
  const documentation = filesUnder(repository, relativeDirectory, ".rs").filter((entry) => {
    if (basename === "mod.rs") return /\/(?:documentation|examples|faqs)(?:\/|\.rs$)/.test(entry);
    return entry === `${relativeDirectory}/documentation/${basename}` || entry.startsWith(`${relativeDirectory}/${stem}/documentation/`);
  });
  for (const documentationPath of documentation) {
    item.catalogDocumentationPaths.add(documentationPath);
    const source = read(repository, documentationPath);
    item.documentationEvidence.push({ path: documentationPath, kind: "catalog-source", populated_fields: null, examples: [...source.matchAll(/\bBuiltinExample\s*\{/g)].length });
    if (/\#\[cfg\(test\)\]|\bmod\s+(?:tests|inference_tests)\b/.test(source)) item.testPaths.add(documentationPath);
  }
}

function scanJson(repository, root, member, record, diagnostics) {
  for (const sourcePath of filesUnder(repository, root, ".json")) {
    const parsed = jsonDocument(repository, sourcePath);
    const filename = path.posix.basename(sourcePath, ".json");
    const declared = typeof parsed.value?.name === "string" ? parsed.value.name : filename;
    const item = record(declared);
    item.spellings.add(declared);
    item[member].add(sourcePath);
    if (declared.toLowerCase() !== filename.toLowerCase()) diagnostics.push({ severity: "error", code: "json-identity-filename-mismatch", path: sourcePath, detail: `declares ${declared}` });
    if (parsed.error) diagnostics.push({ severity: "error", code: "invalid-json", path: sourcePath, detail: parsed.error });
    else item.documentationEvidence.push({ path: sourcePath, kind: member === "sidecarPaths" ? "sidecar" : "runtime-shadow", populated_fields: Object.values(parsed.value ?? {}).filter(isPopulated).length, examples: Array.isArray(parsed.value?.examples) ? parsed.value.examples.length : 0 });
  }
}

function isPopulated(value) { return value !== null && value !== undefined && value !== "" && (!Array.isArray(value) || value.length > 0); }
