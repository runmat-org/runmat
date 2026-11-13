import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { repositoryPath } from "../schema.mjs";
import { fileEvidence, lstat, resolve, uniqueEvidence } from "./baseline-observation-files.mjs";
import { parseHandwrittenComposition } from "./handwritten-parser.mjs";
import { companionDefinitions } from "./companions.mjs";
import { expandComposedSource } from "./composed-source.mjs";
import { moduleCompositionProductRegistry } from "./registry.mjs";
import { canonicalGitRepository, observeSignedCleanHead, verifySignedHeadFiles } from "./baseline-source-head.mjs";
import { conditionKey } from "./condition.mjs";

export function observeModuleCompositionBaseline(repository, trustedSignerFingerprint, options = {}) {
  const root = canonicalGitRepository(repository, options.git);
  const source = observeSignedCleanHead(root, trustedSignerFingerprint, options);
  const observedProducts = observeWorkingTreeModuleCompositionProducts(root);
  const workingFiles = uniqueEvidence(observedProducts.flatMap((product) => [
    product.parent_evidence, ...product.children.map((child) => child.source_evidence),
  ].filter(Boolean)));
  const files = verifySignedHeadFiles(root, source.revision, workingFiles, {
    git: options.git,
    platform: options.platform,
  });
  const signedEvidence = new Map(files.map((file) => [file.path, file]));
  const products = observedProducts.map((product) => ({
    ...product,
    parent_evidence: product.parent_evidence === null
      ? null : signedEvidence.get(product.parent_evidence.path),
    children: product.children.map((child) => ({
      ...child,
      source_evidence: signedEvidence.get(child.source_evidence.path),
    })),
  }));
  const finalSource = observeSignedCleanHead(root, trustedSignerFingerprint, options);
  if (JSON.stringify(finalSource) !== JSON.stringify(source)) {
    throw new Error("module composition source HEAD changed during observation");
  }
  return {
    registry_digest: evidenceDigest(moduleCompositionProductRegistry()),
    source: { ...source, files_digest: evidenceDigest(files) },
    products,
  };
}

export function observeWorkingTreeModuleCompositionProducts(repository) {
  return observeSelectedWorkingTreeModuleCompositionProducts(repository,
    moduleCompositionProductRegistry().map((entry) => entry.product_id));
}

export function observeSelectedWorkingTreeModuleCompositionProducts(repository, productIds) {
  const root = fs.realpathSync(repository);
  const selected = new Set(productIds);
  if (selected.size !== productIds.length) throw new Error("selected composition product ids repeat");
  const definitions = moduleCompositionProductRegistry();
  if (selected.size !== definitions.filter((entry) => selected.has(entry.product_id)).length) {
    throw new Error("selected composition product id is not registered");
  }
  return definitions.filter((entry) => selected.has(entry.product_id))
    .map((definition) => observeProduct(root, definition));
}

function observeProduct(root, definition) {
  const target = resolve(root, definition.path);
  const state = lstat(target);
  if (state === null) return {
    ...definition, aggregation_exports: [], state: "absent", parent_evidence: null, children: [],
  };
  const parentEvidence = fileEvidence(root, definition.path);
  const companionSources = new Map(companionDefinitions()
    .filter((entry) => entry.parent_id === definition.product_id)
    .filter((entry) => lstat(resolve(root, entry.path)) !== null)
    .map((entry) => [entry.path, fs.readFileSync(resolve(root, entry.path), "utf8")]));
  const composed = expandComposedSource(
    definition.product_id, fs.readFileSync(target, "utf8"), companionSources,
  );
  const parsed = parseHandwrittenComposition(composed.expanded, definition.product_id);
  validateAggregationExports(parsed.aggregation_exports, definition);
  const contributions = aggregationIndex(parsed.aggregations, definition);
  const exports = reexportIndex(parsed.reexports, definition.product_id);
  const children = parsed.declarations.map((declaration, declarationOrder) => {
    const storage = resolveChild(root, definition, declaration);
    const companion = composed.active.find((entry) => entry.module === declaration.module
      && entry.path === storage.source_path);
    return {
      module: declaration.module,
      source_kind: storage.source_kind,
      source_path: storage.source_path,
      source_evidence: fileEvidence(root, storage.source_path),
      role: null,
      visibility: declaration.visibility,
      declaration_condition: declaration.declaration_condition,
      declaration_order: declarationOrder,
      macro_use: declaration.macro_use,
      reexports: exports.get(declaration.module) ?? [],
      aggregation_sources: contributions.get(declaration.module) ?? [],
      ...(companion ? { companion_section: companion.section } : {}),
    };
  });
  for (const companion of composed.active.filter((entry) => entry.section === "declarations")) {
    children.push({
      module: companion.module, source_kind: "file", source_path: companion.path,
      source_evidence: fileEvidence(root, companion.path), role: null, visibility: "private",
      declaration_condition: { kind: "always" }, declaration_order: children.length,
      macro_use: false, reexports: [], aggregation_sources: [], companion_section: companion.section,
    });
  }
  children.sort((left, right) => compareCodePoint(left.module, right.module));
  rejectOrphans(parsed, new Set(children.map((child) => child.module)), definition.product_id);
  return { ...definition, state: "present", parent_evidence: parentEvidence, children };
}

function aggregationIndex(aggregations, definition) {
  const roles = aggregations.map((entry) => entry.role);
  const exported = new Set(definition.aggregation_exports.map((entry) => entry.role));
  const expected = definition.aggregations.filter((role) => !exported.has(role));
  if (JSON.stringify(roles) !== JSON.stringify(expected)) {
    throw new Error(`${definition.product_id}: observed aggregation roles differ from the fixed registry`);
  }
  const values = new Map();
  for (const aggregation of aggregations) for (const child of aggregation.children) {
    const rows = values.get(child.module) ?? [];
    rows.push({ role: aggregation.role, kind: child.source_kind, order: child.order, condition: child.condition });
    values.set(child.module, rows);
  }
  return values;
}

function validateAggregationExports(observed, definition) {
  if (JSON.stringify(observed) !== JSON.stringify(definition.aggregation_exports)) {
    throw new Error(`${definition.product_id}: observed aggregation exports differ from the fixed registry`);
  }
}

function reexportIndex(reexports, productId) {
  const values = new Map();
  for (const entry of reexports) {
    const rows = values.get(entry.module) ?? [];
    rows.push({
      ...entry.reexport, visibility: entry.visibility,
      condition: entry.condition, doc_hidden: entry.doc_hidden,
    });
    values.set(entry.module, rows);
  }
  for (const [module, rows] of values) {
    rows.sort((left, right) => compareCodePoint(reexportKey(left), reexportKey(right)));
    const keys = rows.map((entry) => JSON.stringify(entry));
    if (new Set(keys).size !== keys.length) throw new Error(`${productId}/${module}: duplicate reexport`);
  }
  return values;
}

function reexportKey(value) {
  const items = value.kind === "named"
    ? value.items.map((item) => `${item.name}\0${item.alias ?? ""}`).join("\0") : "";
  return `${conditionKey(value.condition)}\0${value.visibility}\0${value.doc_hidden ? 1 : 0}\0${value.kind}\0${items}`;
}

function rejectOrphans(parsed, declared, productId) {
  for (const entry of parsed.reexports) if (!declared.has(entry.module)) {
    throw new Error(`${productId}: reexport references undeclared child ${entry.module}`);
  }
  for (const aggregation of parsed.aggregations) for (const entry of aggregation.children) {
    if (!declared.has(entry.module)) throw new Error(`${productId}: aggregation references undeclared child ${entry.module}`);
  }
  for (const entry of parsed.aggregation_exports) if (!declared.has(entry.module)) {
    throw new Error(`${productId}: aggregation export references undeclared child ${entry.module}`);
  }
}

function resolveChild(root, product, declaration) {
  const parent = path.posix.dirname(product.path);
  if (declaration.path_attribute !== null) {
    const sourcePath = repositoryPath(path.posix.join(parent, declaration.path_attribute), `${product.product_id}/${declaration.module} source path`);
    const kind = sourcePath.endsWith("/mod.rs") ? "directory" : "file";
    fileEvidence(root, sourcePath);
    return { source_kind: kind, source_path: sourcePath };
  }
  const leaf = declaration.module.startsWith("r#") ? declaration.module.slice(2) : declaration.module;
  const candidates = [
    { source_kind: "file", source_path: `${parent}/${leaf}.rs` },
    { source_kind: "directory", source_path: `${parent}/${leaf}/mod.rs` },
  ].filter((entry) => lstat(resolve(root, entry.source_path)) !== null);
  if (candidates.length !== 1) throw new Error(`${product.product_id}/${declaration.module}: canonical child source must resolve exactly once`);
  fileEvidence(root, candidates[0].source_path);
  return candidates[0];
}
