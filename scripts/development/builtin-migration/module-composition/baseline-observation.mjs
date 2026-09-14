import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { repositoryPath } from "../schema.mjs";
import { parseHandwrittenComposition } from "./handwritten-parser.mjs";
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
  const root = fs.realpathSync(repository);
  return moduleCompositionProductRegistry().map((definition) => observeProduct(root, definition));
}

function observeProduct(root, definition) {
  const target = resolve(root, definition.path);
  const state = lstat(target);
  if (state === null) return {
    ...definition, aggregation_exports: [], state: "absent", parent_evidence: null, children: [],
  };
  const parentEvidence = fileEvidence(root, definition.path);
  const parsed = parseHandwrittenComposition(fs.readFileSync(target, "utf8"), definition.product_id);
  validateAggregationExports(parsed.aggregation_exports, definition);
  const contributions = aggregationIndex(parsed.aggregations, definition);
  const exports = reexportIndex(parsed.reexports, definition.product_id);
  const children = parsed.declarations.map((declaration, declarationOrder) => {
    const storage = resolveChild(root, definition, declaration);
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
    };
  }).sort((left, right) => compareCodePoint(left.module, right.module));
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

function fileEvidence(root, sourcePath) {
  const target = resolve(root, sourcePath);
  const stat = fs.lstatSync(target);
  if (!stat.isFile() || fs.realpathSync(target) !== target) throw new Error(`${sourcePath}: source must be a canonical regular file`);
  return {
    path: sourcePath,
    working_executable: Boolean(stat.mode & 0o111),
    content_digest: contentDigest(fs.readFileSync(target)),
  };
}

function resolve(root, sourcePath) {
  const target = path.resolve(root, ...repositoryPath(sourcePath, "module composition source path").split("/"));
  const relative = path.relative(root, target);
  if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw new Error(`${sourcePath}: source escapes repository`);
  const existing = nearestExisting(target);
  const canonical = fs.realpathSync(existing);
  const canonicalRelative = path.relative(root, canonical);
  if (canonicalRelative === ".." || canonicalRelative.startsWith(`..${path.sep}`) || path.isAbsolute(canonicalRelative)) {
    throw new Error(`${sourcePath}: source escapes repository through a symbolic link`);
  }
  if (canonical !== existing) throw new Error(`${sourcePath}: source has a noncanonical repository ancestor`);
  return target;
}

function uniqueEvidence(values) {
  const byPath = new Map();
  for (const value of values) {
    const prior = byPath.get(value.path);
    if (prior && JSON.stringify(prior) !== JSON.stringify(value)) throw new Error(`${value.path}: conflicting source evidence`);
    byPath.set(value.path, value);
  }
  return [...byPath.values()].sort((left, right) => compareCodePoint(left.path, right.path));
}
function lstat(target) { try { return fs.lstatSync(target); } catch (error) { if (["ENOENT", "ENOTDIR"].includes(error?.code)) return null; throw error; } }
function nearestExisting(target) { let current = target; while (!fs.existsSync(current)) { const parent = path.dirname(current); if (parent === current) throw new Error(`no existing ancestor for ${target}`); current = parent; } return current; }
