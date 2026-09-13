import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { auditHandwrittenComposition } from "./handwritten.mjs";
import { childPathAttribute } from "./schema.mjs";

export function inspectCompositionRepository(repository, products) {
  const root = canonicalRepository(repository);
  const inventory = [];
  for (const product of products) inventory.push(inspectProduct(root, product));
  return { repository: root, products: inventory };
}

export function inspectMaterializationTargets(repository, products) {
  const root = canonicalRepository(repository);
  return products.map((product) => {
    const target = resolveRepositoryProduct(root, product.path);
    const state = lstat(target);
    if (state !== null && !state.isFile()) throw new Error(`${product.product_id}: product path is not a regular file`);
    if (state !== null && fs.realpathSync(target) !== target) throw new Error(`${product.product_id}: product path is not canonical`);
    if (state === null && product.children.length === 0) {
      return observedProductState(root, product);
    }
    validateChildStorage(root, product, null);
    return observedProductState(root, product);
  });
}

export function inspectMaterializationParentStates(repository, products) {
  const root = canonicalRepository(repository);
  return products.map((product) => observedProductState(root, product));
}

export function inspectRepositoryRegularFile(repository, sourcePath, label) {
  const root = canonicalRepository(repository);
  const target = resolveRepositoryProduct(root, sourcePath);
  const observed = readRegularNoFollow(target, label);
  return {
    file_identity: identity(observed.stat),
    content_digest: digestBytes(observed.bytes),
  };
}

export function resolveRepositoryProduct(repository, repositoryPath) {
  const root = canonicalRepository(repository);
  const target = path.resolve(root, ...repositoryPath.split("/"));
  assertWithin(root, target, repositoryPath);
  const existing = nearestExisting(target);
  const canonicalExisting = fs.realpathSync(existing);
  assertWithin(root, canonicalExisting, repositoryPath);
  if (canonicalExisting !== existing) throw new Error(`${repositoryPath} has a noncanonical repository ancestor`);
  return target;
}

export function assertProductStateUnchanged(repository, expected) {
  const root = canonicalRepository(repository);
  const observed = observedProductState(root, { product_id: expected.product_id, path: expected.path });
  const prior = Object.fromEntries(Object.keys(observed).map((key) => [key, expected[key]]));
  if (JSON.stringify(observed) !== JSON.stringify(prior)) {
    throw new Error(`${expected.product_id}: audited composition before-state changed`);
  }
}

function inspectProduct(root, product) {
  const target = resolveRepositoryProduct(root, product.path);
  const state = lstat(target);
  if (state !== null && !state.isFile()) throw new Error(`${product.product_id}: product path is not a regular file`);
  if (state === null) {
    validateChildStorage(root, product, null);
    return { ...observedProductState(root, product), children: [] };
  }
  if (fs.realpathSync(target) !== target) throw new Error(`${product.product_id}: product path is not canonical`);
  const source = readRegularNoFollow(target, `${product.product_id}: product`).bytes.toString("utf8");
  const declarations = auditHandwrittenComposition(source, product);
  validateChildStorage(root, product, declarations);
  return { ...observedProductState(root, product), children: declarations };
}

function observedProductState(root, product) {
  const target = resolveRepositoryProduct(root, product.path);
  const state = lstat(target);
  const anchor = state === null ? nearestExisting(target) : path.dirname(target);
  const result = {
    product_id: product.product_id, path: product.path,
    state: state === null ? "absent" : "present",
    anchor_path: path.relative(root, anchor) || ".",
    anchor_identity: identity(readDirectoryIdentity(anchor, `${product.product_id}: product anchor`)),
  };
  if (state === null) return result;
  if (!state.isFile() || fs.realpathSync(target) !== target) throw new Error(`${product.product_id}: product path is not a canonical regular file`);
  const observed = readRegularNoFollow(target, `${product.product_id}: product`);
  return { ...result, file_identity: identity(observed.stat), content_digest: digestBytes(observed.bytes) };
}

function validateChildStorage(root, product, declarations) {
  const represented = new Set();
  const declarationsByModule = declarations === null
    ? null
    : new Map(declarations.map((declaration) => [declaration.module, declaration]));
  for (const child of product.children) {
    const declaration = declarationsByModule?.get(child.module) ?? null;
    const target = resolveRepositoryProduct(root, child.source_path);
    const state = lstat(target);
    if (state === null || !state.isFile() || fs.realpathSync(target) !== target) {
      throw new Error(`${product.product_id}/${child.module}: reviewed ${child.source_kind} source is not a canonical regular file`);
    }
    if (child.source_kind === "directory") {
      const directory = path.dirname(target);
      if (!fs.statSync(directory).isDirectory()) throw new Error(`${product.product_id}/${child.module}: directory child has no directory`);
    }
    represented.add(path.normalize(target));
    validateCanonicalCollision(root, product, child);
    if (declaration && declaration.path_attribute !== childPathAttribute(product.path, child)) {
      throw new Error(`${product.product_id}/${child.module}: declaration path differs from review`);
    }
  }
  const parentDirectory = path.dirname(resolveRepositoryProduct(root, product.path));
  for (const candidate of directModuleCandidates(parentDirectory)) {
    if (!represented.has(candidate)) throw new Error(`${product.product_id}: unreviewed direct module source ${path.relative(root, candidate)}`);
  }
}

function validateCanonicalCollision(root, product, child) {
  const parent = path.dirname(resolveRepositoryProduct(root, product.path));
  const leaf = child.module.startsWith("r#") ? child.module.slice(2) : child.module;
  const file = path.join(parent, `${leaf}.rs`);
  const directory = path.join(parent, leaf, "mod.rs");
  if (fs.existsSync(file) && fs.existsSync(directory)) {
    throw new Error(`${product.product_id}/${child.module}: file and directory module sources collide`);
  }
}

function directModuleCandidates(directory) {
  if (!fs.existsSync(directory)) return [];
  const candidates = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    if (entry.isFile() && entry.name.endsWith(".rs") && entry.name !== "mod.rs") candidates.push(path.join(directory, entry.name));
    if (entry.isDirectory()) {
      const candidate = path.join(directory, entry.name, "mod.rs");
      if (fs.existsSync(candidate)) candidates.push(candidate);
    }
  }
  return candidates.map(path.normalize).sort();
}

function lstat(target) { try { return fs.lstatSync(target); } catch (error) { if (["ENOENT", "ENOTDIR"].includes(error?.code)) return null; throw error; } }
function nearestExisting(target) { let current = target; while (!fs.existsSync(current)) { const parent = path.dirname(current); if (parent === current) throw new Error(`no existing ancestor for ${target}`); current = parent; } return current; }
function canonicalRepository(repository) { const root = fs.realpathSync(path.resolve(repository)); if (!fs.statSync(root).isDirectory()) throw new Error("repository root must be a directory"); return root; }
function assertWithin(root, target, label) { const relative = path.relative(root, target); if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw new Error(`${label} escapes the repository root`); }
function identity(stat) { return { device: String(stat.dev), inode: String(stat.ino) }; }
function readRegularNoFollow(target, label) {
  const prior = fs.lstatSync(target);
  if (!prior.isFile()) throw new Error(`${label} is not a regular file`);
  const descriptor = fs.openSync(target, fs.constants.O_RDONLY | noFollow());
  try {
    const stat = fs.fstatSync(descriptor);
    if (!stat.isFile() || stat.dev !== prior.dev || stat.ino !== prior.ino) {
      throw new Error(`${label} changed while opening`);
    }
    return { bytes: fs.readFileSync(descriptor), stat };
  } finally { fs.closeSync(descriptor); }
}
function readDirectoryIdentity(target, label) {
  const prior = fs.lstatSync(target);
  if (!prior.isDirectory()) throw new Error(`${label} is not a directory`);
  const directoryOnly = fs.constants.O_DIRECTORY;
  if (directoryOnly === undefined) {
    const observed = fs.lstatSync(target);
    if (!observed.isDirectory() || observed.dev !== prior.dev || observed.ino !== prior.ino) {
      throw new Error(`${label} changed while observing`);
    }
    return observed;
  }
  const descriptor = fs.openSync(target, fs.constants.O_RDONLY | directoryOnly | noFollow());
  try {
    const stat = fs.fstatSync(descriptor);
    if (!stat.isDirectory() || stat.dev !== prior.dev || stat.ino !== prior.ino) {
      throw new Error(`${label} changed while opening`);
    }
    return stat;
  } finally { fs.closeSync(descriptor); }
}
function digestBytes(bytes) { return `sha256:${crypto.createHash("sha256").update(bytes).digest("hex")}`; }
function noFollow() { return fs.constants.O_NOFOLLOW ?? 0; }
