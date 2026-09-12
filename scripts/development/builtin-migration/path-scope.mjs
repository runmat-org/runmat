import { compareCodePoint } from "./constants.mjs";
import { array, exact, repositoryPath } from "./schema.mjs";

const EXCLUDED_FILES_FIELD = "excluded_files";

export function deriveIntegrationProductExclusions(scopes, integrationProducts) {
  if (!(integrationProducts instanceof Map)) {
    throw new Error("integration-product exclusions require the complete reviewed product registry");
  }
  const products = [...integrationProducts.values()]
    .map((product) => repositoryPath(product.path, `${product.product_id} integration product path`));
  assertNoCaseFoldCollisions(products, "integration product paths");
  return scopes.map((scope, index) => {
    const parsed = parsePathScope(scope, `authored scope ${index}`, { exclusions: false });
    const exactCollision = products.find((productPath) => pathsEqualFolded(parsed.path, productPath));
    if (exactCollision) {
      throw new Error(`${parsed.path}: authored scope overlaps integration product ${exactCollision}`);
    }
    if (parsed.kind === "file") return parsed;
    const excludedFiles = [];
    for (const productPath of products) {
      if (!startsWithPathFolded(productPath, parsed.path)) continue;
      if (!productPath.startsWith(`${parsed.path}/`)) {
        throw new Error(`${parsed.path}: authored tree and integration product ${productPath} collide case-insensitively`);
      }
      excludedFiles.push(productPath);
    }
    return excludedFiles.length === 0
      ? parsed
      : { ...parsed, [EXCLUDED_FILES_FIELD]: excludedFiles.sort(compareCodePoint) };
  });
}

export function assertDerivedIntegrationProductExclusions(scopes, integrationProducts, label) {
  for (const [index, scope] of scopes.entries()) {
    const expected = deriveIntegrationProductExclusions(
      [withoutExclusions(scope)], integrationProducts,
    )[0];
    if (JSON.stringify(scope) !== JSON.stringify(expected)) {
      throw new Error(`${label} scope ${index} exclusions are not exactly derived from globally reviewed integration products`);
    }
  }
}

export function parseEffectivePathScope(value, label) {
  return parsePathScope(value, label, { exclusions: true });
}

export function pathAllowed(scopes, candidate) {
  if (!isRepositoryPath(candidate)) return false;
  return scopes.some((scope, index) => contains(
    parsePathScope(scope, `path scope ${index}`, { exclusions: true }), candidate,
  ));
}

export function scopesOverlap(left, right) {
  const parsedLeft = parseComparableScope(left, "left path scope");
  const parsedRight = parseComparableScope(right, "right path scope");
  return contains(parsedLeft, parsedRight.path, { foldedBase: true })
    || contains(parsedRight, parsedLeft.path, { foldedBase: true });
}

function contains(scope, candidate, { foldedBase = false } = {}) {
  if (!isRepositoryPath(candidate)) return false;
  if (scope.kind === "file") {
    return foldedBase ? pathsEqualFolded(candidate, scope.path) : candidate === scope.path;
  }
  const contained = foldedBase
    ? pathsEqualFolded(candidate, scope.path) || startsWithPathFolded(candidate, scope.path)
    : candidate === scope.path || candidate.startsWith(`${scope.path}/`);
  if (!contained) return false;
  return !(scope.excluded_files ?? []).some((excluded) => pathsEqualFolded(excluded, candidate));
}

function parsePathScope(value, label, { exclusions }) {
  const fields = value?.kind === "tree" && Object.hasOwn(value ?? {}, EXCLUDED_FILES_FIELD)
    ? ["kind", "path", EXCLUDED_FILES_FIELD]
    : ["kind", "path"];
  exact(value, fields, label);
  if (!["file", "tree"].includes(value.kind)) throw new Error(`${label} kind must be one of file, tree`);
  const scopePath = repositoryPath(value.path, `${label} path`);
  if (!Object.hasOwn(value, EXCLUDED_FILES_FIELD)) return { kind: value.kind, path: scopePath };
  if (!exclusions) throw new Error(`${label} cannot declare exclusions`);
  if (value.kind !== "tree") throw new Error(`${label} exclusions require a tree scope`);
  const excludedFiles = array(value.excluded_files, `${label} excluded files`)
    .map((entry) => repositoryPath(entry, `${label} excluded file`));
  if (JSON.stringify(excludedFiles) !== JSON.stringify([...excludedFiles].sort(compareCodePoint))) {
    throw new Error(`${label} excluded files must use canonical order`);
  }
  assertNoCaseFoldCollisions(excludedFiles, `${label} excluded files`);
  for (const excluded of excludedFiles) {
    if (!startsWithPathFolded(excluded, scopePath) || pathsEqualFolded(excluded, scopePath)) {
      throw new Error(`${label} excluded file must be a strict descendant of its tree`);
    }
  }
  return { kind: "tree", path: scopePath, excluded_files: excludedFiles };
}

function parseComparableScope(value, label) {
  if (value?.kind === "file"
    && Object.hasOwn(value, "product_id")
    && Object.hasOwn(value, "producer")) {
    exact(value, ["kind", "product_id", "path", "producer"], label);
    return parsePathScope({ kind: value.kind, path: value.path }, label, { exclusions: true });
  }
  if (!Object.hasOwn(value ?? {}, "kind")
    && Object.hasOwn(value ?? {}, "product_id")
    && Object.hasOwn(value ?? {}, "producer")) {
    exact(value, ["product_id", "path", "producer"], label);
    return parsePathScope({ kind: "file", path: value.path }, label, { exclusions: true });
  }
  return parsePathScope(value, label, { exclusions: true });
}

function withoutExclusions(scope) {
  const { excluded_files: _ignored, ...base } = parsePathScope(
    scope, "effective authored scope", { exclusions: true },
  );
  return base;
}

function isRepositoryPath(value) {
  try {
    repositoryPath(value, "candidate repository path");
    return true;
  } catch {
    return false;
  }
}

function pathsEqualFolded(left, right) {
  return left.toLowerCase() === right.toLowerCase();
}

function startsWithPathFolded(candidate, parent) {
  return candidate.toLowerCase().startsWith(`${parent.toLowerCase()}/`);
}

function assertNoCaseFoldCollisions(paths, label) {
  const folded = paths.map((entry) => entry.toLowerCase());
  if (new Set(folded).size !== folded.length) {
    throw new Error(`${label} must be unique without case-fold collisions`);
  }
}
