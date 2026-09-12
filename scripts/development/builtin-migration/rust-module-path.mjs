export function canonicalRustModulePath(value) {
  return value.replace(/^(?:crate|runmat_runtime)::/, "");
}

export function rustModuleIsWithinScope(modulePath, scopePath) {
  const module = canonicalRustModulePath(modulePath);
  const scope = canonicalRustModulePath(scopePath);
  return module === scope || module.startsWith(`${scope}::`);
}

export function rustModuleScopesOverlap(left, right) {
  return rustModuleIsWithinScope(left, right) || rustModuleIsWithinScope(right, left);
}
