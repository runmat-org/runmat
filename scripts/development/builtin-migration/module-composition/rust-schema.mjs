import path from "node:path";

// Strict and reserved keywords require raw syntax when Rust permits them as an
// identifier. Weak keywords such as `union` remain ordinary identifiers in
// module paths and item names, so canonical projections must not prefix them.
const RAW_REQUIRED_KEYWORDS = new Set([
  "Self", "abstract", "as", "async", "await", "become", "box", "break", "const",
  "continue", "crate", "do", "dyn", "else", "enum", "extern", "false", "final",
  "fn", "for", "gen", "if", "impl", "in", "let", "loop", "macro", "match",
  "mod", "move", "mut", "override", "priv", "pub", "ref", "return", "self",
  "static", "struct", "super", "trait", "true", "try", "type", "typeof",
  "unsafe", "unsized", "use", "virtual", "where", "while", "yield",
]);
const UNRAWABLE_IDENTIFIERS = new Set(["Self", "_", "crate", "self", "super"]);

export function rustModuleIdentifier(value, label) {
  if (typeof value !== "string" || !/^(?:r#)?[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new Error(`${label} must be a canonical Rust module identifier`);
  }
  const raw = value.startsWith("r#");
  const bare = raw ? value.slice(2) : value;
  if (UNRAWABLE_IDENTIFIERS.has(bare)) {
    throw new Error(`${label} uses an identifier Rust reserves in module paths`);
  }
  if (raw !== RAW_REQUIRED_KEYWORDS.has(bare)) {
    throw new Error(`${label} must use r# exactly for a Rust keyword`);
  }
  return value;
}

export function rustItemIdentifier(value, label) {
  if (typeof value !== "string" || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)
    || RAW_REQUIRED_KEYWORDS.has(value)) {
    throw new Error(`${label} must be a non-keyword Rust item identifier`);
  }
  return value;
}

export function rustParentModule(value, label) {
  if (typeof value !== "string" || !value.startsWith("crate::")) {
    throw new Error(`${label} must start with crate::`);
  }
  value.split("::").slice(1).forEach((entry) => rustModuleIdentifier(entry, label));
  return value;
}

export function validateParentPath(crateRole, productPath, modulePath, id) {
  const root = crateRole === "catalog" ? "crates/runmat-builtins/src" : "crates/runmat-runtime/src";
  const requiredRoot = crateRole === "catalog" ? "crate::catalog" : "crate::builtins";
  if (!(modulePath === requiredRoot || modulePath.startsWith(`${requiredRoot}::`))) {
    throw new Error(`${id}: parent module is outside its crate role`);
  }
  const relative = modulePath.split("::").slice(1).map(stripRaw).join("/");
  if (productPath !== `${root}/${relative}/mod.rs`) {
    throw new Error(`${id}: product path does not match its logical parent module`);
  }
}

export function validateChildPath(sourcePath, sourceKind, module, productPath) {
  const directory = path.posix.dirname(productPath);
  if (sourcePath === productPath || !sourcePath.startsWith(`${directory}/`)) {
    throw new Error(`${module}: child source is outside its reviewed parent`);
  }
  const relative = sourcePath.slice(directory.length + 1);
  const invalid = sourceKind === "file"
    ? !relative.endsWith(".rs") || relative.endsWith("/mod.rs")
    : !relative.endsWith("/mod.rs");
  if (invalid) throw new Error(`${module}: child source does not match its declared source kind`);
}

export function canonicalChildSourcePath(productPath, child) {
  const directory = path.posix.dirname(productPath);
  const leaf = stripRaw(child.module);
  return child.source_kind === "file" ? `${directory}/${leaf}.rs` : `${directory}/${leaf}/mod.rs`;
}

export function childPathAttribute(productPath, child) {
  if (child.source_path === canonicalChildSourcePath(productPath, child)) return null;
  return path.posix.relative(path.posix.dirname(productPath), child.source_path);
}

export function declaredChildSourcePath(productPath, child, pathAttribute) {
  if (pathAttribute === null) return canonicalChildSourcePath(productPath, child);
  if (path.posix.isAbsolute(pathAttribute)) return path.posix.normalize(pathAttribute);
  return path.posix.normalize(path.posix.join(path.posix.dirname(productPath), pathAttribute));
}

function stripRaw(value) { return value.startsWith("r#") ? value.slice(2) : value; }
