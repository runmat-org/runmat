const { hasSerde, serde, splitTopLevel } = require("./syntax.cjs");

const TERMINALS = new Map([
  ["bool", "boolean"],
  ["String", "string"],
  ["str", "string"],
  ["f32", "number"], ["f64", "number"],
  ["i8", "number"], ["i16", "number"], ["i32", "number"], ["i64", "number"], ["isize", "number"],
  ["u8", "number"], ["u16", "number"], ["u32", "number"], ["u64", "number"], ["usize", "number"],
  ["LogicalSourcePath", "string"], ["SourceSymbol", "string"], ["ExecutionProfileIdentity", "string"],
  ["Span", "{ start: number; end: number }"],
]);

function renderReachable(registry, roots) {
  const reachable = new Set();
  const visiting = new Set();
  const visit = (name) => {
    if (reachable.has(name) || TERMINALS.has(name)) return;
    const declaration = registry.get(name);
    if (!declaration) throw new Error(`Unresolved serialized Rust type ${name}`);
    if (declaration.conflicts?.length) {
      throw new Error(`Ambiguous serialized Rust type ${name}: ${[declaration.file, ...declaration.conflicts].join(", ")}`);
    }
    if (visiting.has(name)) return;
    visiting.add(name);
    for (const dependency of dependencies(declaration, registry)) visit(dependency);
    visiting.delete(name);
    reachable.add(name);
  };
  roots.forEach(visit);
  return [...reachable].sort().map((name) => renderDeclaration(registry.get(name), registry)).join("\n\n");
}

function dependencies(declaration, registry) {
  const identifiers = declaration.body.match(/\b[A-Z][A-Za-z0-9_]*\b/g) ?? [];
  return [...new Set(identifiers)].filter((name) => registry.has(name) && !declaration.generics.includes(name));
}

function renderDeclaration(declaration, registry) {
  declaration = { ...declaration, body: declaration.body.replace(/\bSelf\b/g, declaration.name) };
  const generics = declaration.generics.length ? `<${declaration.generics.join(", ")}>` : "";
  if (declaration.kind === "type") {
    return `export type ${declaration.name}${generics} = ${renderType(declaration.body, registry, declaration.generics)};`;
  }
  if (declaration.kind === "struct") {
    if (hasSerde(declaration.attrs, "transparent") && declaration.shape === "tuple") {
      return `export type ${declaration.name}${generics} = ${renderType(splitTopLevel(declaration.body)[0], registry, declaration.generics)};`;
    }
    if (declaration.shape === "named") {
      return `export interface ${declaration.name}${generics} ${renderFields(declaration.body, declaration.attrs, registry, declaration.generics)}`;
    }
    if (declaration.shape === "tuple") {
      const members = splitTopLevel(declaration.body).map((value) => renderType(stripVisibility(value), registry, declaration.generics));
      return `export type ${declaration.name}${generics} = ${members.length === 1 ? members[0] : `[${members.join(", ")}]`};`;
    }
    return `export type ${declaration.name}${generics} = Record<never, never>;`;
  }
  return `export type ${declaration.name}${generics} = ${renderEnum(declaration, registry)};`;
}

function renderFields(body, containerAttrs, registry, generics) {
  const renameAll = serde(containerAttrs, "rename_all");
  const fields = splitTopLevel(body).flatMap((raw) => {
    const match = raw.match(/^((?:#\[[\s\S]*?\]\s*)*)(?:pub\s+)?([A-Za-z_][A-Za-z0-9_]*):\s*([\s\S]+)$/);
    if (!match || hasSerde(match[1], "skip") || hasSerde(match[1], "skip_serializing")) return [];
    const option = unwrap(match[3].trim(), "Option");
    const optional = Boolean(option && hasSerde(match[1], "skip_serializing_if"));
    const type = renderType(option ?? match[3].trim(), registry, generics) + (option && !optional ? " | null" : "");
    const explicit = serde(match[1], "rename");
    const name = explicit ?? rename(match[2], renameAll);
    return [`  ${safeProperty(name)}${optional ? "?" : ""}: ${type};`];
  });
  return `{\n${fields.join("\n")}\n}`;
}

function renderEnum(declaration, registry) {
  const renameAll = serde(declaration.attrs, "rename_all");
  const tag = serde(declaration.attrs, "tag");
  const content = serde(declaration.attrs, "content");
  const untagged = hasSerde(declaration.attrs, "untagged");
  const variants = splitTopLevel(declaration.body).flatMap((raw) => {
    const parsed = parseVariant(raw);
    if (!parsed || hasSerde(parsed.attrs, "skip") || hasSerde(parsed.attrs, "skip_serializing")) return [];
    const name = serde(parsed.attrs, "rename") ?? rename(parsed.name, renameAll);
    const payload = variantPayload(parsed, declaration, registry);
    if (untagged) return [payload ?? "null"];
    if (tag) {
      const tagged = `${safeProperty(tag)}: ${JSON.stringify(name)}`;
      if (!payload) return [`{ ${tagged} }`];
      if (content) return [`{ ${tagged}; ${safeProperty(content)}: ${payload} }`];
      if (parsed.shape === "named") return [`${payload.slice(0, -1)}${tagged} }`];
      return [`${payload} & { ${tagged} }`];
    }
    if (!payload) return [JSON.stringify(name)];
    return [`{ ${safeProperty(name)}: ${payload} }`];
  });
  return variants.join("\n  | ");
}

function parseVariant(raw) {
  const match = raw.match(/^((?:#\[[\s\S]*?\]\s*)*)([A-Za-z_][A-Za-z0-9_]*)([\s\S]*)$/);
  if (!match) return null;
  const tail = match[3].trim();
  if (!tail) return { attrs: match[1], name: match[2], shape: "unit", body: "" };
  if (tail.startsWith("(")) return { attrs: match[1], name: match[2], shape: "tuple", body: tail.slice(1, -1) };
  if (tail.startsWith("{")) return { attrs: match[1], name: match[2], shape: "named", body: tail.slice(1, -1) };
  throw new Error(`Unsupported enum variant ${raw}`);
}

function variantPayload(variant, declaration, registry) {
  if (variant.shape === "unit") return null;
  if (variant.shape === "named") return renderFields(variant.body, declaration.attrs, registry, declaration.generics);
  const values = splitTopLevel(variant.body).map((value) => renderType(stripVisibility(value), registry, declaration.generics));
  return values.length === 1 ? values[0] : `[${values.join(", ")}]`;
}

function renderType(raw, registry, generics) {
  let source = stripVisibility(raw.trim()).replace(/^&(?:'\w+\s+)?/, "").trim();
  const array = source.match(/^\[([\s\S]+);\s*\d+\]$/);
  if (array) return `${renderType(array[1], registry, generics)}[]`;
  const tuple = source.startsWith("(") && source.endsWith(")") ? splitTopLevel(source.slice(1, -1)) : null;
  if (tuple) return `[${tuple.map((value) => renderType(value, registry, generics)).join(", ")}]`;
  for (const wrapper of ["Vec", "BTreeSet", "HashSet", "Box"]) {
    const inner = unwrap(source, wrapper);
    if (inner) return wrapper === "Box" ? renderType(inner, registry, generics) : `${renderType(inner, registry, generics)}[]`;
  }
  const option = unwrap(source, "Option");
  if (option) return `${renderType(option, registry, generics)} | null`;
  for (const map of ["BTreeMap", "HashMap"]) {
    const inner = unwrap(source, map);
    if (inner) {
      const [key, value] = splitTopLevel(inner);
      return `Record<${renderType(key, registry, generics)}, ${renderType(value, registry, generics)}>`;
    }
  }
  const generic = source.match(/^(?:[A-Za-z_][A-Za-z0-9_]*::)*([A-Za-z_][A-Za-z0-9_]*)<([\s\S]+)>$/);
  if (generic) return `${generic[1]}<${splitTopLevel(generic[2]).map((value) => renderType(value, registry, generics)).join(", ")}>`;
  const name = source.split("::").at(-1);
  if (TERMINALS.has(name)) return TERMINALS.get(name);
  if (generics.includes(name)) return name;
  if (registry.has(name)) return name;
  throw new Error(`Unresolved Rust field type ${source}`);
}

function unwrap(source, wrapper) {
  const prefix = `${wrapper}<`;
  return source.startsWith(prefix) && source.endsWith(">") ? source.slice(prefix.length, -1).trim() : null;
}

function stripVisibility(value) {
  return value.trim().replace(/^pub(?:\([^)]*\))?\s+/, "");
}

function rename(value, convention) {
  if (!convention) return value;
  if (convention === "camelCase") return value.replace(/_([a-z])/g, (_, character) => character.toUpperCase());
  if (convention === "snake_case") return value.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase();
  if (convention === "SCREAMING_SNAKE_CASE") return value.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toUpperCase();
  if (convention === "kebab-case") return rename(value, "snake_case").replaceAll("_", "-");
  return value;
}

function safeProperty(value) {
  return /^[A-Za-z_$][A-Za-z0-9_$]*$/.test(value) ? value : JSON.stringify(value);
}

module.exports = { renderReachable };
