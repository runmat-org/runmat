const fs = require("fs");
const path = require("path");

function rustFiles(root) {
  const result = [];
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    const target = path.join(root, entry.name);
    if (entry.isDirectory()) result.push(...rustFiles(target));
    else if (entry.isFile() && entry.name.endsWith(".rs")) result.push(target);
  }
  return result;
}

function matching(source, start, open, close) {
  let depth = 0;
  let string = false;
  let escaped = false;
  for (let index = start; index < source.length; index += 1) {
    const character = source[index];
    if (string) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') string = false;
      continue;
    }
    if (character === '"') string = true;
    else if (character === open) depth += 1;
    else if (character === close && --depth === 0) return index;
  }
  throw new Error(`Unclosed ${open} at byte ${start}`);
}

function splitTopLevel(source, separator = ",") {
  const output = [];
  let start = 0;
  let angle = 0;
  let round = 0;
  let square = 0;
  let curly = 0;
  let string = false;
  let escaped = false;
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (string) {
      if (escaped) escaped = false;
      else if (character === "\\") escaped = true;
      else if (character === '"') string = false;
      continue;
    }
    if (character === '"') string = true;
    else if (character === "<") angle += 1;
    else if (character === ">") angle -= 1;
    else if (character === "(") round += 1;
    else if (character === ")") round -= 1;
    else if (character === "[") square += 1;
    else if (character === "]") square -= 1;
    else if (character === "{") curly += 1;
    else if (character === "}") curly -= 1;
    else if (character === separator && angle === 0 && round === 0 && square === 0 && curly === 0) {
      output.push(source.slice(start, index).trim());
      start = index + 1;
    }
  }
  const tail = source.slice(start).trim();
  if (tail) output.push(tail);
  return output;
}

function attributesBefore(source, declarationStart) {
  let cursor = declarationStart;
  while (cursor > 0 && /\s/.test(source[cursor - 1])) cursor -= 1;
  const attributes = [];
  while (cursor >= 2 && source.slice(0, cursor).endsWith("]")) {
    let depth = 0;
    let start = cursor - 1;
    for (; start >= 0; start -= 1) {
      if (source[start] === "]") depth += 1;
      else if (source[start] === "[") {
        depth -= 1;
        if (depth === 0 && source[start - 1] === "#") {
          start -= 1;
          break;
        }
      }
    }
    if (start < 0) break;
    attributes.unshift(source.slice(start, cursor));
    cursor = start;
    while (cursor > 0 && /\s/.test(source[cursor - 1])) cursor -= 1;
  }
  return attributes.join("\n");
}

function parseDeclarations(files, options = {}) {
  const declarations = new Map();
  const preferred = options.preferredDeclarations ?? {};
  const register = (declaration) => {
    const existing = declarations.get(declaration.name);
    if (!existing) {
      declarations.set(declaration.name, declaration);
      return;
    }
    const preference = preferred[declaration.name];
    const existingPreferred = preference && normalizePath(existing.file).endsWith(preference);
    const declarationPreferred = preference && normalizePath(declaration.file).endsWith(preference);
    if (declarationPreferred && !existingPreferred) {
      declarations.set(declaration.name, declaration);
      return;
    }
    if (existingPreferred && !declarationPreferred) return;
    if (sameDeclaration(existing, declaration)) return;
    existing.conflicts = [...(existing.conflicts ?? []), declaration.file];
  };
  for (const file of files) {
    const source = fs.readFileSync(file, "utf8");
    // Expand the one canonical quantity catalog for its actual serde wire
    // projections. Do not infer physical schemas from field suffixes.
    const physicalCatalog = source.match(/catalog!\s*\{([\s\S]*?)\n\}/);
    if (physicalCatalog) {
      const entries = [...physicalCatalog[1].matchAll(/([A-Za-z_][A-Za-z0-9_]*),\s*"([^"]+)",\s*\[[^\]]+\],\s*(?:true|false);/g)];
      if (!entries.length) throw new Error(`Empty physical quantity catalog in ${file}`);
      register({ name: "QuantityKind", kind: "enum", attrs: '#[derive(Serialize)] #[serde(rename_all = "snake_case")]', body: entries.map(e=>e[1]).join(","), shape: "named", generics: [], file });
      for (const entry of entries) {
        const storage = ["Position","Translation"].includes(entry[1]) ? "[f64; 3]" : "f64";
        register({ name:entry[1], kind:"struct", attrs:"#[derive(Serialize)]", body:`pub ${entry[2]}: ${storage}`, shape:"named", generics:[], file });
      }
    }
    if (source.includes("pub struct UnitDefinitionWire")) {
      for (const name of ["UnitDefinition","LengthUnit"]) register({ name, kind:"type", attrs:"#[derive(Serialize)]", body:"UnitDefinitionWire", shape:"alias", generics:[], file });
    }
    if (/pub use runmat_types::physical::QuantityKind as PhysicalQuantityKind/.test(source)) {
      register({name:"PhysicalQuantityKind",kind:"type",attrs:"#[derive(Serialize)]",body:"QuantityKind",shape:"alias",generics:[],file});
    }
    if (/pub type DimensionedScalar\s*=\s*Quantity<DynamicDimension>/.test(source)) {
      register({name:"DimensionedScalar",kind:"type",attrs:"#[derive(Serialize)]",body:"DimensionedQuantityWire<f64>",shape:"alias",generics:[],file});
    }
    // Stable public aliases share the serialization implementation of their
    // physical target; Rust aliases do not carry a redundant Serialize derive.
    for (const alias of source.matchAll(/pub type ([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*(DimensionedScalar|SourceQuantity|DynamicDimension)\s*;/g)) {
      register({name:alias[1],kind:"type",attrs:"#[derive(Serialize)]",body:alias[2],shape:"alias",generics:[],file});
    }
    for (const quantity of source.matchAll(/scalar_quantity!\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\);/g)) {
      register({
        name: quantity[1],
        kind: "struct",
        attrs: "#[derive(Serialize)]",
        body: `pub ${quantity[2]}: f64`,
        shape: "named",
        generics: [],
        file,
      });
    }
    const pattern = /\bpub\s+(struct|enum|type)\s+([A-Za-z_][A-Za-z0-9_]*)(\s*<[^;{(]+>)?/g;
    for (const match of source.matchAll(pattern)) {
      const kind = match[1];
      const name = match[2];
      let attrs = attributesBefore(source, match.index);
      if (!/\bSerialize\b/.test(attrs)) {
        // A single-field tuple struct with a hand-written `impl Serialize` that delegates to its
        // field (for example an interned `Arc` handle) has its field's wire format.
        const delegating = new RegExp(`impl\\s+Serialize\\s+for\\s+${name}\\s*\\{[\\s\\S]*?self\\.0\\.serialize\\(serializer\\)`);
        if (kind !== "struct" || !delegating.test(source)) continue;
        attrs = `${attrs}\n#[serde(transparent)]`;
      }
      let cursor = match.index + match[0].length;
      while (/\s/.test(source[cursor])) cursor += 1;
      let body = "";
      let shape = "alias";
      if (kind === "type") {
        const end = source.indexOf(";", cursor);
        body = source.slice(cursor, end).replace(/^\s*=\s*/, "").trim();
      } else if (source[cursor] === "{") {
        const end = matching(source, cursor, "{", "}");
        body = source.slice(cursor + 1, end);
        shape = "named";
      } else if (source[cursor] === "(") {
        const end = matching(source, cursor, "(", ")");
        body = source.slice(cursor + 1, end);
        shape = "tuple";
      } else if (source[cursor] === ";") {
        shape = "unit";
      } else {
        continue;
      }
      body = body.replace(/\/\/.*$/gm, "").replace(/\/\*[\s\S]*?\*\//g, "");
      if (body.includes("$")) continue; // macro templates are not declarations
      if (physicalCatalog && ["QuantityKind","AnyQuantity"].includes(name)) continue;
      register({
        name,
        kind,
        attrs,
        body,
        shape,
        generics: genericNames(match[3] ?? ""),
        file,
      });
    }
  }
  return declarations;
}

function normalizePath(value) {
  return value.replaceAll("\\", "/");
}

function sameDeclaration(left, right) {
  return left.kind === right.kind
    && left.shape === right.shape
    && left.body.replace(/\s+/g, " ").trim() === right.body.replace(/\s+/g, " ").trim()
    && left.attrs.replace(/\s+/g, " ").trim() === right.attrs.replace(/\s+/g, " ").trim();
}

function genericNames(source) {
  if (!source.trim()) return [];
  return splitTopLevel(source.trim().slice(1, -1))
    .map((value) => value.split(":")[0].trim())
    .filter((value) => /^[A-Z][A-Za-z0-9_]*$/.test(value));
}

function serde(attrs, key) {
  const pattern = new RegExp(`\\b${key}\\s*=\\s*"([^"]+)"`);
  return attrs.match(pattern)?.[1] ?? null;
}

function hasSerde(attrs, key) {
  return new RegExp(`\\b${key}\\b`).test(attrs);
}

module.exports = {
  hasSerde,
  parseDeclarations,
  rustFiles,
  serde,
  splitTopLevel,
};
