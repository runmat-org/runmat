import fs from "node:fs";
import path from "node:path";
import { compareCodePoint } from "./constants.mjs";

export function filesUnder(root, relativeDirectory, suffix = null) {
  const absoluteRoot = path.join(root, relativeDirectory);
  if (!fs.existsSync(absoluteRoot)) return [];
  const results = [];
  const visit = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const absolute = path.join(directory, entry.name);
      if (entry.isDirectory()) visit(absolute);
      else if (entry.isFile() && (!suffix || entry.name.endsWith(suffix))) {
        results.push(path.relative(root, absolute).split(path.sep).join("/"));
      }
    }
  };
  visit(absoluteRoot);
  return results.sort(compareCodePoint);
}

export function read(root, relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

export function lineCount(text) {
  return text.length === 0 ? 0 : text.split("\n").length;
}

// Returns balanced Rust attribute bodies. This deliberately does not attempt to
// parse Rust: it only recognizes the narrow registration syntax owned by the
// runtime macro and reports malformed attributes to the caller.
export function runtimeAttributes(text) {
  const marker = /#\[(?:(?:runmat_macros|crate)::)?runtime_builtin\s*\(/g;
  const attributes = [];
  for (const match of text.matchAll(marker)) {
    let depth = 1;
    let quote = false;
    let escaped = false;
    let index = match.index + match[0].length;
    for (; index < text.length && depth > 0; index += 1) {
      const character = text[index];
      if (quote) {
        if (escaped) escaped = false;
        else if (character === "\\") escaped = true;
        else if (character === '"') quote = false;
      } else if (character === '"') quote = true;
      else if (character === "(") depth += 1;
      else if (character === ")") depth -= 1;
    }
    if (depth !== 0 || text[index] !== "]") {
      attributes.push({ error: "unterminated runtime_builtin attribute", offset: match.index });
      continue;
    }
    const body = text.slice(match.index + match[0].length, index - 1);
    const following = text.slice(index + 1, index + 500);
    const functionMatch = following.match(/(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)/);
    attributes.push({
      body,
      offset: match.index,
      name: body.match(/\bname\s*=\s*"([^"]+)"/)?.[1] ?? null,
      category: body.match(/\bcategory\s*=\s*"([^"]+)"/)?.[1] ?? null,
      bindingVariant: body.match(/\bbinding_variant\s*=\s*"([^"]+)"/)?.[1] ?? "default",
      builtinPath: body.match(/\bbuiltin_path\s*=\s*"([^"]+)"/)?.[1] ?? null,
      functionName: functionMatch?.[1] ?? null,
      hasResolver: /\btype_resolver(?:_ctx)?\s*\(/.test(body),
      hasDescriptor: /\bdescriptor\s*\(/.test(body),
      hasCapabilities: /\brequired_capabilities\s*=/.test(body),
    });
  }
  return attributes;
}

export function runtimeMacroInvocations(text) {
  const definitions = [...text.matchAll(/\bmacro_rules!\s+([A-Za-z_][A-Za-z0-9_]*)/g)];
  const registrations = [];
  for (let definitionIndex = 0; definitionIndex < definitions.length; definitionIndex += 1) {
    const definition = definitions[definitionIndex];
    const openingBrace = text.indexOf("{", definition.index + definition[0].length);
    if (openingBrace < 0) continue;
    let depth = 1;
    let definitionEnd = openingBrace + 1;
    for (; definitionEnd < text.length && depth > 0; definitionEnd += 1) {
      if (text[definitionEnd] === "{") depth += 1;
      else if (text[definitionEnd] === "}") depth -= 1;
    }
    if (depth !== 0 || !/runtime_builtin\s*\(/.test(text.slice(openingBrace, definitionEnd))) continue;
    const macroName = definition[1].replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const invocation = new RegExp(`\\b${macroName}!\\s*\\(\\s*([A-Za-z_][A-Za-z0-9_]*)?[^\"]*\"([A-Za-z][A-Za-z0-9_.]*)\"`, "g");
    for (const match of text.matchAll(invocation)) {
      if (match.index < definitionEnd) continue;
      registrations.push({ name: match[2], functionName: match[1] ?? null, macro: definition[1] });
    }
  }
  return registrations;
}

export function entryMacroIdentityLiterals(text) {
  const results = [];
  const marker = /\b([A-Za-z_][A-Za-z0-9_]*(?:entry|contract))!\s*\(/g;
  for (const match of text.matchAll(marker)) {
    // Ignore macro definitions; only invocation arguments are inventory evidence.
    if (text.slice(Math.max(0, match.index - 20), match.index).includes("macro_rules!")) continue;
    let depth = 1;
    let quote = false;
    let escaped = false;
    let index = match.index + match[0].length;
    for (; index < text.length && depth > 0; index += 1) {
      const character = text[index];
      if (quote) {
        if (escaped) escaped = false;
        else if (character === "\\") escaped = true;
        else if (character === '"') quote = false;
      } else if (character === '"') quote = true;
      else if (character === "(") depth += 1;
      else if (character === ")") depth -= 1;
    }
    if (depth !== 0) continue;
    const body = text.slice(match.index + match[0].length, index - 1);
    const named = body.match(/\bname:\s*"([A-Za-z][A-Za-z0-9_.]*)"/)?.[1];
    const firstLiteral = body.match(/"([A-Za-z][A-Za-z0-9_.]*)"/)?.[1];
    if (named ?? firstLiteral) results.push(named ?? firstLiteral);
  }
  return results;
}

export function jsonDocument(root, relativePath) {
  try {
    return { value: JSON.parse(read(root, relativePath)), error: null };
  } catch (error) {
    return { value: null, error: error instanceof Error ? error.message : String(error) };
  }
}
