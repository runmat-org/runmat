import { companionDefinitions } from "./companions.mjs";
import { generatedHeader } from "./generate.mjs";

const ANCHORS = Object.freeze({
  declarations: 'include!("declarations.rs");',
  reexports: "pub use self::exports::*;",
});

// Expand only the two fixed composition sections. The Rust parser then observes
// the resulting declarations and exports independently of the renderer.
export function expandComposedSource(parentId, source, companionSources = new Map()) {
  if (!(companionSources instanceof Map)) throw new Error(`${parentId}: companion sources must be a path map`);
  const definitions = companionDefinitions().filter((entry) => entry.parent_id === parentId);
  let expanded = source;
  const active = [];
  for (const definition of definitions) {
    const anchor = ANCHORS[definition.section];
    const occurrences = expanded.split(anchor).length - 1;
    if (occurrences > 1) throw new Error(`${parentId}: ${definition.section} anchor repeats`);
    if (occurrences === 0) {
      if (companionSources.has(definition.path)) {
        throw new Error(`${parentId}: unreferenced ${definition.section} companion source`);
      }
      continue;
    }
    if (!companionSources.has(definition.path)) {
      throw new Error(`${parentId}: missing ${definition.section} companion source`);
    }
    const companionSource = companionSources.get(definition.path);
    if (typeof companionSource !== "string") throw new Error(`${parentId}: companion source is not text`);
    const childBody = stripGeneratedHeader(companionSource);
    expanded = expanded.replace(anchor, definition.section === "reexports"
      ? childBody.replaceAll("super::", "self::") : childBody);
    active.push(definition);
  }
  for (const sourcePath of companionSources.keys()) {
    if (!active.some((entry) => entry.path === sourcePath)) {
      throw new Error(`${parentId}: unauthorized companion source ${sourcePath}`);
    }
  }
  return { expanded, active };
}

function stripGeneratedHeader(source) {
  const prefix = `${generatedHeader()}\n\n`;
  if (source === `${generatedHeader()}\n`) return "";
  if (!source.startsWith(prefix) || source.includes("\r") || source.includes("\0")
    || !source.endsWith("\n")) {
    throw new Error("composition companion has an invalid generated envelope");
  }
  return source.slice(prefix.length);
}
