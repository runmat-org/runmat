// Companions are derived from the same reviewed parent children. A typed
// support-child transition activates exactly one fixed section and path.
import { parseCompositionProduct } from "./schema.mjs";

const COMPANIONS = Object.freeze([
  Object.freeze({
    parent_id: "catalog-root",
    section: "reexports",
    module: "exports",
    path: "crates/runmat-builtins/src/catalog/entries/exports.rs",
  }),
  Object.freeze({
    parent_id: "catalog-io",
    section: "declarations",
    module: "declarations",
    path: "crates/runmat-builtins/src/catalog/entries/io/declarations.rs",
  }),
  Object.freeze({
    parent_id: "catalog-io",
    section: "reexports",
    module: "exports",
    path: "crates/runmat-builtins/src/catalog/entries/io/exports.rs",
  }),
  Object.freeze({
    parent_id: "catalog-io-repl-fs",
    section: "declarations",
    module: "declarations",
    path: "crates/runmat-builtins/src/catalog/entries/io/repl_fs/declarations.rs",
  }),
]);

export function companionDefinitions() { return COMPANIONS; }

export function companionDefinition(parentId, section) {
  return COMPANIONS.find((entry) => entry.parent_id === parentId && entry.section === section) ?? null;
}

export function activeCompanions(product) {
  return product.children.flatMap((child) => {
    if (child.companion_section == null) return [];
    const definition = companionDefinition(product.product_id, child.companion_section);
    if (!definition || child.module !== definition.module || child.source_path !== definition.path) {
      throw new Error(`${product.product_id}/${child.module}: companion differs from fixed section/path`);
    }
    return [definition];
  });
}

export function derivedCompanionProducts(product) {
  product = parseCompositionProduct(product);
  const active = new Set(activeCompanions(product).map((entry) => entry.section));
  return COMPANIONS.filter((entry) => entry.parent_id === product.product_id).map((entry) => ({
    product_id: `${entry.parent_id}-${entry.section}`,
    path: entry.path,
    state: product.state === "present" && active.has(entry.section) ? "present" : "absent",
    children: [],
    companion_parent: product,
    companion_section: entry.section,
  }));
}

export function parseDerivedCompanionProduct(value) {
  if (!value || typeof value !== "object" || !Object.hasOwn(value, "companion_parent")) {
    throw new Error("derived companion product is missing its reviewed parent");
  }
  const parent = parseCompositionProduct(value.companion_parent);
  const expected = derivedCompanionProducts(parent)
    .find((entry) => entry.product_id === value.product_id);
  if (!expected || value.path !== expected.path || value.state !== expected.state
    || value.companion_section !== expected.companion_section
    || !Array.isArray(value.children) || value.children.length !== 0
    || JSON.stringify(Object.keys(value).sort()) !== JSON.stringify(Object.keys(expected).sort())) {
    throw new Error("derived companion product differs from its fixed parent section/path/state");
  }
  return expected;
}

export function companionSourcesForParent(product, rendered) {
  const parent = parseCompositionProduct(product);
  const sources = new Map();
  for (const definition of activeCompanions(parent)) {
    const match = rendered.find((entry) => entry.product.product_id === `${parent.product_id}-${definition.section}`);
    if (!match || match.product.path !== definition.path || typeof match.content !== "string") {
      throw new Error(`${parent.product_id}: active ${definition.section} companion is missing from the composition set`);
    }
    const companion = parseDerivedCompanionProduct(match.product);
    if (JSON.stringify(companion.companion_parent) !== JSON.stringify(parent)) {
      throw new Error(`${parent.product_id}: companion is not derived from the selected parent projection`);
    }
    sources.set(definition.path, match.content);
  }
  return sources;
}
