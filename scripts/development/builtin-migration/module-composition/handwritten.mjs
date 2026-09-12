import { parseHandwrittenComposition } from "./handwritten-parser.mjs";
import { expectedCompositionSurface } from "./surface.mjs";

export function auditHandwrittenComposition(source, product) {
  const observed = parseHandwrittenComposition(source, product.product_id);
  const expected = expectedCompositionSurface(product);
  rejectDuplicateDeclarations(observed.declarations, product.product_id);
  rejectNonchildReexports(observed, product.product_id);
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error(`${product.product_id}: handwritten composition surface differs from review${declarationDifference(observed, expected)}`);
  }
  return observed.declarations;
}

function rejectNonchildReexports(surface, productId) {
  const children = new Set(surface.declarations.map((entry) => entry.module));
  const outside = surface.reexports.find((entry) => !children.has(entry.module));
  if (outside) throw new Error(`${productId}: reexport source ${outside.module} is not a declared direct child`);
}

function rejectDuplicateDeclarations(declarations, productId) {
  const names = declarations.map((entry) => entry.module.toLowerCase());
  if (new Set(names).size !== names.length) throw new Error(`${productId}: direct module declarations collide or repeat`);
}

function declarationDifference(observed, expected) {
  const expectedNames = new Set(expected.declarations.map((entry) => entry.module));
  const observedNames = new Set(observed.declarations.map((entry) => entry.module));
  const missing = expected.declarations.filter((entry) => !observedNames.has(entry.module)).map((entry) => entry.module);
  const extra = observed.declarations.filter((entry) => !expectedNames.has(entry.module)).map((entry) => entry.module);
  const details = [
    missing.length ? `missing ${missing.join(", ")}` : null,
    extra.length ? `unexpected ${extra.join(", ")}` : null,
  ].filter(Boolean);
  return details.length ? `: ${details.join("; ")}` : "";
}
