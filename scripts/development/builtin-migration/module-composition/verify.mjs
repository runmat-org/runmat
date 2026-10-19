import { isDeepStrictEqual } from "node:util";

import { parseHandwrittenComposition } from "./handwritten-parser.mjs";
import { normalizeDeclarationPaths } from "./handwritten.mjs";
import { expandComposedSource } from "./composed-source.mjs";
import { generatedHeader, renderModuleCompositionCompanion, renderModuleCompositionProduct } from "./generate.mjs";
import { parseCompositionProduct } from "./schema.mjs";
import { expectedCompositionSurface } from "./surface.mjs";
import { activeCompanions, parseDerivedCompanionProduct } from "./companions.mjs";

export function verifyModuleCompositionProduct(productValue, source, companionSources = new Map()) {
  if (Object.hasOwn(productValue, "companion_parent")) {
    const companion = parseDerivedCompanionProduct(productValue);
    const expected = renderModuleCompositionCompanion(
      companion.companion_parent, companion.companion_section,
    );
    if (source !== expected || companion.state !== "present") {
      throw new Error(`${companion.product_id}: derived companion bytes differ from its reviewed parent`);
    }
    return { product_id: companion.product_id, path: companion.path, result: "pass" };
  }
  const product = parseCompositionProduct(productValue);
  const observed = parseGeneratedCompositionSet(product, source, companionSources);
  const expected = expectedCompositionSurface(product);
  if (!isDeepStrictEqual(observed, expected)) {
    throw new Error(`${product.product_id}: generated Rust topology differs from its typed projection`);
  }
  if (source !== renderModuleCompositionProduct(product)) {
    throw new Error(`${product.product_id}: generated Rust bytes are not canonical`);
  }
  return { product_id: product.product_id, path: product.path, result: "pass" };
}

export function parseGeneratedCompositionSet(product, source, companionSources = new Map()) {
  const companions = activeCompanions(product);
  const composed = expandComposedSource(product.product_id, source, companionSources);
  if (!isDeepStrictEqual(composed.active, companions)) {
    throw new Error(`${product.product_id}: observed companion sections differ from its typed projection`);
  }
  return normalizeDeclarationPaths(parseGeneratedModuleComposition(composed.expanded), product);
}

export function parseGeneratedModuleComposition(source) {
  if (typeof source !== "string" || source.includes("\r")
    || source.includes("\0") || !source.endsWith("\n")) {
    throw new Error("generated module source must be canonical LF-terminated UTF-8 text");
  }
  const prefix = `${generatedHeader()}\n\n`;
  if (source === `${generatedHeader()}\n`) return parseHandwrittenComposition("", "generated module");
  if (!source.startsWith(prefix)) throw new Error("generated module header is invalid");
  return parseHandwrittenComposition(source.slice(prefix.length), "generated module");
}
