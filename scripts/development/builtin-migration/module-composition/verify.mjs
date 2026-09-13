import { parseHandwrittenComposition } from "./handwritten-parser.mjs";
import { generatedHeader, renderModuleCompositionProduct } from "./generate.mjs";
import { parseCompositionProduct } from "./schema.mjs";
import { expectedCompositionSurface } from "./surface.mjs";

export function verifyModuleCompositionProduct(productValue, source) {
  const product = parseCompositionProduct(productValue);
  const observed = parseGeneratedModuleComposition(source);
  const expected = expectedCompositionSurface(product);
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error(`${product.product_id}: generated Rust topology differs from its typed projection`);
  }
  if (source !== renderModuleCompositionProduct(product)) {
    throw new Error(`${product.product_id}: generated Rust bytes are not canonical`);
  }
  return { product_id: product.product_id, path: product.path, result: "pass" };
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
