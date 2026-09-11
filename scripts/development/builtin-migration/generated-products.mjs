import { compareCodePoint } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { array, digest, enumValue, exact, integer, kind, repositoryPath, stableId } from "./schema.mjs";

export function parseGeneratedProductsProof(value, expected) {
  kind(value, 1, "runmat-builtin-generated-products-proof", "generated products proof");
  exact(value, ["schema_version", "kind", "authority", "products", "result"], "generated products proof");
  if (value.authority !== "machine-derived-integration-evidence") throw new Error("generated products proof has invalid authority");
  enumValue(value.result, ["pass", "fail"], "generated products result");
  const products = array(value.products, "generated products").map(parseProduct);
  const keys = products.map((entry) => entry.product_id);
  if (new Set(keys).size !== keys.length || JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error("generated products must be unique and canonically ordered");
  }
  const expectedProducts = expected.integration_outputs
    .map((entry) => ({ product_id: entry.product_id, path: entry.path }))
    .sort((left, right) => compareCodePoint(left.product_id, right.product_id));
  if (JSON.stringify(products.map(({ product_id, path }) => ({ product_id, path }))) !== JSON.stringify(expectedProducts)) {
    throw new Error("generated products do not exactly cover the reviewed integration outputs");
  }
  for (const product of products) {
    const generator = expected.source_files.find((entry) => entry.path === product.generator.path);
    if (!generator || generator.content_digest !== product.generator.content_digest) {
      throw new Error(`${product.product_id}: generator differs from the frozen source inventory`);
    }
  }
  const passed = products.every((entry) => entry.deterministic && entry.synchronized);
  if ((value.result === "pass") !== passed) throw new Error("generated product aggregate result is inconsistent");
  return { ...value, products };
}

export function generatedProductChecks(proof, identities) {
  const result = proof.result === "pass" ? "pass" : "fail";
  const proofDigest = evidenceDigest(proof);
  return identities.map((identity) => ({
    id: `deterministic-products:${identity}`,
    result,
    evidence_digest: proofDigest,
  }));
}

function parseProduct(value) {
  exact(value, ["product_id", "path", "generator", "checked_in", "first", "second", "deterministic", "synchronized"], "generated product");
  stableId(value.product_id, "generated product id");
  repositoryPath(value.path, `${value.product_id} generated path`);
  exact(value.generator, ["path", "content_digest"], `${value.product_id} generator`);
  repositoryPath(value.generator.path, `${value.product_id} generator path`);
  digest(value.generator.content_digest, `${value.product_id} generator digest`);
  for (const field of ["checked_in", "first", "second"]) parseObservation(value[field], `${value.product_id} ${field}`);
  if (typeof value.deterministic !== "boolean" || typeof value.synchronized !== "boolean") throw new Error(`${value.product_id}: generated product result fields must be booleans`);
  const deterministic = value.first.content_digest === value.second.content_digest;
  const synchronized = value.checked_in.content_digest === value.first.content_digest;
  if (value.deterministic !== deterministic || value.synchronized !== synchronized) throw new Error(`${value.product_id}: generated product result conflicts with observed digests`);
  if (value.first.byte_length !== value.second.byte_length || (synchronized && value.checked_in.byte_length !== value.first.byte_length)) {
    throw new Error(`${value.product_id}: generated product byte lengths conflict with content identity`);
  }
  return value;
}

function parseObservation(value, label) {
  exact(value, ["byte_length", "content_digest"], label);
  integer(value.byte_length, `${label} byte length`);
  digest(value.content_digest, `${label} content digest`);
}
