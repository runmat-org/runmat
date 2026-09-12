import { compareCodePoint } from "./constants.mjs";
import { evidenceDigest } from "./evidence.mjs";
import { parseIntegrationProductVerification } from "./integration-products.mjs";
import { bindModuleCompositionProjection } from "./module-composition/binding.mjs";
import { array, digest, enumValue, exact, integer, kind, repositoryPath, stableId } from "./schema.mjs";

export function parseGeneratedProductDefinitions(value) {
  const products = array(value, "reviewed generated products", { empty: true }).map((entry) => {
    exact(entry, ["product_id", "path", "producer", "generator", "baseline_digest", "verification"], "reviewed generated product");
    const productId = stableId(entry.product_id, "reviewed generated product id");
    const productPath = repositoryPath(entry.path, `${productId} reviewed product path`);
    if (entry.producer !== "integration") throw new Error(`${productId}: reviewed product must be integration-owned`);
    exact(entry.generator, ["path", "baseline_digest"], `${productId} reviewed generator`);
    const generatorPath = repositoryPath(entry.generator.path, `${productId} reviewed generator path`);
    const generatorDigest = digest(entry.generator.baseline_digest, `${productId} reviewed generator digest`);
    if (entry.baseline_digest !== null) digest(entry.baseline_digest, `${productId} reviewed product baseline digest`);
    return { ...entry, product_id: productId, path: productPath, generator: { path: generatorPath, baseline_digest: generatorDigest }, verification: parseIntegrationProductVerification(entry.verification, productId, productPath) };
  });
  const ids = products.map((entry) => entry.product_id);
  if (new Set(ids).size !== ids.length || JSON.stringify(ids) !== JSON.stringify([...ids].sort(compareCodePoint))) {
    throw new Error("reviewed generated products must be unique and canonically ordered");
  }
  return products;
}

export function parseGeneratedProductsProof(value, expected) {
  kind(value, 2, "runmat-builtin-generated-products-proof", "generated products proof");
  exact(value, ["schema_version", "kind", "authority", "products", "result"], "generated products proof");
  if (value.authority !== "machine-derived-integration-evidence") throw new Error("generated products proof has invalid authority");
  enumValue(value.result, ["pass", "fail"], "generated products result");
  const definitions = parseGeneratedProductDefinitions(expected.integration_products);
  const compositionProjection = bindModuleCompositionProjection(
    definitions,
    expected.module_composition_projection,
  );
  const compositionProducts = new Map(
    (compositionProjection?.products ?? [])
      .map((entry) => [entry.product_id, entry]),
  );
  const products = array(value.products, "generated products", { empty: true }).map((entry) => {
    const productId = stableId(entry?.product_id, "generated product id");
    const reviewed = definitions.find((definition) => definition.product_id === productId);
    if (!reviewed) throw new Error(`${productId}: generated product is not globally reviewed`);
    return parseProduct(
      entry,
      reviewed,
      expected.native_registration_manifest,
      compositionProducts.get(productId) ?? null,
    );
  });
  const keys = products.map((entry) => entry.product_id);
  if (new Set(keys).size !== keys.length || JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error("generated products must be unique and canonically ordered");
  }
  const expectedProducts = definitions
    .map((entry) => ({ product_id: entry.product_id, path: entry.path }))
    .sort((left, right) => compareCodePoint(left.product_id, right.product_id));
  if (JSON.stringify(products.map(({ product_id, path }) => ({ product_id, path }))) !== JSON.stringify(expectedProducts)) {
    throw new Error("generated products do not exactly cover the reviewed integration outputs");
  }
  for (const product of products) {
    const reviewed = definitions.find((entry) => entry.product_id === product.product_id);
    if (product.generator.path !== reviewed.generator.path
      || product.generator.content_digest !== reviewed.generator.baseline_digest) {
      throw new Error(`${product.product_id}: generator differs from the globally reviewed product registry`);
    }
    const source = expected.source_files.find((entry) => entry.path === reviewed.generator.path);
    if (!source || source.content_digest !== reviewed.generator.baseline_digest) {
      throw new Error(`${product.product_id}: reviewed generator differs from the frozen source inventory`);
    }
  }
  const passed = products.every((entry) => entry.deterministic && entry.synchronized && entry.verification.result === "pass");
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

function parseProduct(value, reviewed, nativeManifest, compositionProduct) {
  exact(value, ["product_id", "path", "generator", "checked_in", "first", "second", "deterministic", "synchronized", "verification"], "generated product");
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
  parseProductVerification(
    value.verification,
    reviewed.verification,
    nativeManifest,
    compositionProduct,
    value.product_id,
  );
  return value;
}

function parseProductVerification(value, contract, nativeManifest, compositionProduct, id) {
  if (contract.kind === "content_identity") {
    exact(value, ["kind", "result"], `${id} product verification`);
    if (value.kind !== contract.kind || value.result !== "pass") throw new Error(`${id}: content identity verification failed`);
    return;
  }
  if (contract.kind === "rust_module_composition") {
    exact(value, ["kind", "projection_digest", "result"], `${id} product verification`);
    if (value.kind !== contract.kind || value.result !== "pass") throw new Error(`${id}: module composition verification failed`);
    if (!compositionProduct || evidenceDigest(compositionProduct) !== digest(value.projection_digest, `${id} composition projection digest`)) {
      throw new Error(`${id}: composition proof differs from the exact staged projection`);
    }
    return;
  }
  exact(value, ["kind", "generated_manifest", "native_manifest", "result"], `${id} product verification`);
  if (value.kind !== contract.kind) throw new Error(`${id}: product verification contract differs from review`);
  const generated = registrationManifestIdentity(value.generated_manifest, `${id} generated WASM manifest`);
  const native = registrationManifestIdentity(value.native_manifest, `${id} native manifest`);
  const expectedNative = expectedRegistrationManifestIdentity(nativeManifest, `${id} expected native manifest`);
  if (!sameRegistrationManifestIdentity(native, expectedNative)) throw new Error(`${id}: proof native manifest differs from subject compiled inventory`);
  const matches = sameRegistrationManifestIdentity(generated, native);
  if (value.result !== (matches ? "pass" : "fail")) throw new Error(`${id}: registration manifest parity result is inconsistent`);
}

export function registrationManifestIdentity(value, label = "registration manifest identity") {
  exact(value, ["schema_version", "digest", "counts"], label);
  if (value.schema_version !== 1) throw new Error(`${label} schema must be 1`);
  if (typeof value.digest !== "string" || !/^[a-f0-9]{64}$/.test(value.digest)) throw new Error(`${label} digest must be canonical SHA-256 hex`);
  exact(value.counts, ["builtin", "constant", "gpu_spec", "fusion_spec"], `${label} counts`);
  for (const kind of ["builtin", "constant", "gpu_spec", "fusion_spec"]) integer(value.counts[kind], `${label} ${kind} count`);
  return { schema_version: 1, digest: value.digest, counts: { ...value.counts } };
}

export function expectedRegistrationManifestIdentity(value, label = "expected registration manifest") {
  if (value && Object.prototype.hasOwnProperty.call(value, "entries")) {
    exact(value, ["schema_version", "digest", "counts", "entries"], label);
    return registrationManifestIdentity({ schema_version: value.schema_version, digest: value.digest, counts: value.counts }, label);
  }
  return registrationManifestIdentity(value, label);
}

function sameRegistrationManifestIdentity(left, right) {
  return left.schema_version === right.schema_version
    && left.digest === right.digest
    && ["builtin", "constant", "gpu_spec", "fusion_spec"]
      .every((kind) => left.counts[kind] === right.counts[kind]);
}

function parseObservation(value, label) {
  exact(value, ["byte_length", "content_digest"], label);
  integer(value.byte_length, `${label} byte length`);
  digest(value.content_digest, `${label} content digest`);
}
