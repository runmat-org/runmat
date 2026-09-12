import { compareCodePoint } from "./constants.mjs";
import { parseModuleCompositionContract } from "./module-composition/schema.mjs";
import { array, digest, exact, object, repositoryPath, stableId } from "./schema.mjs";

export function parseIntegrationProductRegistry(value, inventory) {
  const registry = object(value, "integration product registry");
  const ids = Object.keys(registry);
  if (JSON.stringify(ids) !== JSON.stringify([...ids].sort(compareCodePoint))) {
    throw new Error("integration product registry must use canonical product-id order");
  }
  const sourceFiles = new Map(inventory.source.files.map((entry) => [entry.path, entry.content_digest]));
  const products = new Map();
  const paths = new Map();
  for (const id of ids) {
    stableId(id, "integration product id");
    const entry = registry[id];
    const fields = ["path", "producer", "generator", "baseline_digest", "verification"];
    if (Object.hasOwn(entry, "lifecycle")) fields.push("lifecycle");
    exact(entry, fields, `${id} integration product`);
    const productPath = repositoryPath(entry.path, `${id} integration product path`);
    if (entry.producer !== "integration") throw new Error(`${id}: integration product producer must be integration`);
    exact(entry.generator, ["path", "baseline_digest"], `${id} integration product generator`);
    const generatorPath = repositoryPath(entry.generator.path, `${id} generator path`);
    const generatorDigest = digest(entry.generator.baseline_digest, `${id} generator baseline digest`);
    if (sourceFiles.get(generatorPath) !== generatorDigest) {
      throw new Error(`${id}: generator does not match the frozen source inventory`);
    }
    const observedProductDigest = sourceFiles.get(productPath) ?? null;
    if (entry.baseline_digest === null) {
      if (observedProductDigest !== null) throw new Error(`${id}: existing integration product requires its baseline digest`);
    } else if (digest(entry.baseline_digest, `${id} product baseline digest`) !== observedProductDigest) {
      throw new Error(`${id}: product baseline does not match the frozen source inventory`);
    }
    const prior = paths.get(productPath);
    if (prior) throw new Error(`${id}: integration product path is already owned by ${prior}`);
    paths.set(productPath, id);
    const verification = parseIntegrationProductVerification(entry.verification, id, productPath);
    const lifecycle = parseIntegrationProductLifecycle(entry.lifecycle, verification, id);
    products.set(id, { product_id: id, ...entry, verification, lifecycle });
  }
  return products;
}

export function parseIntegrationProductReferences(value, id) {
  const references = array(value, `${id} integration product references`, { empty: true })
    .map((entry) => stableId(entry, `${id} integration product reference`));
  if (new Set(references).size !== references.length
    || JSON.stringify(references) !== JSON.stringify([...references].sort(compareCodePoint))) {
    throw new Error(`${id}: integration product references must be unique and canonically ordered`);
  }
  return references;
}

export function resolveIntegrationProducts(references, products, id) {
  return references.map((productId) => {
    const product = products.get(productId);
    if (!product) throw new Error(`${id}: integration product reference ${productId} is not globally reviewed`);
    return {
      kind: "file",
      product_id: productId,
      path: product.path,
      producer: product.producer,
    };
  });
}

export function reviewedIntegrationProducts(references, products, id) {
  const productIds = [...new Set([
    ...references,
    ...[...products.values()]
      .filter((product) => product.lifecycle.kind === "reviewed-baseline-only")
      .map((product) => product.product_id),
  ])].sort(compareCodePoint);
  return productIds.map((productId) => {
    const product = products.get(productId);
    if (!product) throw new Error(`${id}: integration product reference ${productId} is not globally reviewed`);
    return {
      product_id: product.product_id,
      path: product.path,
      producer: product.producer,
      generator: {
        path: product.generator.path,
        baseline_digest: product.generator.baseline_digest,
      },
      baseline_digest: product.baseline_digest,
      verification: structuredClone(product.verification),
    };
  });
}

export function parseIntegrationProductVerification(value, id, productPath) {
  if (value?.kind === "rust_module_composition") {
    return parseModuleCompositionContract(value, productPath, id);
  }
  exact(value, ["kind"], `${id} integration product verification`);
  if (!["content_identity", "native_wasm_registration_manifest"].includes(value.kind)) {
    throw new Error(`${id}: unsupported integration product verification contract ${value.kind}`);
  }
  return { kind: value.kind };
}

function parseIntegrationProductLifecycle(value, verification, id) {
  if (value === undefined) return { kind: "bundle-referenced" };
  exact(value, ["kind"], `${id} integration product lifecycle`);
  if (!["bundle-referenced", "reviewed-baseline-only"].includes(value.kind)) {
    throw new Error(`${id}: unsupported integration product lifecycle ${value.kind}`);
  }
  if (value.kind === "reviewed-baseline-only"
    && verification.kind !== "rust_module_composition") {
    throw new Error(`${id}: reviewed-baseline-only lifecycle is restricted to module composition products`);
  }
  return { kind: value.kind };
}

export function validateIntegrationProductCoverage(bundles, products) {
  const referenced = new Set();
  for (const bundle of bundles.values()) {
    for (const productId of bundle.integration_product_refs) {
      const product = products.get(productId);
      if (!product) throw new Error(`${bundle.id}: unknown integration product ${productId}`);
      if (product.lifecycle.kind === "reviewed-baseline-only") {
        throw new Error(`${bundle.id}: reviewed-baseline-only integration product ${productId} cannot be bundle referenced`);
      }
      referenced.add(productId);
    }
  }
  const expected = [...products.values()]
    .filter((product) => product.lifecycle.kind === "bundle-referenced")
    .map((product) => product.product_id)
    .sort(compareCodePoint);
  const observed = [...referenced].sort(compareCodePoint);
  if (JSON.stringify(observed) !== JSON.stringify(expected)) {
    throw new Error("global integration products must exactly equal the products referenced by bundles");
  }
}

export function validateGeneratedRegistryCoverage(inventory, bundles, identities, products) {
  const productsByPath = new Map([...products.values()].map((entry) => [entry.path, entry.product_id]));
  for (const row of inventory.identities) {
    const registryPaths = row.dependencies?.generated_registry ?? [];
    if (registryPaths.length === 0) continue;
    const bundleId = identities.get(row.identity)?.bundle_id;
    const bundle = bundles.get(bundleId);
    if (!bundle) throw new Error(`${row.identity}: generated-registry coverage has no owning bundle`);
    for (const registryPath of registryPaths) {
      const productId = productsByPath.get(registryPath);
      if (!productId) {
        throw new Error(`${row.identity}: generated registry ${registryPath} is not globally reviewed`);
      }
      if (!bundle.integration_product_refs.includes(productId)) {
        throw new Error(`${row.identity}: bundle does not reference generated registry ${productId}`);
      }
    }
  }
}
