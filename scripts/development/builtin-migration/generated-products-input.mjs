import { canonicalJson } from "./evidence.mjs";
import { expectedRegistrationManifestIdentity, parseGeneratedProductDefinitions, registrationManifestIdentity } from "./generated-products.mjs";
import { exact, kind } from "./schema.mjs";

const INPUT_KIND = "runmat-builtin-generated-products-input";

export function stageGeneratedProductsInput(products, nativeRegistrationManifest) {
  const parsedProducts = parseGeneratedProductDefinitions(products);
  const needsManifest = parsedProducts.some((entry) => entry.verification.kind === "native_wasm_registration_manifest");
  const envelope = {
    schema_version: 2,
    kind: INPUT_KIND,
    products: parsedProducts,
    native_registration_manifest: needsManifest
      ? expectedRegistrationManifestIdentity(nativeRegistrationManifest)
      : null,
  };
  return { stdin: `${canonicalJson(envelope)}\n`, integration_products: envelope.products };
}

export function readGeneratedProductsInput(encoded) {
  let value;
  try { value = JSON.parse(encoded); } catch (error) { throw new Error(`generated product input is not valid JSON: ${error.message}`); }
  kind(value, 2, INPUT_KIND, "generated product input");
  exact(value, ["schema_version", "kind", "products", "native_registration_manifest"], "generated product input");
  const products = parseGeneratedProductDefinitions(value.products);
  const needsManifest = products.some((entry) => entry.verification.kind === "native_wasm_registration_manifest");
  const nativeRegistrationManifest = needsManifest
    ? registrationManifestIdentity(value.native_registration_manifest)
    : value.native_registration_manifest;
  if (!needsManifest && nativeRegistrationManifest !== null) throw new Error("unused native registration manifest is not admitted");
  return { products, native_registration_manifest: nativeRegistrationManifest };
}
