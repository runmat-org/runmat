import { canonicalJson } from "./evidence.mjs";
import { expectedRegistrationManifestIdentity, parseGeneratedProductDefinitions, registrationManifestIdentity } from "./generated-products.mjs";
import { bindModuleCompositionProjection } from "./module-composition/binding.mjs";
import { exact, kind } from "./schema.mjs";

const INPUT_KIND = "runmat-builtin-generated-products-input";

export function stageGeneratedProductsInput(products, nativeRegistrationManifest, compositionProjectionValue = null) {
  const parsedProducts = parseGeneratedProductDefinitions(products);
  const needsManifest = parsedProducts.some((entry) => entry.verification.kind === "native_wasm_registration_manifest");
  const compositionProjection = bindModuleCompositionProjection(parsedProducts, compositionProjectionValue);
  const envelope = {
    schema_version: 3,
    kind: INPUT_KIND,
    products: parsedProducts,
    native_registration_manifest: needsManifest
      ? expectedRegistrationManifestIdentity(nativeRegistrationManifest)
      : null,
    module_composition_projection: compositionProjection,
  };
  return { stdin: `${canonicalJson(envelope)}\n`, integration_products: envelope.products, module_composition_projection: compositionProjection };
}

export function readGeneratedProductsInput(encoded) {
  let value;
  try { value = JSON.parse(encoded); } catch (error) { throw new Error(`generated product input is not valid JSON: ${error.message}`); }
  kind(value, 3, INPUT_KIND, "generated product input");
  exact(value, ["schema_version", "kind", "products", "native_registration_manifest", "module_composition_projection"], "generated product input");
  const products = parseGeneratedProductDefinitions(value.products);
  const needsManifest = products.some((entry) => entry.verification.kind === "native_wasm_registration_manifest");
  const nativeRegistrationManifest = needsManifest
    ? registrationManifestIdentity(value.native_registration_manifest)
    : value.native_registration_manifest;
  if (!needsManifest && nativeRegistrationManifest !== null) throw new Error("unused native registration manifest is not admitted");
  const moduleCompositionProjection = bindModuleCompositionProjection(products, value.module_composition_projection);
  return { products, native_registration_manifest: nativeRegistrationManifest, module_composition_projection: moduleCompositionProjection };
}
