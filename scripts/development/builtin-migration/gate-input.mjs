import { stageExampleGateInput } from "./example-gate-input.mjs";
import { stageGeneratedProductsInput } from "./generated-products-input.mjs";

export function prepareGateProcessInput(
  plan, inputs, temporaryDirectory, expected, moduleCompositionProjection = null,
) {
  if (plan.parser === "example_reconciliation") {
    return stageExampleGateInput(inputs, temporaryDirectory, expected);
  }
  if (plan.parser === "generated_products") {
    return stageGeneratedProductsInput(
      expected.integration_products,
      expected.native_registration_manifest,
      moduleCompositionProjection,
    );
  }
  return null;
}
