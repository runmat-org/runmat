import { evidenceDigest } from "../evidence.mjs";
import { parseInventoryEvidence } from "../inventory.mjs";
import { stableId } from "../schema.mjs";
import { assertValidatedTopologyView } from "../topology/freeze.mjs";
import { parseBundleControlReview } from "./bundle-review.mjs";
import { parseGlobalControlReview } from "./global-review.mjs";
import { sealReviewerAuthoredPayload } from "./review-input.mjs";
import { validateBundleControlReferences } from "./review-set.mjs";
import { assertValidatedControlOverlayScaffold } from "./scaffold.mjs";

const REPORT_KIND = "runmat-builtin-migration-control-review-validation";

export function validateGlobalControlReviewInput(value, { scaffold, topology, inventory }) {
  const context = validatedContext(scaffold, topology, inventory);
  const sealed = sealReviewerAuthoredPayload(value, "global control review input");
  const review = parseGlobalControlReview(
    sealed, context.scaffold, context.topology, context.inventory,
  );
  return validationReport("global", null, review.digest, context);
}

export function validateBundleControlReviewInput(
  value, expectedBundleId, globalValue, { scaffold, topology, inventory },
) {
  const context = validatedContext(scaffold, topology, inventory);
  const expected = stableId(expectedBundleId, "expected bundle review id");
  const sealedGlobal = sealReviewerAuthoredPayload(
    globalValue, "global control review input",
  );
  const globalReview = parseGlobalControlReview(
    sealedGlobal, context.scaffold, context.topology, context.inventory,
  );
  const sealedBundle = sealReviewerAuthoredPayload(value, "bundle control review input");
  const bundleReview = parseBundleControlReview(
    sealedBundle, context.scaffold, context.topology, context.inventory, globalReview,
  );
  if (bundleReview.bundleId !== expected) {
    throw new Error(
      `bundle control review is for ${bundleReview.bundleId}, not expected bundle ${expected}`,
    );
  }
  validateBundleControlReferences(
    bundleReview, globalReview, context.inventory, context.topology,
  );
  return validationReport("bundle", expected, bundleReview.digest, context, globalReview.digest);
}

function validatedContext(scaffoldValue, topology, inventoryValue) {
  return {
    scaffold: assertValidatedControlOverlayScaffold(scaffoldValue),
    topology: assertValidatedTopologyView(topology),
    inventory: parseInventoryEvidence(inventoryValue),
  };
}

function validationReport(subjectKind, bundleId, reviewDigest, context, globalReviewDigest = null) {
  const payload = {
    schema_version: 1,
    kind: REPORT_KIND,
    authority: "development-validation-evidence-only",
    result: "pass",
    subject: subjectKind === "global"
      ? { kind: "global" }
      : { kind: "bundle", bundle_id: bundleId },
    bindings: {
      inventory_digest: context.inventory.digest,
      topology_digest: context.topology.digest,
      scaffold_digest: context.scaffold.digest,
      global_review_digest: globalReviewDigest,
      review_digest: reviewDigest,
    },
  };
  return { ...payload, digest: evidenceDigest(payload) };
}
