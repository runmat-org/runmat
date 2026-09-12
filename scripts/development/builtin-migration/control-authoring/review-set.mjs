import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { validateIntegrationProductCoverage } from "../integration-products.mjs";
import { parseGatePlans } from "../gate-plan.mjs";
import { parseInventoryEvidence } from "../inventory.mjs";
import { validateModuleCompositionControl } from "../module-composition/control.mjs";
import { array, digest, exact, kind, repositoryPath, stableId } from "../schema.mjs";
import { assertValidatedTopologyView } from "../topology/freeze.mjs";
import { parseBundleControlReview } from "./bundle-review.mjs";
import { parseGlobalControlReview } from "./global-review.mjs";
import { assertValidatedControlOverlayScaffold } from "./scaffold.mjs";

export const CONTROL_REVIEW_SET_KIND = "runmat-builtin-migration-control-review-set";
const PROGRAM = "RM-1064/C00-C07";
const VALIDATED_REVIEW_SETS = new WeakSet();

export function loadControlReviewSet(manifestPath, { scaffold, topology, inventory: inventoryValue } = {}) {
  if (!scaffold || !topology || !inventoryValue) throw new Error("control review-set loading requires the exact scaffold, reviewed topology, and inventory");
  const inventory = parseInventoryEvidence(inventoryValue);
  assertValidatedControlOverlayScaffold(scaffold);
  assertValidatedTopologyView(topology);
  assertInputBindings(scaffold, topology, inventory);
  const canonicalManifest = fs.realpathSync(path.resolve(manifestPath));
  const directory = fs.realpathSync(path.dirname(canonicalManifest));
  const manifestBytes = fs.readFileSync(canonicalManifest);
  const value = parseManifest(parseJson(manifestBytes, "control review-set manifest"), scaffold, topology, inventory);
  const globalBytes = readBoundFile(directory, value.global_review, "global control review");
  const globalReview = parseGlobalControlReview(
    parseJson(globalBytes, "global control review"), scaffold, topology, inventory,
  );
  const bundleReviews = new Map();
  for (const reference of value.bundle_reviews) {
    const bytes = readBoundFile(directory, reference, `${reference.bundle_id} bundle control review`);
    const review = parseBundleControlReview(
      parseJson(bytes, `${reference.bundle_id} bundle control review`),
      scaffold,
      topology,
      inventory,
      globalReview,
    );
    if (review.bundleId !== reference.bundle_id) throw new Error(`${reference.bundle_id}: manifest key and bundle review differ`);
    bundleReviews.set(reference.bundle_id, review);
  }
  validateProfileReferences(bundleReviews, globalReview, inventory);
  validateIntegrationProductCoverage(
    new Map([...bundleReviews].map(([id, review]) => [id, review.bundleControl])),
    globalReview.integrationProducts,
  );
  validateBundleReferences(bundleReviews, topology);
  const moduleComposition = validateModuleCompositionControl(
    globalReview.value.module_composition_baseline,
    globalReview.integrationProducts,
    reviewedCompositionBundles(bundleReviews, topology),
  );
  const parsed = deepImmutable({
    value, digest: value.digest, bundleReviews, globalReview, moduleComposition,
  });
  VALIDATED_REVIEW_SETS.add(parsed);
  return parsed;
}

function reviewedCompositionBundles(bundleReviews, topology) {
  return new Map([...bundleReviews].map(([bundleId, review]) => {
    const control = review.bundleControl;
    return [bundleId, {
      integration_product_refs: control.integration_product_refs,
      module_composition_transition: control.module_composition_transition,
      authored_write_set: [
        ...topology.bundles.get(bundleId).composition.authored_write_set,
        ...control.additional_authored_write_set,
      ],
    }];
  }));
}

export function assertValidatedControlReviewSet(value) {
  if (!VALIDATED_REVIEW_SETS.has(value)) throw new Error("operation requires the exact validated control review set");
  return value;
}

function parseManifest(value, scaffold, topology, inventory) {
  kind(value, 1, CONTROL_REVIEW_SET_KIND, "control review-set manifest");
  exact(value, ["schema_version", "kind", "authority", "program", "bindings", "global_review", "bundle_reviews", "digest"], "control review-set manifest");
  if (value.authority !== "content-addressed-review-index-only" || value.program !== PROGRAM) throw new Error("control review-set manifest has invalid authority or program");
  exact(value.bindings, ["scaffold_digest", "topology_digest", "inventory_digest"], "control review-set bindings");
  if (digest(value.bindings.scaffold_digest, "review-set scaffold digest") !== scaffold.digest) throw new Error("control review set does not bind the exact scaffold");
  if (digest(value.bindings.topology_digest, "review-set topology digest") !== topology.digest) throw new Error("control review set does not bind the exact topology");
  if (digest(value.bindings.inventory_digest, "review-set inventory digest") !== inventory.digest) throw new Error("control review set does not bind the exact inventory");
  parseFileReference(value.global_review, "global review reference");
  const rows = array(value.bundle_reviews, "bundle review references").map((entry) => {
    exact(entry, ["bundle_id", "path", "content_digest"], "bundle review reference");
    stableId(entry.bundle_id, "bundle review reference id");
    parseFileReference(entry, `${entry.bundle_id} bundle review reference`, true);
    return entry;
  });
  const ids = rows.map((entry) => entry.bundle_id);
  const expected = [...topology.bundles.keys()].sort(compareCodePoint);
  if (JSON.stringify(ids) !== JSON.stringify(expected)) throw new Error("bundle review references must exactly cover every topology bundle in canonical order");
  const paths = [value.global_review.path, ...rows.map((entry) => entry.path)];
  if (new Set(paths).size !== paths.length) throw new Error("control review-set paths must be unique");
  digest(value.digest, "control review-set manifest digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("control review-set manifest digest mismatch");
  return value;
}

function parseFileReference(value, label, bundle = false) {
  if (!bundle) exact(value, ["path", "content_digest"], label);
  repositoryPath(value.path, `${label} path`);
  digest(value.content_digest, `${label} content digest`);
}

function readBoundFile(directory, reference, label) {
  const requested = path.resolve(directory, ...reference.path.split("/"));
  const canonical = fs.realpathSync(requested);
  const relative = path.relative(directory, canonical);
  if (!relative || relative.startsWith(`..${path.sep}`) || relative === ".." || path.isAbsolute(relative)) {
    throw new Error(`${label} path must resolve within the manifest directory`);
  }
  const bytes = fs.readFileSync(canonical);
  if (contentDigest(bytes) !== reference.content_digest) throw new Error(`${label} exact byte content digest mismatch`);
  return bytes;
}

function parseJson(bytes, label) {
  try { return JSON.parse(bytes.toString("utf8")); }
  catch (error) { throw new Error(`${label} is not valid UTF-8 JSON: ${error.message}`); }
}

function validateProfileReferences(bundleReviews, globalReview, inventory) {
  const used = new Set();
  for (const [bundleId, review] of bundleReviews) {
    const expanded = review.bundleControl.gate_plans.map(({ program_profile_id: profileId, ...row }) => {
      const profile = globalReview.programProfiles.get(profileId);
      if (!profile) throw new Error(`${bundleId}: gate plan references unknown global program profile ${profileId}`);
      used.add(profileId);
      return { ...row, program: profile.program };
    });
    parseGatePlans(expanded, bundleId, inventory);
  }
  const defined = [...globalReview.programProfiles.keys()].sort(compareCodePoint);
  const referenced = [...used].sort(compareCodePoint);
  if (JSON.stringify(referenced) !== JSON.stringify(defined)) throw new Error("global program profiles must exactly equal the set referenced by bundle gate plans");
}

function validateBundleReferences(bundleReviews, topology) {
  for (const [bundleId, review] of bundleReviews) {
    for (const prerequisite of review.bundleControl.prerequisites) {
      if (!topology.bundles.has(prerequisite.bundle_id)) {
        throw new Error(`${bundleId}: prerequisite references unknown topology bundle ${prerequisite.bundle_id}`);
      }
    }
  }
}

function assertInputBindings(scaffold, topology, inventory) {
  if (scaffold.bindings?.inventory_digest !== inventory.digest
    || scaffold.bindings?.revision !== inventory.source.revision
    || scaffold.bindings?.source_digest !== inventory.source.digest) {
    throw new Error("control authoring scaffold does not bind the exact inventory");
  }
  if (scaffold.bindings?.reviewed_topology_digest !== topology.digest
    || topology.baseline?.inventory_digest !== inventory.digest
    || topology.baseline?.control_draft_digest !== scaffold.bindings?.control_draft_digest) {
    throw new Error("control authoring scaffold does not bind the exact reviewed topology");
  }
  const { digest: _ignored, ...payload } = scaffold;
  if (evidenceDigest(payload) !== scaffold.digest) throw new Error("control authoring scaffold digest mismatch");
}
