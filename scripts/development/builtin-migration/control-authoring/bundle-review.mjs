import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { assertValidatedTopologyView } from "../topology/freeze.mjs";
import { validateModuleCompositionControl } from "../module-composition/control.mjs";
import { array, digest, exact, identity, kind, stableId } from "../schema.mjs";
import { assertValidatedControlOverlayScaffold } from "./scaffold.mjs";
import { assertValidatedGlobalControlReview } from "./global-review.mjs";
import {
  assertEvidenceDigest, assertScaffoldTopologyBinding, parseBundleControlPolicy,
  parseIdentityControlPolicy, parseReviewedEvidence,
} from "./policy-schema.mjs";

export const BUNDLE_CONTROL_REVIEW_KIND = "runmat-builtin-migration-bundle-control-review";
const PROGRAM = "RM-1064/C00-C07";
const VALIDATED_BUNDLE_REVIEWS = new WeakSet();

export function parseBundleControlReview(
  value, scaffoldValue, topology, inventory, globalReviewValue,
) {
  const scaffold = assertValidatedControlOverlayScaffold(scaffoldValue);
  const globalReview = assertValidatedGlobalControlReview(globalReviewValue);
  assertValidatedTopologyView(topology);
  assertScaffoldTopologyBinding(scaffold, topology);
  kind(value, 5, BUNDLE_CONTROL_REVIEW_KIND, "bundle control review");
  exact(value, ["schema_version", "kind", "authority", "program", "bindings", "bundle_control", "identity_controls", "review", "digest"], "bundle control review");
  if (value.authority !== "reviewer-authored-development-input" || value.program !== PROGRAM) throw new Error("bundle control review has invalid authority or program");
  const bundle = parseBindings(value.bindings, scaffold, topology);
  parseBundleControlPolicy(value.bundle_control, bundle.id, bundle.identities, inventory);
  validateModuleCompositionControl(
    globalReview.moduleCompositionBaseline,
    globalReview.integrationProducts,
    new Map([[bundle.id, reviewedCompositionBundle(bundle, value.bundle_control)]]),
    { deferEffectiveSequence: true },
  );
  const identityControls = parseIdentityControls(value.identity_controls, bundle, scaffold, topology);
  parseReviewedEvidence(value.review, `${bundle.id} bundle review artifact`);
  assertEvidenceDigest(value, "bundle control review");
  const parsed = deepImmutable({ value, digest: value.digest, bundleId: bundle.id, bundleControl: value.bundle_control, identityControls });
  VALIDATED_BUNDLE_REVIEWS.add(parsed);
  return parsed;
}

function reviewedCompositionBundle(bundle, control) {
  return {
    prerequisites: control.prerequisites,
    integration_product_refs: control.integration_product_refs,
    module_composition_transition: control.module_composition_transition,
    authored_write_set: [
      ...bundle.composition.authored_write_set,
      ...control.additional_authored_write_set,
    ],
  };
}

export function assertValidatedBundleControlReview(value) {
  if (!VALIDATED_BUNDLE_REVIEWS.has(value)) throw new Error("operation requires an exact validated bundle control review");
  return value;
}

function parseBindings(value, scaffold, topology) {
  exact(value, ["scaffold_digest", "topology_digest", "bundle_id", "scaffold_bundle_row_digest", "topology_bundle_row_digest", "identity_rows"], "bundle review bindings");
  if (digest(value.scaffold_digest, "bundle review scaffold digest") !== scaffold.digest) throw new Error("bundle review does not bind the exact authoring scaffold");
  if (digest(value.topology_digest, "bundle review topology digest") !== topology.digest) throw new Error("bundle review does not bind the exact reviewed topology");
  const id = stableId(value.bundle_id, "bundle review bundle id");
  const bundle = topology.bundles.get(id);
  if (!bundle) throw new Error(`bundle review references unknown topology bundle ${id}`);
  const scaffoldBundle = scaffold.bundle_rows.find((entry) => entry.bundle_id === id);
  if (!scaffoldBundle || digest(value.scaffold_bundle_row_digest, `${id} scaffold bundle digest`) !== evidenceDigest(scaffoldBundle)) throw new Error(`${id}: bundle review scaffold row digest mismatch`);
  if (digest(value.topology_bundle_row_digest, `${id} topology bundle digest`) !== evidenceDigest(bundle)) throw new Error(`${id}: bundle review topology row digest mismatch`);
  const rows = array(value.identity_rows, `${id} bound identity rows`).map((entry) => parseIdentityBinding(entry, id));
  canonicalUnique(rows, (entry) => entry.identity.toLowerCase(), `${id} bound identity rows`);
  const expected = [...bundle.identities];
  if (JSON.stringify(rows.map((entry) => entry.identity)) !== JSON.stringify(expected)) throw new Error(`${id}: bound identities differ from the topology bundle`);
  const scaffoldRows = new Map(scaffold.identity_rows.map((entry) => [entry.identity.toLowerCase(), entry]));
  for (const row of rows) {
    const scaffoldRow = scaffoldRows.get(row.identity.toLowerCase());
    const topologyRow = topology.identities.get(row.identity.toLowerCase());
    if (!scaffoldRow || row.scaffold_identity_row_digest !== evidenceDigest(scaffoldRow)) throw new Error(`${row.identity}: scaffold identity row digest mismatch`);
    if (!topologyRow || row.topology_identity_digest !== evidenceDigest(topologyRow)) throw new Error(`${row.identity}: topology identity row digest mismatch`);
    const proposal = scaffold.authority_proposals.identity_rows
      .find((entry) => entry.identity.toLowerCase() === row.identity.toLowerCase());
    if (!proposal || row.authority_proposal_digest !== proposal.proposal_digest) {
      throw new Error(`${row.identity}: authority proposal digest mismatch`);
    }
  }
  return bundle;
}

function parseIdentityBinding(value, bundleId) {
  exact(value, ["identity", "scaffold_identity_row_digest", "topology_identity_digest", "authority_proposal_digest"], `${bundleId} identity binding`);
  identity(value.identity, `${bundleId} bound identity`);
  digest(value.scaffold_identity_row_digest, `${value.identity} scaffold identity row digest`);
  digest(value.topology_identity_digest, `${value.identity} topology identity row digest`);
  digest(value.authority_proposal_digest, "identity authority proposal digest");
  return value;
}

function parseIdentityControls(value, bundle, scaffold, topology) {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error(`${bundle.id} identity controls must be an object`);
  const keys = Object.keys(value);
  if (JSON.stringify(keys) !== JSON.stringify([...bundle.identities])) throw new Error(`${bundle.id}: identity controls must exactly cover the topology bundle in canonical order`);
  const scaffoldIds = new Set(scaffold.identity_rows.map((entry) => entry.identity.toLowerCase()));
  const controls = new Map();
  for (const id of keys) {
    const normalized = identity(id, `${bundle.id} identity control key`).toLowerCase();
    if (!scaffoldIds.has(normalized) || !topology.identities.has(normalized)) throw new Error(`${id}: identity control is absent from bound inputs`);
    const control = parseIdentityControlPolicy(value[id], id);
    controls.set(normalized, control);
  }
  return controls;
}

function canonicalUnique(values, key, label) {
  const keys = values.map(key);
  if (new Set(keys).size !== keys.length) throw new Error(`${label} must be unique`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error(`${label} must use canonical order`);
  }
}
