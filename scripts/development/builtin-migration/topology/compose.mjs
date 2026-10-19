import { compareCodePoint } from "../constants.mjs";
import { parseControlDraft } from "../control-draft.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { parseInventoryEvidence } from "../inventory.mjs";
import { digest } from "../schema.mjs";
import { parseAuthorityComponentGraph } from "./component-graph.mjs";
import { TOPOLOGY_CANDIDATE_KIND, TOPOLOGY_CANDIDATE_VERSION } from "./candidate.mjs";
import { validateTopologyClaims } from "./components.mjs";
import { parseStabilityCorrectionArtifact } from "./corrections.mjs";
import { parseReconciliationArtifact } from "./reconciliation.mjs";
import { parseCohortReview, parseReviewedBundle } from "./reviews.mjs";
import { TOPOLOGY_PROGRAM } from "./schema.mjs";
import {
  applyReviewedBundleUpdates,
  claimsFromBundles,
  requireClaimsEqual,
  topologyStateFromReviews,
} from "./state.mjs";

export function composeTopologyCandidate({
  baselineInventory: baselineValue,
  componentGraph: componentGraphValue,
  controlDraft: controlDraftValue,
  reviewValues,
  reconciliationValue,
  stabilityCorrectionsValue,
}) {
  const baseline = parseInventoryEvidence(baselineValue);
  const componentGraph = parseAuthorityComponentGraph(componentGraphValue, baselineValue);
  const controlDraft = parseControlDraft(controlDraftValue, baselineValue);
  const reviews = parseReviews(reviewValues, componentGraph.index);
  const baselineBinding = {
    revision: baseline.source.revision,
    inventory_digest: baseline.digest,
    component_graph_digest: componentGraph.digest,
    control_draft_digest: controlDraft.digest,
  };
  const reviewDigests = Object.fromEntries([...reviews].map(([role, review]) => [role, review.digest]));
  const initial = topologyStateFromReviews(reviews);
  requireMixedFamilySourceEvidence(initial.bundles, componentGraph.sharedSources);

  const reconciliation = parseReconciliationArtifact(reconciliationValue, {
    baseline: baselineBinding,
    reviewDigests,
    componentIndex: componentGraph.index,
    claims: initial.claims,
  });
  const reconciledBundles = applyReviewedBundleUpdates(
    initial.bundles,
    reconciliation.bundleUpdates,
    reconciliation.deletedBundles,
  );
  requireClaimsEqual(claimsFromBundles(reconciledBundles), reconciliation.claims, "topology reconciliation");
  requireMixedFamilySourceEvidence(reconciledBundles, componentGraph.sharedSources);

  const stabilityCorrections = parseStabilityCorrectionArtifact(stabilityCorrectionsValue, {
    baseline: baselineBinding,
    reviewDigests,
    reconciliationDigest: reconciliation.digest,
    componentIndex: componentGraph.index,
    claims: claimsFromBundles(reconciledBundles),
    bundles: reconciledBundles,
  });
  const finalBundles = applyReviewedBundleUpdates(
    reconciledBundles,
    stabilityCorrections.bundleUpdates,
    stabilityCorrections.deletedBundles,
  );
  requireClaimsEqual(claimsFromBundles(finalBundles), stabilityCorrections.result.claims, "topology stability corrections");
  requireMixedFamilySourceEvidence(finalBundles, componentGraph.sharedSources);

  return buildCandidate({
    baseline,
    componentGraph,
    controlDraft,
    reviews,
    reconciliationDigest: reconciliation.digest,
    stabilityCorrectionsDigest: stabilityCorrections.digest,
    bundles: [...finalBundles.values()],
  });
}

function buildCandidate({ baseline, componentGraph, controlDraft, reviews, reconciliationDigest, stabilityCorrectionsDigest, bundles: bundleValues }) {
  const finalBundles = new Map();
  for (const [position, value] of bundleValues.entries()) {
    const bundle = parseReviewedBundle(value, position, componentGraph.index);
    if (finalBundles.has(bundle.id)) throw new Error(`duplicate final topology bundle ${bundle.id}`);
    finalBundles.set(bundle.id, bundle);
  }
  const orderedBundles = new Map([...finalBundles].sort(([left], [right]) => compareCodePoint(left, right)));
  if (!same([...finalBundles.keys()], [...orderedBundles.keys()])) throw new Error("final topology bundles must use canonical order");
  const claims = claimsFromBundles(orderedBundles);
  const compositions = new Map([...orderedBundles].map(([id, bundle]) => [id, bundle.composition]));
  const conservation = validateTopologyClaims(componentGraph.index, claims, { compositions });
  const inventoryByIdentity = new Map(baseline.identities.map((row) => [row.identity, row]));
  const identityRows = new Map();
  for (const bundle of orderedBundles.values()) {
    for (const target of bundle.identity_targets) {
      const inventory = inventoryByIdentity.get(target.identity);
      if (!inventory) throw new Error(`${target.identity}: final topology identity is absent from reviewed baseline`);
      if (identityRows.has(target.identity)) throw new Error(`${target.identity}: final topology identity occurs more than once`);
      identityRows.set(target.identity, {
        identity: target.identity,
        bundle_id: bundle.id,
        cohort: bundle.cohort,
        domain: target.domain,
        family: target.family,
        disposition: structuredClone(inventory.disposition),
        classification: target.classification,
        evidence: [...target.evidence],
      });
    }
  }
  const expectedIdentities = baseline.identities.map((row) => row.identity).sort(compareCodePoint);
  const actualIdentities = [...identityRows.keys()].sort(compareCodePoint);
  if (!same(actualIdentities, expectedIdentities)) throw new Error("final topology identities differ from the reviewed baseline identity set");
  const identities = Object.fromEntries(actualIdentities.map((identity) => [identity, identityRows.get(identity)]));
  const payload = {
    schema_version: TOPOLOGY_CANDIDATE_VERSION,
    kind: TOPOLOGY_CANDIDATE_KIND,
    authority: "composed-unreviewed-candidate",
    program: TOPOLOGY_PROGRAM,
    baseline: {
      revision: baseline.source.revision,
      inventory_digest: baseline.digest,
      component_graph_digest: componentGraph.digest,
      control_draft_digest: controlDraft.digest,
    },
    inputs: {
      c01_c03_review: reviews.get("c01_c03").digest,
      c04_c05_review: reviews.get("c04_c05").digest,
      c06_c07_review: reviews.get("c06_c07").digest,
      reconciliation: digest(reconciliationDigest, "topology reconciliation digest"),
      stability_corrections: digest(stabilityCorrectionsDigest, "topology stability corrections digest"),
    },
    bundles: Object.fromEntries(orderedBundles),
    identities,
    summary: {
      components: conservation.summary.components,
      identities: conservation.summary.identities,
      bundles: conservation.summary.bundles,
      cohorts: [...new Set([...orderedBundles.values()].map((bundle) => bundle.cohort))].sort(compareCodePoint),
      target_packages: new Set([...orderedBundles.values()].flatMap((bundle) => bundle.composition.target_packages.map((target) => `${target.domain}/${target.family}`))).size,
    },
    review: { status: "unreviewed", evidence: [] },
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function requireMixedFamilySourceEvidence(bundles, sharedSources) {
  for (const bundle of bundles.values()) {
    if (bundle.composition.kind !== "mixed-family-component") continue;
    const component = bundle.authority_components[0];
    const expected = sharedSources.get(component);
    if (!expected || JSON.stringify(bundle.composition.shared_authority_sources) !== JSON.stringify(expected)) {
      throw new Error(`${bundle.id}: mixed-family shared authority sources differ from the frozen component graph`);
    }
  }
}

function parseReviews(values, componentIndex) {
  if (!(values instanceof Map)) throw new Error("topology review values must be a role-keyed Map");
  const expected = new Map([
    ["c01_c03", ["C01", "C02", "C03"]],
    ["c04_c05", ["C04", "C05"]],
    ["c06_c07", ["C06", "C07"]],
  ]);
  if (JSON.stringify([...values.keys()].sort()) !== JSON.stringify([...expected.keys()].sort())) {
    throw new Error("topology review values must contain exactly the three cohort review roles");
  }
  return new Map([...values].map(([role, value]) => {
    const review = parseCohortReview(value, componentIndex);
    if (JSON.stringify([...review.cohorts]) !== JSON.stringify(expected.get(role))) {
      throw new Error(`${role}: topology review declares the wrong cohort set`);
    }
    return [role, review];
  }));
}

function same(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}
