import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import {
  candidateInputDigests, freezeReviewedTopology, parseReviewedTopology, reviewedTopologyView,
} from "../topology/freeze.mjs";

export function pilotFixture() {
  const topology = topologyFixture();
  const prerequisites = new Map([
    ["pilot-c01", []],
    ["pilot-c02", [prerequisite("pilot-c01")]],
    ["pilot-c03", [prerequisite("pilot-c02")]],
    ["pilot-c04", [prerequisite("pilot-c03")]],
    ["pilot-c05", [prerequisite("pilot-c04")]],
    ["pilot-c06", [prerequisite("pilot-c05")]],
    ["pilot-c07", [prerequisite("pilot-c06")]],
    ["unselected-c01", []],
  ]);
  const cohorts = [
    cohortCount("C01", 1, 1, 1, 0),
    cohortCount("C02", 1, 2, 1, 1),
    cohortCount("C03", 1, 1, 1, 0),
    cohortCount("C04", 1, 1, 1, 0),
    cohortCount("C05", 1, 1, 1, 0),
    cohortCount("C06", 1, 1, 1, 0),
    cohortCount("C07", 1, 1, 1, 0),
  ];
  const policy = {
    schema_version: 2,
    kind: "runmat-builtin-migration-pilot-policy",
    pilot_id: "rm-1064-accelerated",
    waves: [1, 2, 3, 4, 5, 6, 7].map((order) => ({
      wave_id: `wave-${order}`,
      order,
      bundle_ids: [`pilot-c0${order}`],
    })),
    derived_counts: {
      bundles: 7,
      identities: 8,
      public_identities: 7,
      internal_identities: 1,
      cohorts,
    },
    admission: {
      minimum_public_identities_per_aggregate_hour: { numerator: 9, denominator: 2 },
      maximum_elapsed_milliseconds: 172_800_000,
      required_waived_gate_count: 0,
      ordinary_gate_policy: "all-required-gates-must-pass",
      below_target_obligation: {
        production_transition: "requires-reviewer-accepted-obligation",
        required_findings: ["concrete-limiter", "revised-forecast"],
      },
    },
    review: { status: "reviewed", evidence: ["reviewed pilot selection and admission"] },
  };
  return { topology, prerequisites, policy };
}

export function prerequisite(bundleId) {
  return { bundle_id: bundleId, kind: "semantic" };
}

export function cohortCount(cohort, bundles, identities, publicIdentities, internalIdentities) {
  return {
    cohort,
    bundles,
    identities,
    public_identities: publicIdentities,
    internal_identities: internalIdentities,
  };
}

export function reviewedPilotPolicy(topology, prerequisitesByBundle, selectedBundleIds = null) {
  const selected = new Set(selectedBundleIds ?? [...topology.bundles.keys()]);
  const remaining = new Set(selected);
  const waves = [];
  while (remaining.size) {
    const ready = [...remaining].filter((bundleId) => prerequisitesByBundle.get(bundleId)
      .every((entry) => selected.has(entry.bundle_id) && !remaining.has(entry.bundle_id)))
      .sort(compareCodePoint);
    if (!ready.length) throw new Error("fixture pilot selection is not prerequisite-closed and acyclic");
    const order = waves.length + 1;
    waves.push({ wave_id: `wave-${order}`, order, bundle_ids: ready });
    for (const bundleId of ready) remaining.delete(bundleId);
  }
  const counts = new Map();
  let publicIdentities = 0;
  let internalIdentities = 0;
  for (const bundleId of selected) {
    const bundle = topology.bundles.get(bundleId);
    const count = counts.get(bundle.cohort) ?? cohortCount(bundle.cohort, 0, 0, 0, 0);
    count.bundles += 1;
    for (const identityId of bundle.identities) {
      count.identities += 1;
      if (topology.identities.get(identityId).disposition.kind === "internal") {
        count.internal_identities += 1;
        internalIdentities += 1;
      } else {
        count.public_identities += 1;
        publicIdentities += 1;
      }
    }
    counts.set(bundle.cohort, count);
  }
  return {
    schema_version: 2,
    kind: "runmat-builtin-migration-pilot-policy",
    pilot_id: "fixture-accelerated-pilot",
    waves,
    derived_counts: {
      bundles: selected.size,
      identities: publicIdentities + internalIdentities,
      public_identities: publicIdentities,
      internal_identities: internalIdentities,
      cohorts: [...counts.values()].sort((left, right) => compareCodePoint(left.cohort, right.cohort)),
    },
    admission: {
      minimum_public_identities_per_aggregate_hour: { numerator: 1, denominator: 1 },
      maximum_elapsed_milliseconds: 172_800_000,
      required_waived_gate_count: 0,
      ordinary_gate_policy: "all-required-gates-must-pass",
      below_target_obligation: {
        production_transition: "requires-reviewer-accepted-obligation",
        required_findings: ["concrete-limiter", "revised-forecast"],
      },
    },
    review: { status: "reviewed", evidence: ["fixture pilot policy review"] },
  };
}

function topologyFixture() {
  const bundleRows = [];
  const identityRows = [];
  for (let order = 1; order <= 7; order += 1) {
    const cohort = `C0${order}`;
    const bundleId = `pilot-c0${order}`;
    const identities = [`public${order}`];
    if (order === 2) identities.push("__internal2");
    bundleRows.push([bundleId, topologyBundle(bundleId, cohort, identities)]);
    for (const identity of identities) {
      identityRows.push([identity, topologyIdentity(identity, bundleId, cohort)]);
    }
  }
  bundleRows.push([
    "unselected-c01",
    topologyBundle("unselected-c01", "C01", ["unselected"]),
  ]);
  identityRows.push([
    "unselected",
    topologyIdentity("unselected", "unselected-c01", "C01"),
  ]);
  const candidatePayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-topology-candidate",
    authority: "composed-unreviewed-candidate",
    program: "RM-1064/C00-C07",
    baseline: {
      revision: `git:${"1".repeat(40)}`,
      inventory_digest: digest("a"),
      component_graph_digest: digest("b"),
      control_draft_digest: digest("c"),
    },
    inputs: {
      c01_c03_review: digest("d"),
      c04_c05_review: digest("e"),
      c06_c07_review: digest("f"),
      reconciliation: digest("1"),
      stability_corrections: digest("2"),
    },
    bundles: Object.fromEntries(bundleRows),
    identities: Object.fromEntries(identityRows),
    summary: {
      components: bundleRows.length,
      identities: identityRows.length,
      bundles: bundleRows.length,
      cohorts: ["C01", "C02", "C03", "C04", "C05", "C06", "C07"],
      target_packages: 1,
    },
    review: { status: "unreviewed", evidence: [] },
  };
  const candidate = { ...candidatePayload, digest: evidenceDigest(candidatePayload) };
  const attestation = {
    schema_version: 1,
    kind: "runmat-builtin-topology-attestation",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    candidate_digest: candidate.digest,
    input_digests: candidateInputDigests(candidate),
    review: { status: "reviewed", evidence: ["independent fixture topology review"] },
  };
  const frozen = freezeReviewedTopology(candidate, attestation, candidate);
  return reviewedTopologyView(parseReviewedTopology(frozen, candidate, attestation, candidate));
}

function topologyBundle(id, cohort, identities) {
  return {
    id,
    cohort,
    authority_components: [`component-${id}`],
    identities,
    atomic_reason: "Fixture atomic bundle",
    composition: {
      kind: "single-component",
      target_packages: [{ domain: "math", family: "basic" }],
      authored_write_set: [],
      shared_authority_sources: [],
      evidence: ["fixture topology review"],
    },
    identity_targets: identities.map((identity) => ({
      identity,
      domain: "math",
      family: "basic",
      classification: "preserved",
      evidence: ["fixture topology review"],
    })),
    review: { status: "reviewed", evidence: ["fixture topology review"] },
  };
}

function topologyIdentity(identity, bundleId, cohort) {
  const internal = identity.startsWith("__");
  return {
    identity,
    bundle_id: bundleId,
    cohort,
    domain: "math",
    family: "basic",
    disposition: internal
      ? { kind: "internal", canonical: null, reason: "Fixture helper", source: "reviewed-input" }
      : { kind: "canonical", canonical: null, reason: null, source: "reviewed-input" },
    classification: "preserved",
    evidence: ["fixture topology review"],
  };
}

function digest(character) {
  return `sha256:${character.repeat(64)}`;
}
