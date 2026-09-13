import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { loadControlReviewSet } from "../control-authoring/review-set.mjs";
import {
  fixtureCatalogCompositionChild, fixtureCompositionChild,
  fixtureIntegrationProducts, fixtureModuleCompositionBaseline,
  fixtureReviewedModuleCompositionBaseline, fixtureTargetPolicy,
} from "./helpers.mjs";
import {
  SEQUENTIAL_BUNDLES, SEQUENTIAL_IDENTITIES,
} from "./sequential-shared-parent-definition-fixture.mjs";
import { reviewedPilotPolicy } from "./pilot-policy-fixture.mjs";

export function sequentialReviewedControlSet({
  repository, inventory, topology, scaffold, bundleControls, identityControls,
}) {
  const root = path.join(path.dirname(repository), "sequential-control-review");
  fs.mkdirSync(path.join(root, "bundles"), { recursive: true });
  const { profileByProgram, programProfiles } = programProfilesFor(bundleControls);
  const baseline = fixtureReviewedModuleCompositionBaseline(
    fixtureModuleCompositionBaseline(
      new Map([
        ["catalog-math", [fixtureCatalogCompositionChild()]],
        ["runtime-math", [fixtureCompositionChild()]],
      ]),
      new Set(["catalog-math", "runtime-math"]),
    ),
  );
  const global = globalReview({
    inventory, topology, scaffold, baseline, programProfiles, bundleControls,
  });
  const globalBytes = writeJson(path.join(root, "global.json"), global);
  const bundleReviews = SEQUENTIAL_BUNDLES.map((bundleId, index) => bundleReview({
    root,
    bundleId,
    identity: SEQUENTIAL_IDENTITIES[index],
    topology,
    scaffold,
    control: bundleControls.get(bundleId),
    identityControl: identityControls.get(SEQUENTIAL_IDENTITIES[index]),
    profileByProgram,
  }));
  const manifestPayload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-review-set",
    authority: "content-addressed-review-index-only",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      inventory_digest: inventory.digest,
    },
    global_review: { path: "global.json", content_digest: contentDigest(globalBytes) },
    bundle_reviews: bundleReviews,
  };
  const manifest = { ...manifestPayload, digest: evidenceDigest(manifestPayload) };
  const manifestPath = path.join(root, "review-set.json");
  writeJson(manifestPath, manifest);
  return loadControlReviewSet(manifestPath, { scaffold, topology, inventory });
}

function programProfilesFor(bundleControls) {
  const programs = new Map();
  for (const control of bundleControls.values()) {
    for (const plan of control.gate_plans) programs.set(JSON.stringify(plan.program), plan.program);
  }
  const ordered = [...programs].sort(([left], [right]) => left.localeCompare(right));
  return {
    profileByProgram: new Map(ordered.map(([key], index) => [key, `program-${index + 1}`])),
    programProfiles: Object.fromEntries(ordered.map(([, program], index) => [
      `program-${index + 1}`,
      { program, review: { status: "reviewed", evidence: ["sequential fixture executable review"] } },
    ])),
  };
}

function globalReview({
  inventory, topology, scaffold, baseline, programProfiles, bundleControls,
}) {
  const findingRows = inventory.migration_findings.map((finding) => ({
    finding_digest: evidenceDigest(finding),
    ...finding,
    disposition: "bundle-work",
    bundle_id: SEQUENTIAL_BUNDLES[0],
    reason: "Sequential fixture migration work",
    evidence: ["sequential fixture review"],
  }));
  const payload = {
    schema_version: 6,
    kind: "runmat-builtin-migration-global-control-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      migration_finding_rows_digest: evidenceDigest(scaffold.migration_finding_rows),
      module_composition_reviewed_baseline_digest: baseline.digest,
    },
    program_profiles: programProfiles,
    integration_products: fixtureIntegrationProducts(inventory, true),
    module_composition_baseline: baseline,
    migration_findings: {
      schema_version: 1,
      kind: "runmat-builtin-migration-finding-dispositions",
      rows: findingRows,
      review: { status: "reviewed", evidence: ["sequential fixture review"] },
    },
    exception_manifest: {
      entries: [], review: { status: "reviewed", evidence: ["sequential fixture review"] },
    },
    target_policy: fixtureTargetPolicy([{
      operating_system: inventory.compiled_inventory.build.operating_system,
      architecture: inventory.compiled_inventory.build.architecture,
    }]),
    storage_policy: storagePolicy(inventory),
    pilot_policy: reviewedPilotPolicy(
      topology,
      new Map([...bundleControls].map(([bundleId, control]) => [
        bundleId, control.prerequisites,
      ])),
    ),
    review: { status: "reviewed", evidence: ["sequential fixture global review"] },
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

function bundleReview({
  root, bundleId, identity, topology, scaffold, control, identityControl,
  profileByProgram,
}) {
  const payload = {
    schema_version: 5,
    kind: "runmat-builtin-migration-bundle-control-review",
    authority: "reviewer-authored-development-input",
    program: "RM-1064/C00-C07",
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      bundle_id: bundleId,
      scaffold_bundle_row_digest: evidenceDigest(
        scaffold.bundle_rows.find((row) => row.bundle_id === bundleId),
      ),
      topology_bundle_row_digest: evidenceDigest(topology.bundles.get(bundleId)),
      identity_rows: [{
        identity,
        scaffold_identity_row_digest: evidenceDigest(
          scaffold.identity_rows.find((row) => row.identity === identity),
        ),
        topology_identity_digest: evidenceDigest(topology.identities.get(identity)),
        authority_proposal_digest: scaffold.authority_proposals.identity_rows
          .find((row) => row.identity === identity).proposal_digest,
      }],
    },
    bundle_control: {
      ...structuredClone(control),
      gate_plans: control.gate_plans.map(({ program, ...plan }) => ({
        ...plan, program_profile_id: profileByProgram.get(JSON.stringify(program)),
      })),
    },
    identity_controls: { [identity]: identityControl },
    review: { status: "reviewed", evidence: ["sequential fixture bundle review"] },
  };
  const value = { ...payload, digest: evidenceDigest(payload) };
  const relative = `bundles/${bundleId}.json`;
  const bytes = writeJson(path.join(root, relative), value);
  return { bundle_id: bundleId, path: relative, content_digest: contentDigest(bytes) };
}

function storagePolicy(inventory) {
  return {
    host_profiles: {
      "fixture-host": {
        operating_system: inventory.compiled_inventory.build.operating_system,
        architecture: inventory.compiled_inventory.build.architecture,
        execution_host: os.hostname(),
        volume_roles: {
          source_worktree: { role: "source-worktree", mount_path: "/System/Volumes/Data", filesystem_id: "posix-dev:1", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
          target_temp: { role: "target-temp", mount_path: "/private/tmp/runmat-integration-tmp", filesystem_id: "posix-dev:2", minimum_free_bytes: 1, pause_below_bytes: 2, maximum_observation_age_seconds: 60 },
        },
      },
    },
    targets_must_be_disjoint: true,
    occt_default: "disabled-unless-affected",
  };
}

function writeJson(target, value) {
  const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
  fs.writeFileSync(target, bytes);
  return bytes;
}
