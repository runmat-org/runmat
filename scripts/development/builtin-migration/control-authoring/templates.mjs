import fs from "node:fs";
import path from "node:path";

import { compareCodePoint } from "../constants.mjs";
import { contentDigest, evidenceDigest } from "../evidence.mjs";
import { parseInventoryEvidence } from "../inventory.mjs";
import { assertValidatedTopologyView } from "../topology/freeze.mjs";
import { parseBundleControlReview } from "./bundle-review.mjs";
import { parseGlobalControlReview } from "./global-review.mjs";
import { loadControlReviewSet } from "./review-set.mjs";
import { assertValidatedControlOverlayScaffold } from "./scaffold.mjs";

const PROGRAM = "RM-1064/C00-C07";

export function initializeControlReviewTemplates(directory, scaffoldValue, topology) {
  const scaffold = assertValidatedControlOverlayScaffold(scaffoldValue);
  assertValidatedTopologyView(topology);
  const root = createNewDirectory(directory);
  fs.mkdirSync(path.join(root, "bundles"));
  writeNewJson(path.join(root, "global.json"), globalTemplate(scaffold, topology));
  for (const bundleId of [...topology.bundles.keys()].sort(compareCodePoint)) {
    writeNewJson(path.join(root, "bundles", `${bundleId}.json`), bundleTemplate(bundleId, scaffold, topology));
  }
  return { directory: root, bundle_templates: topology.bundles.size, global_template: "global.json" };
}

export function indexControlReviews(reviewDirectory, outputDirectory, { scaffold: scaffoldValue, topology, inventory }) {
  const scaffold = assertValidatedControlOverlayScaffold(scaffoldValue);
  assertValidatedTopologyView(topology);
  const parsedInventory = parseInventoryEvidence(inventory);
  const source = fs.realpathSync(path.resolve(reviewDirectory));
  assertExactTemplateFiles(source, topology);

  const global = sealReviewedPayload(readJson(path.join(source, "global.json"), "global control review"), "global control review");
  parseGlobalControlReview(global, scaffold, topology);
  const bundles = [...topology.bundles.keys()].sort(compareCodePoint).map((bundleId) => {
    const relative = `bundles/${bundleId}.json`;
    const bundle = sealReviewedPayload(readJson(path.join(source, relative), `${bundleId} control review`), `${bundleId} control review`);
    parseBundleControlReview(bundle, scaffold, topology);
    return { bundleId, relative, bundle };
  });

  const root = createNewDirectory(outputDirectory);
  fs.mkdirSync(path.join(root, "bundles"));
  const globalBytes = writeNewJson(path.join(root, "global.json"), global);
  const references = [];
  for (const { bundleId, relative, bundle } of bundles) {
    const bytes = writeNewJson(path.join(root, relative), bundle);
    references.push({ bundle_id: bundleId, path: relative, content_digest: contentDigest(bytes) });
  }
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-control-review-set",
    authority: "content-addressed-review-index-only",
    program: PROGRAM,
    bindings: { scaffold_digest: scaffold.digest, topology_digest: topology.digest, inventory_digest: parsedInventory.digest },
    global_review: { path: "global.json", content_digest: contentDigest(globalBytes) },
    bundle_reviews: references,
  };
  const manifestPath = path.join(root, "review-set.json");
  writeNewJson(manifestPath, { ...payload, digest: evidenceDigest(payload) });
  loadControlReviewSet(manifestPath, { scaffold, topology, inventory: parsedInventory });
  return { directory: root, manifest: manifestPath, bundle_reviews: references.length };
}

function globalTemplate(scaffold, topology) {
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-global-control-review",
    authority: "reviewer-authored-development-input",
    program: PROGRAM,
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      migration_finding_rows_digest: evidenceDigest(scaffold.migration_finding_rows),
    },
    program_profiles: {},
    migration_findings: {
      schema_version: 1,
      kind: "runmat-builtin-migration-finding-dispositions",
      rows: scaffold.migration_finding_rows.map((row) => ({
        finding_digest: row.finding_digest,
        ...structuredClone(row.observations),
        disposition: null,
        bundle_id: null,
        reason: null,
        evidence: [],
      })),
      review: unreviewed(),
    },
    exception_manifest: { entries: [], review: unreviewed() },
    execution_targets: null,
    storage_policy: { host_profiles: {}, targets_must_be_disjoint: true, occt_default: "disabled-unless-affected" },
    review: unreviewed(),
  };
}

function bundleTemplate(bundleId, scaffold, topology) {
  const bundle = topology.bundles.get(bundleId);
  const scaffoldBundle = scaffold.bundle_rows.find((row) => row.bundle_id === bundleId);
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-bundle-control-review",
    authority: "reviewer-authored-development-input",
    program: PROGRAM,
    bindings: {
      scaffold_digest: scaffold.digest,
      topology_digest: topology.digest,
      bundle_id: bundleId,
      scaffold_bundle_row_digest: evidenceDigest(scaffoldBundle),
      topology_bundle_row_digest: evidenceDigest(bundle),
      identity_rows: bundle.identities.map((identity) => ({
        identity,
        scaffold_identity_row_digest: evidenceDigest(scaffold.identity_rows.find((row) => row.identity === identity)),
        topology_identity_digest: evidenceDigest(topology.identities.get(identity)),
      })),
    },
    bundle_control: {
      prerequisites: null,
      additional_authored_write_set: null,
      integration_outputs: null,
      gate_plans: null,
      owner_role: null,
      complexity: null,
      review: unreviewed(),
    },
    identity_controls: Object.fromEntries(bundle.identities.map((identity) => [identity, {
      public_spelling: null,
      runtime_owner: null,
      shared_dependencies: null,
      complexity: null,
      maturity: null,
      expected_authorities: null,
      expected_removals: null,
      baseline_evidence: null,
      owner: null,
      review: unreviewed(),
    }])),
    review: unreviewed(),
  };
}

function sealReviewedPayload(value, label) {
  if (Object.hasOwn(value, "digest")) throw new Error(`${label} template must not supply its own digest`);
  return { ...value, digest: evidenceDigest(value) };
}

function assertExactTemplateFiles(root, topology) {
  const expected = ["global.json", ...[...topology.bundles.keys()].sort(compareCodePoint).map((id) => `bundles/${id}.json`)].sort(compareCodePoint);
  const observed = walkFiles(root);
  if (JSON.stringify(observed) !== JSON.stringify(expected)) throw new Error("control review directory must contain exactly the expected global and bundle templates");
}

function walkFiles(root, relative = "") {
  const result = [];
  for (const entry of fs.readdirSync(path.join(root, relative), { withFileTypes: true })) {
    if (entry.isSymbolicLink()) throw new Error("control review templates cannot be symbolic links");
    const child = relative ? `${relative}/${entry.name}` : entry.name;
    if (entry.isDirectory()) result.push(...walkFiles(root, child));
    else if (entry.isFile()) result.push(child);
    else throw new Error(`unsupported control review directory entry ${child}`);
  }
  return result.sort(compareCodePoint);
}

function createNewDirectory(value) {
  const target = path.resolve(value);
  fs.mkdirSync(target);
  return fs.realpathSync(target);
}

function writeNewJson(target, value) {
  const bytes = Buffer.from(`${JSON.stringify(value, null, 2)}\n`);
  fs.writeFileSync(target, bytes, { flag: "wx" });
  return bytes;
}

function readJson(target, label) {
  try { return JSON.parse(fs.readFileSync(target, "utf8")); }
  catch (error) { throw new Error(`${label} is not valid JSON: ${error.message}`); }
}

function unreviewed() { return { status: "unreviewed", evidence: [] }; }
