import { CATALOG_ROOT, compareCodePoint, rustLeaf } from "../constants.mjs";
import {
  bundleTypedPathEvidence, identityTypedBaselineEvidence,
} from "../baseline-evidence.mjs";
import { parseControlDraft } from "../control-draft.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { parseInventoryEvidence } from "../inventory.mjs";
import { digest, kind } from "../schema.mjs";
import { assertValidatedTopologyView } from "../topology/freeze.mjs";
import {
  buildAuthorityProposals, canonicalCallableOwnerPath, migrationFindingRoutingProposal,
} from "./authority-proposals.mjs";

const PROGRAM = "RM-1064/C00-C07";
const KIND = "runmat-builtin-migration-control-overlay-scaffold";
const AUTHORITY = "machine-derived-unreviewed-scaffold-only";
const VALIDATED_SCAFFOLDS = new WeakSet();

const BUNDLE_DECISIONS = Object.freeze([
  "prerequisites",
  "additional_authored_write_set",
  "integration_product_refs",
  "module_composition_transition",
  "source_migrations",
  "expected_removals",
  "baseline_evidence",
  "gate_plans",
  "owner_role",
  "complexity",
]);

const IDENTITY_DECISIONS = Object.freeze([
  "public_identity",
  "forms",
  "implementation",
  "shared_dependencies",
  "complexity",
  "maturity",
  "expected_authorities",
  "owner",
]);

const FINDING_DECISIONS = Object.freeze([
  "disposition",
  "bundle_id",
  "reason",
  "evidence",
]);

const GLOBAL_DECISIONS = Object.freeze([
  "exception_manifest",
  "integration_products",
  "module_composition_baseline",
  "pilot_policy",
  "target_policy",
  "storage_policy",
]);

export function buildControlOverlayScaffold(inventoryValue, draftValue, reviewedTopology) {
  const inventory = parseInventoryEvidence(inventoryValue);
  const draft = parseControlDraft(draftValue, inventory);
  const topology = assertValidatedTopologyView(reviewedTopology);
  assertBindings(inventory, draft, topology);

  const inventoryByIdentity = new Map(inventory.identities.map((row) => [row.identity, row]));
  const sourceFilesByPath = new Map(inventory.source.files.map((file) => [file.path, file]));
  const identityRows = canonicalEntries(topology.identities).map(([identity, topologyRow]) => {
    const sourceRow = inventoryByIdentity.get(identity);
    if (!sourceRow) throw new Error(`${identity}: reviewed topology identity is absent from the exact inventory`);
    return {
      identity,
      source_row_digest: evidenceDigest(sourceRow),
      topology_row_digest: evidenceDigest(topologyRow),
      observations: identityObservations(sourceRow, topologyRow, sourceFilesByPath),
      candidate_paths: identityCandidatePaths(sourceRow, topologyRow, sourceFilesByPath),
      decisions: unresolvedDecisions(IDENTITY_DECISIONS),
      review: unreviewed(),
    };
  });

  const candidatesByIdentity = new Map(identityRows.map((row) => [row.identity, row.candidate_paths]));
  const bundleRows = canonicalEntries(topology.bundles).map(([bundleId, topologyRow]) => {
    const bundleSourceRows = topologyRow.identities.map((identity) => inventoryByIdentity.get(identity));
    if (bundleSourceRows.some((row) => row === undefined)) {
      throw new Error(`${bundleId}: reviewed topology bundle refers to an identity absent from the exact inventory`);
    }
    return {
      bundle_id: bundleId,
      source_rows_digest: evidenceDigest(bundleSourceRows),
      topology_row_digest: evidenceDigest(topologyRow),
      observations: {
        cohort: topologyRow.cohort,
        identities: [...topologyRow.identities],
        target_packages: clone(topologyRow.composition.target_packages),
        topology_authored_write_set: clone(topologyRow.composition.authored_write_set),
        shared_authority_sources: [...topologyRow.composition.shared_authority_sources],
        typed_paths: bundleTypedPathEvidence(inventory, topologyRow.identities),
      },
      candidate_paths: mergeCandidatePaths(topologyRow.identities.map((identity) => candidatesByIdentity.get(identity))),
      decisions: unresolvedDecisions(BUNDLE_DECISIONS),
      review: unreviewed(),
    };
  });

  const findingRows = inventory.migration_findings
    .map((finding) => ({
      finding_digest: evidenceDigest(finding),
      observations: clone(finding),
      routing_proposal: migrationFindingRoutingProposal(finding, topology, inventoryByIdentity),
      decisions: unresolvedDecisions(FINDING_DECISIONS),
      review: unreviewed(),
    }))
    .sort((left, right) => compareCodePoint(left.finding_digest, right.finding_digest));

  const payload = {
    schema_version: 6,
    kind: KIND,
    authority: AUTHORITY,
    program: PROGRAM,
    bindings: {
      revision: inventory.source.revision,
      source_digest: inventory.source.digest,
      inventory_digest: inventory.digest,
      control_draft_digest: draft.digest,
      reviewed_topology_digest: topology.digest,
    },
    bundle_rows: bundleRows,
    identity_rows: identityRows,
    migration_finding_rows: findingRows,
    authority_proposals: buildAuthorityProposals(inventoryByIdentity, topology, inventory.migration_findings),
    decisions: unresolvedDecisions(GLOBAL_DECISIONS),
    review: unreviewed(),
  };
  const scaffold = deepImmutable({ ...payload, digest: evidenceDigest(payload) });
  VALIDATED_SCAFFOLDS.add(scaffold);
  return scaffold;
}

export function parseControlOverlayScaffold(value, inventoryValue, draftValue, reviewedTopology) {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("control overlay scaffold must be an object");
  }
  if (value.authority !== AUTHORITY) {
    throw new Error("control overlay scaffold cannot claim or accept reviewed authority");
  }
  kind(value, 6, KIND, "control overlay scaffold");
  if (value.review?.status !== "unreviewed") {
    throw new Error("control overlay scaffold cannot claim review");
  }
  const expected = buildControlOverlayScaffold(inventoryValue, draftValue, reviewedTopology);
  digest(value.digest, "control overlay scaffold digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("control overlay scaffold digest mismatch");
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("control overlay scaffold differs from deterministic reconstruction");
  }
  return expected;
}

export function assertValidatedControlOverlayScaffold(value) {
  if (!VALIDATED_SCAFFOLDS.has(value)) {
    throw new Error("operation requires the exact deterministically validated control overlay scaffold");
  }
  return value;
}

function assertBindings(inventory, draft, topology) {
  if (topology.value?.program !== PROGRAM || draft.program !== PROGRAM) {
    throw new Error("control overlay scaffold inputs must belong to the RM-1064 program");
  }
  if (topology.baseline.inventory_digest !== inventory.digest) {
    throw new Error("reviewed topology does not bind the exact inventory");
  }
  if (topology.baseline.control_draft_digest !== draft.digest) {
    throw new Error("reviewed topology does not bind the exact pre-topology control draft");
  }
  if (topology.baseline.revision !== inventory.source.revision) {
    throw new Error("reviewed topology revision differs from the exact inventory");
  }
  const draftIdentities = draft.identity_rows.map((row) => row.identity);
  const inventoryIdentities = inventory.identities.map((row) => row.identity).sort(compareCodePoint);
  const topologyIdentities = [...topology.identities.keys()].sort(compareCodePoint);
  if (JSON.stringify(draftIdentities) !== JSON.stringify(inventoryIdentities)
    || JSON.stringify(topologyIdentities) !== JSON.stringify(inventoryIdentities)) {
    throw new Error("control overlay scaffold inputs do not share the exact identity set");
  }
}

function identityObservations(row, topologyRow, sourceFilesByPath) {
  return {
    topology: {
      bundle_id: topologyRow.bundle_id,
      cohort: topologyRow.cohort,
      disposition: clone(topologyRow.disposition),
      domain: topologyRow.domain,
      family: topologyRow.family,
      classification: topologyRow.classification,
    },
    source: {
      spellings: [...row.spellings].sort(compareCodePoint),
      disposition: clone(row.disposition),
      source_metrics: {
        files: row.source_metrics?.files?.length ?? 0,
        total_lines: row.source_metrics?.total_lines ?? 0,
        maximum_file_lines: row.source_metrics?.maximum_file_lines ?? 0,
      },
      authority_counts: {
        catalog_entries: row.semantic_authority?.catalog_entries?.length ?? 0,
        catalog_aliases: row.semantic_authority?.catalog_aliases?.length ?? 0,
        catalog_constants: row.semantic_authority?.constants?.length ?? 0,
        legacy_functions: row.semantic_authority?.legacy_functions?.length ?? 0,
        legacy_documentation: row.semantic_authority?.legacy_documentation?.length ?? 0,
        canonical_runtime_bindings: row.semantic_authority?.runtime_bindings?.length ?? 0,
        canonical_runtime_constants: row.semantic_authority?.runtime_constants?.length ?? 0,
        implementation_provenance: row.semantic_authority?.implementation_provenance?.length ?? 0,
        observed_runtime_registrations: row.registrations?.runtime?.length ?? 0,
        observed_wasm_registrations: row.registrations?.wasm?.length ?? 0,
        canonical_provider_records: row.semantic_authority?.gpu_specs?.length ?? 0,
        canonical_fusion_records: row.semantic_authority?.fusion_specs?.length ?? 0,
        observed_provider_paths: row.provider?.gpu_or_wgpu_paths?.length ?? 0,
        observed_fusion_paths: row.provider?.fusion_paths?.length ?? 0,
      },
      test_strength: row.tests?.strength ?? "none",
      documentation_strength: row.documentation?.strength ?? "none",
      unresolved_fields: [...(row.unresolved ?? [])].sort(compareCodePoint),
      typed_paths: identityTypedBaselineEvidence(row, sourceFilesByPath),
      baseline_evidence_candidates: baselineEvidenceCandidates(row, sourceFilesByPath),
    },
  };
}

function identityCandidatePaths(row, topologyRow, sourceFilesByPath) {
  const evidence = identityTypedBaselineEvidence(row, sourceFilesByPath);
  const select = (...kinds) => paths(evidence
    .filter((entry) => kinds.includes(entry.kind))
    .map((entry) => entry.path));
  return {
    observed: {
      catalog: select(
        "catalog-owner", "catalog-alias-provenance", "catalog-constant-provenance",
        "catalog-provenance",
      ),
      runtime: select(
        "runtime-owner", "implementation-provenance", "runtime-registration",
        "runtime-constant-registration", "native-link-runtime-input",
      ),
      documentation: select(
        "catalog-documentation", "legacy-sidecar", "runtime-documentation-shadow",
        "documentation-source",
      ),
      tests: select("test-source"),
      dependencies: select(
        "native-link-catalog-contract", "legacy-resolver", "catalog-resolver",
      ),
      provider: select("provider", "fusion"),
      generated: select("generated-registry"),
    },
    topology_target_proposals: topologyTargetProposals(topologyRow, row),
  };
}

function mergeCandidatePaths(rows) {
  const observed = {};
  for (const field of ["catalog", "runtime", "documentation", "tests", "dependencies", "provider", "generated"]) {
    observed[field] = paths(rows.flatMap((row) => row.observed[field]));
  }
  return {
    observed,
    topology_target_proposals: {
      proposed_catalog_packages: paths(rows.map((row) => row.topology_target_proposals.proposed_catalog_package)),
      proposed_callable_owners: paths(rows.map((row) => row.topology_target_proposals.proposed_callable_owner)),
      basis: "reviewed-topology-target-packages-only",
    },
  };
}

function topologyTargetProposals(row, sourceRow) {
  const leaf = rustLeaf(row.identity);
  if (row.disposition.kind === "alias") {
    return {
      proposed_catalog_package: null,
      proposed_callable_owner: null,
      proposed_callable_bindings: [],
      proposed_canonical_identity: row.disposition.canonical,
      basis: "reviewed-alias-target-no-copied-authority-proposal",
    };
  }
  if (row.disposition.kind === "internal") {
    return {
      proposed_catalog_package: null,
      proposed_callable_owner: null,
      proposed_callable_bindings: [],
      proposed_canonical_identity: null,
      basis: "reviewed-internal-identity-observed-paths-only",
    };
  }
  const observedOwners = [...new Set((sourceRow.semantic_authority?.implementation_provenance ?? [])
    .map((entry) => entry.source_file))].sort(compareCodePoint);
  const runtimeOwner = canonicalCallableOwnerPath(row, observedOwners);
  return {
    proposed_catalog_package: `${CATALOG_ROOT}/${row.domain}/${row.family}/${leaf}/mod.rs`,
    proposed_callable_owner: runtimeOwner,
    proposed_callable_bindings: (sourceRow.semantic_authority?.implementation_provenance ?? []).map((entry) => ({
      path: runtimeOwner,
      function: entry.function,
      variant: entry.binding_variant ?? "default",
    })),
    proposed_canonical_identity: null,
    basis: "reviewed-topology-target-package-with-observed-function-candidates",
  };
}

function baselineEvidenceCandidates(row, sourceFilesByPath) {
  return (row.source_metrics?.files ?? []).map((file) => {
    const source = sourceFilesByPath.get(file.path);
    if (!source) throw new Error(`${row.identity}: source metrics path ${file.path} is absent from the exact source snapshot`);
    return { path: source.path, content_digest: source.content_digest };
  }).sort((left, right) => compareCodePoint(left.path, right.path));
}

function paths(values) {
  return [...new Set(values.filter((value) => typeof value === "string" && value.length > 0))]
    .sort(compareCodePoint);
}

function unresolvedDecisions(fields) {
  return Object.fromEntries(fields.map((field) => [field, { status: "unresolved" }]));
}

function unreviewed() {
  return { status: "unreviewed", evidence: [] };
}

function canonicalEntries(map) {
  return [...map.entries()].sort(([left], [right]) => compareCodePoint(left, right));
}

function clone(value) {
  return structuredClone(value);
}
