import { CATALOG_ROOT, RUNTIME_ROOT, compareCodePoint, rustLeaf } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { observedIdentityForms } from "../identity-authority.mjs";

const AUTHORITY = "machine-derived-non-authoritative-proposal-only";

export function buildAuthorityProposals(inventoryByIdentity, topology, migrationFindings) {
  const identityRows = [...topology.identities.entries()]
    .sort(([left], [right]) => compareCodePoint(left, right))
    .map(([identity, topologyRow]) => {
      const sourceRow = inventoryByIdentity.get(identity);
      if (!sourceRow) throw new Error(`${identity}: authority proposal has no exact inventory row`);
      const proposal = identityAuthorityProposal(sourceRow, topologyRow);
      return { identity, proposal, proposal_digest: evidenceDigest(proposal) };
    });
  const findingRows = migrationFindings.map((finding) => {
    const findingDigest = evidenceDigest(finding);
    const proposal = migrationFindingRoutingProposal(finding, topology, inventoryByIdentity);
    return { finding_digest: findingDigest, proposal, proposal_digest: evidenceDigest(proposal) };
  }).sort((left, right) => compareCodePoint(left.finding_digest, right.finding_digest));
  const payload = {
    schema_version: 1,
    kind: "runmat-builtin-migration-authority-proposals",
    authority: AUTHORITY,
    identity_rows: identityRows,
    migration_finding_rows: findingRows,
  };
  return { ...payload, digest: evidenceDigest(payload) };
}

export function parseAuthorityProposals(value, inventoryByIdentity, topology, migrationFindings) {
  if (value?.authority !== AUTHORITY) {
    throw new Error("authority proposals cannot claim or accept reviewed authority");
  }
  const expected = buildAuthorityProposals(inventoryByIdentity, topology, migrationFindings);
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("authority proposals differ from deterministic reconstruction");
  }
  return expected;
}

export function identityAuthorityProposal(sourceRow, topologyRow) {
  const forms = observedIdentityForms(sourceRow);
  const provenance = structuredClone(sourceRow.semantic_authority?.implementation_provenance ?? []);
  const constantRegistrations = structuredClone(sourceRow.semantic_authority?.runtime_constants ?? []);
  const callableOwnerPaths = spellings(provenance.map((entry) => entry.source_file));
  const constantOwnerPaths = spellings(constantRegistrations.map((entry) => entry.source_file));
  const callableOwner = callableOwnerProposal(topologyRow, forms, callableOwnerPaths);
  const constantOwner = formOwnerProposal(topologyRow, forms.constant_spellings, constantOwnerPaths);
  return {
    authority: AUTHORITY,
    public_identity: publicIdentityProposal(sourceRow, topologyRow),
    forms,
    implementation: {
      callable: {
        observed_owner_paths: callableOwnerPaths,
        observed_bindings: provenance,
        proposed_owner_path: callableOwner,
        basis: ownerBasis(topologyRow, forms.callable_spellings, callableOwner, "callable"),
      },
      constant: {
        observed_owner_paths: constantOwnerPaths,
        observed_bindings: constantRegistrations,
        proposed_owner_path: constantOwner,
        basis: ownerBasis(topologyRow, forms.constant_spellings, constantOwner, "constant"),
      },
    },
    catalog: {
      proposed_package: topologyRow.disposition.kind === "canonical"
        ? `${CATALOG_ROOT}/${topologyRow.domain}/${topologyRow.family}/${rustLeaf(topologyRow.identity)}/mod.rs`
        : null,
      observed_entry_count: sourceRow.semantic_authority?.catalog_entries?.length ?? 0,
      observed_constant_count: sourceRow.semantic_authority?.constants?.length ?? 0,
    },
  };
}

function callableOwnerProposal(topologyRow, forms, observedOwnerPaths) {
  if (forms.callable_spellings.length === 0 || topologyRow.disposition.kind === "alias") return null;
  if (topologyRow.disposition.kind === "internal") {
    return observedOwnerPaths.length === 1 ? observedOwnerPaths[0] : null;
  }
  return typeof topologyRow.domain === "string" && typeof topologyRow.family === "string"
    ? `${RUNTIME_ROOT}/${topologyRow.domain}/${topologyRow.family}/${rustLeaf(topologyRow.identity)}/mod.rs`
    : null;
}

function formOwnerProposal(topologyRow, formSpellings, observedOwnerPaths) {
  if (formSpellings.length === 0 || topologyRow.disposition.kind === "alias") return null;
  return observedOwnerPaths.length === 1 ? observedOwnerPaths[0] : null;
}

function ownerBasis(topologyRow, formSpellings, proposedOwner, form) {
  if (formSpellings.length === 0) return `no-${form}-form-observed`;
  if (topologyRow.disposition.kind === "alias") return "alias-has-no-independent-implementation-owner-proposal";
  if (form === "constant" || topologyRow.disposition.kind === "internal") {
    return proposedOwner === null
      ? `${form}-owner-observations-are-not-unique`
      : `unique-observed-owner-for-${form}-form`;
  }
  return proposedOwner === null
    ? "reviewed-topology-target-is-unresolved"
    : "reviewed-topology-target-for-observed-callable-implementation";
}

export function migrationFindingRoutingProposal(finding, topology, inventoryByIdentity) {
  const affected = structuredClone(finding.affected);
  if (affected.kind === "owner") {
    const identities = affected.owner.affected_identities.map((entry) => entry.name.toLowerCase());
    const candidateBundleIds = bundleIdsForIdentities(identities, topology, inventoryByIdentity);
    return {
      authority: AUTHORITY,
      affected,
      membership: {
        kind: "owner",
        identities,
        candidate_bundle_ids: candidateBundleIds,
        status: "machine-derived-candidate",
      },
      basis: "compiled-typed-owner-membership-and-reviewed-topology",
    };
  }
  const identity = affected.identity.name.toLowerCase();
  const candidateBundleIds = bundleIdsForIdentities([identity], topology, inventoryByIdentity);
  return {
    authority: AUTHORITY,
    affected,
    membership: {
      kind: "identity",
      identity,
      candidate_bundle_ids: candidateBundleIds,
      status: "machine-derived-candidate",
    },
    basis: "typed-affected-identity-reviewed-topology-membership",
  };
}

function bundleIdsForIdentities(identities, topology, inventoryByIdentity) {
  const bundleIds = identities.map((identity) => {
    if (!inventoryByIdentity.has(identity)) {
      throw new Error(`${identity}: typed migration finding identity is absent from the exact inventory`);
    }
    const topologyRow = topology.identities.get(identity);
    if (!topologyRow) throw new Error(`${identity}: typed migration finding identity is absent from reviewed topology`);
    return topologyRow.bundle_id;
  });
  return spellings(bundleIds);
}

function publicIdentityProposal(sourceRow, topologyRow) {
  if (topologyRow.disposition.kind === "internal") {
    return {
      kind: "internal",
      reason: topologyRow.disposition.reason,
      evidence: [`reviewed-topology:${topologyRow.identity}`],
    };
  }
  const spelling = publicSpellingProposal(sourceRow);
  if (topologyRow.disposition.kind === "alias") {
    return {
      kind: "alias",
      alias_spelling: spelling,
      canonical_identity: topologyRow.disposition.canonical,
    };
  }
  return {
    kind: "primary",
    primary_spelling: spelling,
  };
}

function publicSpellingProposal(row) {
  if (!row) throw new Error("public spelling proposal requires an exact inventory row");
  const candidates = spellings(row.spellings ?? []);
  return {
    identity: row.identity,
    spelling: candidates.length === 1 ? candidates[0] : null,
    candidates,
    status: candidates.length === 1 ? "machine-derived-candidate" : "unresolved",
  };
}

function spellings(values) {
  return [...new Set(values.filter((value) => typeof value === "string" && value.length > 0))]
    .sort(compareCodePoint);
}
