import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { digest, exact, kind, sourceRevision, uniqueStrings } from "../schema.mjs";
import {
  assertValidatedModuleCompositionBaselineCandidate,
  deriveModuleCompositionBaselineCandidate,
} from "./baseline-candidate.mjs";
import {
  assertValidatedModuleCompositionBaselineReview,
  parseModuleCompositionBaselineReview,
} from "./baseline-review.mjs";
import { validateFixedModuleCompositionProjection } from "./authority.mjs";
import { parseModuleCompositionProjection } from "./schema.mjs";
import { gitTreeOid, signerFingerprint } from "./baseline-schema.mjs";

const KIND = "runmat-builtin-module-composition-reviewed-baseline";
const VALIDATED = new WeakSet();
const INTEGRITY_VALIDATED = new WeakSet();

export function freezeReviewedModuleCompositionBaseline(candidateValue, reviewValue) {
  const candidate = assertValidatedModuleCompositionBaselineCandidate(candidateValue);
  const review = assertValidatedModuleCompositionBaselineReview(reviewValue);
  if (review.candidate_digest !== candidate.digest) throw new Error("module composition baseline review belongs to another candidate");
  const roles = new Map(review.roles.map((entry) => [`${entry.product_id}\0${entry.module}`, entry.role]));
  const projection = validateFixedModuleCompositionProjection(parseModuleCompositionProjection({
    schema_version: 4,
    kind: "runmat-builtin-module-composition-projection",
    products: candidate.products.map((product) => ({
      product_id: product.product_id,
      crate_role: product.crate_role,
      path: product.path,
      module_path: product.module_path,
      aggregations: product.aggregations,
      aggregation_exports: product.aggregation_exports,
      children: product.children.map(({ source_evidence: _ignored, ...child }) => ({
        ...child, role: roles.get(`${product.product_id}\0${child.module}`),
      })),
    })),
  }));
  const payload = {
    schema_version: 1,
    kind: KIND,
    authority: "reviewed-bootstrap-baseline",
    bindings: {
      candidate_digest: candidate.digest,
      review_digest: review.digest,
      registry_digest: candidate.registry_digest,
      source_revision: candidate.source.revision,
      source_tree_oid: candidate.source.tree_oid,
      source_files_digest: candidate.source.files_digest,
      signer_fingerprint: candidate.source.signer_fingerprint,
    },
    projection: structuredClone(projection),
    review: structuredClone(review.review),
  };
  return validated({ ...payload, digest: evidenceDigest(payload) });
}

export function validateReviewedModuleCompositionBaseline(
  value, trustedDigest, repository, trustedSignerFingerprint, options = {},
) {
  const trusted = parseTrustedReviewedModuleCompositionBaseline(value, trustedDigest);
  const candidate = deriveModuleCompositionBaselineCandidate(
    repository, trustedSignerFingerprint, options,
  );
  const roles = trusted.projection.products.flatMap((product) => product.children.map((child) => ({
    product_id: product.product_id, module: child.module, role: child.role,
  })));
  const reviewPayload = {
    schema_version: 1,
    kind: "runmat-builtin-module-composition-baseline-review",
    authority: "reviewer-authored-development-input",
    candidate_digest: candidate.digest,
    source_revision: candidate.source.revision,
    registry_digest: candidate.registry_digest,
    roles,
    review: structuredClone(trusted.review),
  };
  const review = parseModuleCompositionBaselineReview(
    { ...reviewPayload, digest: evidenceDigest(reviewPayload) }, candidate,
  );
  const expected = freezeReviewedModuleCompositionBaseline(candidate, review);
  if (JSON.stringify(trusted) !== JSON.stringify(expected)) {
    throw new Error("reviewed module composition baseline differs from deterministic reconstruction");
  }
  return expected;
}

export function parseTrustedReviewedModuleCompositionBaseline(value, trustedDigest) {
  parseIntegrity(value, trustedDigest);
  const projection = validateFixedModuleCompositionProjection(
    parseModuleCompositionProjection(value.projection),
  );
  const result = deepImmutable({ ...structuredClone(value), projection: structuredClone(projection) });
  INTEGRITY_VALIDATED.add(result);
  return result;
}

export function assertIntegrityValidatedReviewedModuleCompositionBaseline(value) {
  if (!INTEGRITY_VALIDATED.has(value)) {
    throw new Error("operation requires the integrity-validated reviewed module composition baseline");
  }
  return value;
}

export function assertValidatedReviewedModuleCompositionBaseline(value) {
  if (!VALIDATED.has(value)) throw new Error("operation requires the exact validated reviewed module composition baseline");
  return value;
}

export function moduleCompositionBaselineProvenance(value) {
  const baseline = assertValidatedReviewedModuleCompositionBaseline(value);
  return {
    reviewed_baseline_digest: baseline.digest,
    candidate_digest: baseline.bindings.candidate_digest,
    review_digest: baseline.bindings.review_digest,
    source_revision: baseline.bindings.source_revision,
    source_tree_oid: baseline.bindings.source_tree_oid,
    signer_fingerprint: baseline.bindings.signer_fingerprint,
  };
}

function parseIntegrity(value, trustedDigest) {
  kind(value, 1, KIND, "reviewed module composition baseline");
  exact(value, ["schema_version", "kind", "authority", "bindings", "projection", "review", "digest"], "reviewed module composition baseline");
  if (value.authority !== "reviewed-bootstrap-baseline") throw new Error("reviewed module composition baseline has invalid authority");
  exact(value.bindings, [
    "candidate_digest", "review_digest", "registry_digest", "source_revision",
    "source_tree_oid", "source_files_digest", "signer_fingerprint",
  ], "reviewed module composition baseline bindings");
  for (const field of ["candidate_digest", "review_digest", "registry_digest", "source_files_digest"]) {
    digest(value.bindings[field], `reviewed module composition baseline ${field}`);
  }
  sourceRevision(value.bindings.source_revision, "reviewed module composition baseline source revision");
  gitTreeOid(value.bindings.source_tree_oid, "reviewed module composition baseline source tree");
  signerFingerprint(value.bindings.signer_fingerprint, "reviewed module composition baseline signer fingerprint");
  exact(value.review, ["status", "evidence"], "reviewed module composition baseline evidence");
  if (value.review.status !== "reviewed") {
    throw new Error("reviewed module composition baseline evidence must be reviewed");
  }
  uniqueStrings(value.review.evidence, "reviewed module composition baseline evidence");
  digest(value.digest, "reviewed module composition baseline digest");
  if (digest(trustedDigest, "trusted reviewed module composition baseline digest") !== value.digest) {
    throw new Error("reviewed module composition baseline differs from its trusted digest");
  }
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("reviewed module composition baseline digest mismatch");
}

function validated(value) {
  const result = deepImmutable(value);
  VALIDATED.add(result);
  return result;
}
