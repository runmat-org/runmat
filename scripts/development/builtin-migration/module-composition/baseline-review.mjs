import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { array, digest, enumValue, exact, kind, uniqueStrings } from "../schema.mjs";
import { assertValidatedModuleCompositionBaselineCandidate } from "./baseline-candidate.mjs";

const KIND = "runmat-builtin-module-composition-baseline-review";
const ROLES = Object.freeze(["group", "identity", "support"]);
const VALIDATED = new WeakSet();

export function buildModuleCompositionBaselineReviewTemplate(candidateValue) {
  const candidate = assertValidatedModuleCompositionBaselineCandidate(candidateValue);
  return {
    schema_version: 1,
    kind: KIND,
    authority: "reviewer-authored-development-input",
    candidate_digest: candidate.digest,
    source_revision: candidate.source.revision,
    registry_digest: candidate.registry_digest,
    roles: candidate.products.flatMap((product) => product.children.map((child) => ({
      product_id: product.product_id, module: child.module, role: null,
    }))),
    review: { status: "unreviewed", evidence: [] },
  };
}

export function sealModuleCompositionBaselineReview(value, candidateValue) {
  const candidate = assertValidatedModuleCompositionBaselineCandidate(candidateValue);
  if (Object.hasOwn(value, "digest")) throw new Error("module composition baseline review must not supply its own digest");
  const sealed = { ...structuredClone(value), digest: evidenceDigest(value) };
  return parseModuleCompositionBaselineReview(sealed, candidate);
}

export function parseModuleCompositionBaselineReview(value, candidateValue) {
  const candidate = assertValidatedModuleCompositionBaselineCandidate(candidateValue);
  kind(value, 1, KIND, "module composition baseline review");
  exact(value, [
    "schema_version", "kind", "authority", "candidate_digest", "source_revision",
    "registry_digest", "roles", "review", "digest",
  ], "module composition baseline review");
  if (value.authority !== "reviewer-authored-development-input") throw new Error("module composition baseline review has invalid authority");
  if (value.candidate_digest !== candidate.digest
    || value.source_revision !== candidate.source.revision
    || value.registry_digest !== candidate.registry_digest) {
    throw new Error("module composition baseline review does not bind the exact candidate");
  }
  const roles = array(value.roles, "module composition baseline roles", { empty: true }).map((entry) => {
    exact(entry, ["product_id", "module", "role"], "module composition baseline role");
    return { ...entry, role: enumValue(entry.role, ROLES, `${entry.product_id}/${entry.module} role`) };
  });
  const keys = roles.map((entry) => `${entry.product_id}\0${entry.module}`);
  if (new Set(keys).size !== keys.length || JSON.stringify(keys) !== JSON.stringify([...keys].sort(compareCodePoint))) {
    throw new Error("module composition baseline roles must be unique and canonically ordered");
  }
  const expected = candidate.products.flatMap((product) => product.children.map((child) =>
    `${product.product_id}\0${child.module}`));
  if (JSON.stringify(keys) !== JSON.stringify(expected)) throw new Error("module composition baseline roles must exactly cover every observed child");
  exact(value.review, ["status", "evidence"], "module composition baseline review evidence");
  if (value.review.status !== "reviewed") throw new Error("module composition baseline review must be reviewed");
  uniqueStrings(value.review.evidence, "module composition baseline review evidence");
  digest(value.digest, "module composition baseline review digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("module composition baseline review digest mismatch");
  const result = deepImmutable({ ...structuredClone(value), roles });
  VALIDATED.add(result);
  return result;
}

export function assertValidatedModuleCompositionBaselineReview(value) {
  if (!VALIDATED.has(value)) throw new Error("operation requires the exact validated module composition baseline review");
  return value;
}
