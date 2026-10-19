import { evidenceDigest } from "../evidence.mjs";
import { deepImmutable } from "../immutable.mjs";
import { digest, exact, kind } from "../schema.mjs";
import { observeModuleCompositionBaseline } from "./baseline-observation.mjs";

const KIND = "runmat-builtin-module-composition-baseline-candidate";
const PRODUCER = "runmat-module-composition-baseline-observer-v1";
const VALIDATED = new WeakSet();

export function deriveModuleCompositionBaselineCandidate(repository, trustedSignerFingerprint, options = {}) {
  const observation = observeModuleCompositionBaseline(repository, trustedSignerFingerprint, options);
  const payload = {
    schema_version: 1,
    kind: KIND,
    authority: "machine-derived-unreviewed-candidate-only",
    producer: PRODUCER,
    ...observation,
  };
  return validated({ ...payload, digest: evidenceDigest(payload) });
}

export function parseModuleCompositionBaselineCandidate(
  value, repository, trustedSignerFingerprint, options = {},
) {
  kind(value, 1, KIND, "module composition baseline candidate");
  exact(value, [
    "schema_version", "kind", "authority", "producer", "registry_digest",
    "source", "products", "digest",
  ], "module composition baseline candidate");
  if (value.authority !== "machine-derived-unreviewed-candidate-only" || value.producer !== PRODUCER) {
    throw new Error("module composition baseline candidate has invalid authority or producer");
  }
  digest(value.digest, "module composition baseline candidate digest");
  const expected = deriveModuleCompositionBaselineCandidate(
    repository, trustedSignerFingerprint, options,
  );
  if (JSON.stringify(value) !== JSON.stringify(expected)) {
    throw new Error("module composition baseline candidate differs from deterministic source observation");
  }
  return expected;
}

export function assertValidatedModuleCompositionBaselineCandidate(value) {
  if (!VALIDATED.has(value)) throw new Error("operation requires the exact validated module composition baseline candidate");
  return value;
}

function validated(value) {
  const result = deepImmutable(value);
  VALIDATED.add(result);
  return result;
}
