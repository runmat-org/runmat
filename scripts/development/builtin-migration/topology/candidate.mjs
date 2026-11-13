import { evidenceDigest } from "../evidence.mjs";
import { array, digest, exact, kind, object } from "../schema.mjs";
import { parseReviewBaseline, TOPOLOGY_PROGRAM } from "./schema.mjs";

export const TOPOLOGY_CANDIDATE_KIND = "runmat-builtin-migration-topology-candidate";
export const TOPOLOGY_CANDIDATE_VERSION = 1;

export function parseTopologyCandidate(value, expected) {
  if (!expected) throw new Error("topology candidate parsing requires deterministic recomposition");
  kind(value, TOPOLOGY_CANDIDATE_VERSION, TOPOLOGY_CANDIDATE_KIND, "topology candidate");
  exact(value, ["schema_version", "kind", "authority", "program", "baseline", "inputs", "bundles", "identities", "summary", "review", "digest"], "topology candidate");
  if (value.authority !== "composed-unreviewed-candidate" || value.program !== TOPOLOGY_PROGRAM) {
    throw new Error("topology candidate has invalid authority or program");
  }
  parseReviewBaseline(value.baseline);
  exact(value.inputs, ["c01_c03_review", "c04_c05_review", "c06_c07_review", "reconciliation", "stability_corrections"], "topology candidate inputs");
  for (const [name, inputDigest] of Object.entries(value.inputs)) digest(inputDigest, `topology candidate ${name} digest`);
  object(value.bundles, "topology candidate bundles");
  object(value.identities, "topology candidate identities");
  object(value.summary, "topology candidate summary");
  exact(value.review, ["status", "evidence"], "topology candidate review");
  if (value.review.status !== "unreviewed" || array(value.review.evidence, "topology candidate review evidence", { empty: true }).length !== 0) {
    throw new Error("topology candidate cannot claim review");
  }
  digest(value.digest, "topology candidate digest");
  const { digest: _ignored, ...payload } = value;
  if (evidenceDigest(payload) !== value.digest) throw new Error("topology candidate digest mismatch");
  if (evidenceDigest(value) !== evidenceDigest(expected)) {
    throw new Error("topology candidate differs from deterministic composition");
  }
  return value;
}
