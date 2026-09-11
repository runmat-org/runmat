import { compareCodePoint } from "../constants.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { array, exact, identity, kind, object, stableId } from "../schema.mjs";
import { buildAuthorityComponentGraph, validateComponentIndex } from "./components.mjs";

export function parseAuthorityComponentGraph(value, inventory) {
  kind(value, 1, "runmat-builtin-topology-candidate", "authority component graph");
  exact(value, ["schema_version", "kind", "authority", "baseline_inventory_digest", "summary", "candidates"], "authority component graph");
  if (value.authority !== "discovery-only-not-control-input") {
    throw new Error("authority component graph has invalid authority");
  }
  object(value.summary, "authority component graph summary");
  array(value.candidates, "authority component graph candidates");

  const expected = buildAuthorityComponentGraph(inventory);
  if (evidenceDigest(value) !== evidenceDigest(expected)) {
    throw new Error("authority component graph differs from the deterministic reviewed-baseline projection");
  }

  const index = new Map();
  const sharedSources = new Map();
  for (const [position, candidate] of value.candidates.entries()) {
    object(candidate, `authority component ${position}`);
    const component = stableId(candidate.candidate_id, `authority component ${position} id`);
    const identities = candidate.identities.map((entry) => identity(entry, `${component} identity`));
    if (index.has(component)) throw new Error(`duplicate authority component ${component}`);
    index.set(component, identities);
    sharedSources.set(component, [...candidate.shared_sources]);
  }
  const canonicalIndex = new Map([...index].sort(([left], [right]) => compareCodePoint(left, right)));
  validateComponentIndex(canonicalIndex);
  return {
    value,
    digest: evidenceDigest(value),
    index: canonicalIndex,
    sharedSources,
  };
}
