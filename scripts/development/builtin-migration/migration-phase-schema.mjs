import { compareCodePoint } from "./constants.mjs";
import { parseEffectivePathScope } from "./path-scope.mjs";
import { evidenceDigest } from "./evidence.mjs";
import {
  array, digest, exact, repositoryPath, sourceRevision, stableId,
} from "./schema.mjs";

export function parseMigrationPhases(value) {
  exact(value, [
    "lease_base_revision", "authored_revision", "integrated_revision",
    "authored_changed_paths", "integration_changed_paths",
    "reviewed_authored_write_set", "reviewed_integration_outputs",
    "authored_write_set_digest", "integration_outputs_digest",
  ], "migration phases");
  const parsed = {
    lease_base_revision: sourceRevision(value.lease_base_revision, "phase lease base revision"),
    authored_revision: sourceRevision(value.authored_revision, "phase authored revision"),
    integrated_revision: sourceRevision(value.integrated_revision, "phase integrated revision"),
    authored_changed_paths: canonicalPaths(value.authored_changed_paths, "authored changed paths"),
    integration_changed_paths: canonicalPaths(value.integration_changed_paths, "integration changed paths"),
    reviewed_authored_write_set: parseScopes(value.reviewed_authored_write_set),
    reviewed_integration_outputs: parseOutputs(value.reviewed_integration_outputs),
    authored_write_set_digest: digest(value.authored_write_set_digest, "authored write-set digest"),
    integration_outputs_digest: digest(value.integration_outputs_digest, "integration outputs digest"),
  };
  if (evidenceDigest(parsed.reviewed_authored_write_set) !== parsed.authored_write_set_digest) {
    throw new Error("authored write-set digest mismatch");
  }
  if (evidenceDigest(parsed.reviewed_integration_outputs) !== parsed.integration_outputs_digest) {
    throw new Error("integration outputs digest mismatch");
  }
  return parsed;
}

function canonicalPaths(value, label) {
  const paths = array(value, label, { empty: true }).map((entry) => repositoryPath(entry, label));
  const expected = [...new Set(paths)].sort(compareCodePoint);
  if (JSON.stringify(paths) !== JSON.stringify(expected)) {
    throw new Error(`${label} must be unique and canonically ordered`);
  }
  return paths;
}

function parseScopes(value) {
  return array(value, "reviewed authored write set", { empty: true })
    .map((entry) => parseEffectivePathScope(entry, "reviewed authored scope"));
}

function parseOutputs(value) {
  return array(value, "reviewed integration outputs", { empty: true }).map((entry) => {
    exact(entry, ["product_id", "path", "producer"], "reviewed integration output");
    if (entry.producer !== "integration") {
      throw new Error("reviewed integration output producer must be integration");
    }
    return {
      product_id: stableId(entry.product_id, "reviewed integration product id"),
      path: repositoryPath(entry.path, "reviewed integration output path"),
      producer: entry.producer,
    };
  });
}
