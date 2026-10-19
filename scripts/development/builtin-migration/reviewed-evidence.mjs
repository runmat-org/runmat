import { compareCodePoint } from "./constants.mjs";
import { array, exact, nonempty } from "./schema.mjs";

export function parseUniqueReviewedEvidence(value, label) {
  exact(value, ["status", "evidence"], label);
  if (value.status !== "reviewed") throw new Error(`${label} status must be reviewed`);
  const evidence = array(value.evidence, `${label} evidence`)
    .map((entry) => nonempty(entry, `${label} evidence`));
  if (new Set(evidence).size !== evidence.length) {
    throw new Error(`${label} evidence must be unique`);
  }
  return value;
}

export function parseCanonicalReviewedEvidence(value, label) {
  parseUniqueReviewedEvidence(value, label);
  const evidence = value.evidence.map((entry) => nonempty(entry, `${label} evidence`));
  if (JSON.stringify(evidence) !== JSON.stringify([...evidence].sort(compareCodePoint))) {
    throw new Error(`${label} evidence must be canonically ordered`);
  }
  return value;
}
