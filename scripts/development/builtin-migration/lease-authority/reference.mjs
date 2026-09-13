import { digest, exact } from "../schema.mjs";
import { canonicalAuthorityPath } from "../authority-loading/index.mjs";

export function parseLeaseAuthorityReference(value, label = "lease authority reference") {
  exact(value, ["path", "digest"], label);
  return Object.freeze({
    path: canonicalAuthorityPath(value.path, `${label} path`),
    digest: digest(value.digest, `${label} digest`),
  });
}
