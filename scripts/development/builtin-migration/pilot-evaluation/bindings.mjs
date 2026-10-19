import { canonicalAuthorityPath } from "../authority-loading/index.mjs";
import { digest, exact } from "../schema.mjs";

export function parseArtifactBinding(value, label) {
  exact(value, ["path", "semantic_digest", "content_digest"], label);
  return {
    path: canonicalAuthorityPath(value.path, `${label} path`),
    semantic_digest: digest(value.semantic_digest, `${label} semantic digest`),
    content_digest: digest(value.content_digest, `${label} content digest`),
  };
}

export function artifactBinding(observation) {
  return {
    path: observation.path,
    semantic_digest: observation.semanticDigest,
    content_digest: observation.contentDigest,
  };
}

export function loadedAuthorityBinding(authority) {
  return {
    path: authority.reference.path,
    semantic_digest: authority.reference.digest,
    content_digest: authority.artifact.contentDigest,
  };
}
