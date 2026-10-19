import {
  openAuthorityLoadSession, openAuthorityRoot,
} from "../authority-loading/index.mjs";
import {
  loadInitialQueueReview, recordInitialQueue,
} from "../queue-initialization/index.mjs";
import { canonicalCliChildPath } from "../queue-authority/reference.mjs";

export function runInitialQueueCommand({ options, control }) {
  const root = openAuthorityRoot(options.authorityRoot);
  const session = openAuthorityLoadSession(root);
  const review = loadInitialQueueReview({
    session,
    reference: {
      path: canonicalCliChildPath(
        root.path, options.initialQueueReview, "--initial-queue-review path",
      ),
      digest: options.initialQueueReviewDigest,
    },
    control,
  });
  const queue = recordInitialQueue({ session, control, review });
  return {
    schema_version: 1,
    kind: "runmat-builtin-migration-initialize-queue-command-result",
    state: observation(queue.observations.state),
    checkpoint: observation(queue.observations.checkpoint),
  };
}

function observation(value) {
  return {
    path: value.path,
    digest: value.semanticDigest,
    content_digest: value.contentDigest,
  };
}
