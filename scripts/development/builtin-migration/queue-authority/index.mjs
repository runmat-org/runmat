import path from "node:path";

import {
  assertAuthorityLoadSession, openAuthorityLoadSession, openAuthorityRoot,
  revalidateObservedArtifacts,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import {
  loadQueueCheckpoint, openQueueCheckpointLoader,
} from "./checkpoint-loader.mjs";
import {
  canonicalCliChildPath, openQueueReferenceIndex,
} from "./reference.mjs";
import { openQueueSealLoader } from "./seal-loader.mjs";
import { loadQueueState, openQueueStateLoader } from "./state-loader.mjs";

const SNAPSHOTS = new WeakMap();

export function loadQueueAuthority({
  session, statePath, checkpointPath, trustedCheckpointDigest, control,
}) {
  const authoritySession = assertAuthorityLoadSession(session);
  const validatedControl = assertValidatedControl(control);
  const references = openQueueReferenceIndex(authoritySession);
  const seals = openQueueSealLoader({
    session: authoritySession, references, control: validatedControl,
  });
  const states = openQueueStateLoader({
    session: authoritySession, references, seals, control: validatedControl,
  });
  const checkpoints = openQueueCheckpointLoader({
    session: authoritySession, references, control: validatedControl,
  });
  const loadedState = loadQueueState(states, { path: statePath, digest: null });
  const loadedCheckpoint = loadQueueCheckpoint(checkpoints, {
    path: checkpointPath, digest: trustedCheckpointDigest,
  }, loadedState.value);
  revalidateObservedArtifacts(authoritySession);
  const snapshot = Object.freeze({
    state: loadedState.value,
    checkpoint: loadedCheckpoint.value,
    observations: Object.freeze({
      state: publicObservation(loadedState),
      checkpoint: publicObservation(loadedCheckpoint),
    }),
  });
  SNAPSHOTS.set(snapshot, {
    session: authoritySession,
    control: validatedControl,
    statePath: loadedState.path,
    stateDigest: loadedState.semanticDigest,
    checkpointPath: loadedCheckpoint.path,
    checkpointDigest: loadedCheckpoint.semanticDigest,
  });
  return snapshot;
}

export function loadQueueAuthorityFromCliPaths({
  statePath, checkpointPath, trustedCheckpointDigest, control,
}) {
  const resolvedStatePath = path.resolve(statePath);
  const resolvedCheckpointPath = path.resolve(checkpointPath);
  const lexicalRoot = path.dirname(resolvedStatePath);
  const root = openAuthorityRoot(lexicalRoot);
  const session = openAuthorityLoadSession(root);
  return loadQueueAuthority({
    session,
    statePath: canonicalCliChildPath(
      lexicalRoot, resolvedStatePath, "queue state path",
    ),
    checkpointPath: canonicalCliChildPath(
      lexicalRoot, resolvedCheckpointPath, "queue checkpoint path",
    ),
    trustedCheckpointDigest,
    control,
  });
}

export function assertLoadedQueueAuthority(value, {
  session = null, control = null, statePath = null, stateDigest = null,
  checkpointPath = null, checkpointDigest = null,
} = {}) {
  const record = SNAPSHOTS.get(value);
  if (!record) throw new Error("operation requires an exact loaded queue authority snapshot");
  if (session !== null && record.session !== assertAuthorityLoadSession(session)) {
    throw new Error("queue authority snapshot belongs to another load session");
  }
  if (control !== null && record.control !== assertValidatedControl(control)) {
    throw new Error("queue authority snapshot belongs to another control manifest");
  }
  for (const [expected, observed, label] of [
    [statePath, record.statePath, "state path"],
    [stateDigest, record.stateDigest, "state digest"],
    [checkpointPath, record.checkpointPath, "checkpoint path"],
    [checkpointDigest, record.checkpointDigest, "checkpoint digest"],
  ]) {
    if (expected !== null && expected !== observed) {
      throw new Error(`queue authority snapshot ${label} mismatch`);
    }
  }
  return value;
}

export function revalidateLoadedQueueAuthority(value) {
  const snapshot = assertLoadedQueueAuthority(value);
  revalidateObservedArtifacts(SNAPSHOTS.get(snapshot).session);
  return snapshot;
}

function publicObservation(value) {
  return Object.freeze({
    path: value.path,
    semanticDigest: value.semanticDigest,
    contentDigest: value.contentDigest,
  });
}
