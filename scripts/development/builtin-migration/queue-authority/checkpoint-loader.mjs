import {
  assertAuthorityLoadSession, assertSessionArtifact, loadJsonArtifact,
  loadedJsonValue, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { validateQueueCheckpoint } from "../queue-checkpoint.mjs";
import { digest } from "../schema.mjs";
import { bindQueueReference, assertQueueReferenceIndex } from "./reference.mjs";

const LOADERS = new WeakMap();

export function openQueueCheckpointLoader({ session, references, control }) {
  const authoritySession = assertAuthorityLoadSession(session);
  assertQueueReferenceIndex(references, authoritySession);
  const loader = Object.freeze({});
  LOADERS.set(loader, {
    session: authoritySession,
    references,
    control: assertValidatedControl(control),
    values: new Map(),
  });
  return loader;
}

export function loadQueueCheckpoint(loaderValue, reference, queueState) {
  const loader = loaderRecord(loaderValue);
  const current = loadCheckpointValue(loader, reference);
  return withAuthorityTraversal(loader.session, current.binding, () => {
    assertExpectedDigest(current);
    const value = validateQueueCheckpoint(
      current.raw,
      current.binding.semanticDigest,
      queueState,
      loader.control,
      (predecessor) => {
        const loaded = loadCheckpointValue(loader, {
          path: predecessor.checkpoint_path,
          digest: predecessor.checkpoint_digest,
        });
        return withAuthorityTraversal(loader.session, loaded.binding, () => {
          assertExpectedDigest(loaded);
          return loaded.raw;
        });
      },
    );
    return Object.freeze({
      path: current.binding.path,
      semanticDigest: current.binding.semanticDigest,
      contentDigest: current.artifact.contentDigest,
      value,
    });
  });
}

function loadCheckpointValue(loader, reference) {
  const artifact = loadJsonArtifact(loader.session, reference.path, "queue checkpoint");
  assertSessionArtifact(loader.session, artifact, reference.path);
  const raw = loadedJsonValue(artifact, loader.session.root);
  const declaredDigest = digest(raw?.digest, "queue checkpoint digest");
  const { digest: _ignored, ...payload } = raw;
  const observedDigest = evidenceDigest(payload);
  if (declaredDigest !== observedDigest) {
    throw new Error(`${reference.path}: queue checkpoint semantic digest mismatch`);
  }
  const expectedDigest = digest(reference.digest, "expected queue checkpoint digest");
  const binding = bindQueueReference(
    loader.references, "queue-checkpoint", reference.path, observedDigest,
  );
  const existing = loader.values.get(binding.path);
  if (existing) return Object.freeze({ ...existing, expectedDigest });
  const loaded = Object.freeze({ artifact, binding, raw });
  loader.values.set(binding.path, loaded);
  return Object.freeze({ ...loaded, expectedDigest });
}

function assertExpectedDigest(loaded) {
  if (loaded.expectedDigest !== loaded.binding.semanticDigest) {
    throw new Error(`${loaded.binding.path}: queue checkpoint semantic digest mismatch`);
  }
}

function loaderRecord(value) {
  const record = LOADERS.get(value);
  if (!record) throw new Error("operation requires an exact queue checkpoint loader");
  return record;
}
