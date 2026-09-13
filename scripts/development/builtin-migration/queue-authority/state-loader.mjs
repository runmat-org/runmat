import {
  assertAuthorityLoadSession, assertSessionArtifact, loadJsonArtifact,
  loadedJsonValue, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { validateQueueStateFromAuthority } from "../queue.mjs";
import { digest } from "../schema.mjs";
import { bindQueueReference, assertQueueReferenceIndex } from "./reference.mjs";
import { loadQueueSeal } from "./seal-loader.mjs";

const LOADERS = new WeakMap();

export function openQueueStateLoader({
  session, references, seals, control, validatePilotTransition = null,
}) {
  const authoritySession = assertAuthorityLoadSession(session);
  assertQueueReferenceIndex(references, authoritySession);
  const loader = Object.freeze({});
  LOADERS.set(loader, {
    session: authoritySession,
    references,
    seals,
    control: assertValidatedControl(control),
    validatePilotTransition,
    states: new Map(),
  });
  return loader;
}

export function loadQueueState(loaderValue, reference) {
  const loader = loaderRecord(loaderValue);
  const artifact = loadJsonArtifact(loader.session, reference.path, "queue state");
  assertSessionArtifact(loader.session, artifact, reference.path);
  const raw = loadedJsonValue(artifact, loader.session.root);
  const declaredDigest = digest(raw?.digest, "queue state digest");
  const { digest: _ignored, ...payload } = raw;
  const observedDigest = evidenceDigest(payload);
  const expectedDigest = reference.digest === null
    ? null : digest(reference.digest, "expected queue state digest");
  if (declaredDigest !== observedDigest) {
    throw new Error(`${reference.path}: queue state semantic digest mismatch`);
  }
  const binding = bindQueueReference(
    loader.references, "queue-state", reference.path, observedDigest,
  );
  const existing = loader.states.get(binding.path);
  if (existing) {
    if (expectedDigest !== null && expectedDigest !== existing.semanticDigest) {
      throw new Error(`${reference.path}: queue state semantic digest mismatch`);
    }
    return existing;
  }
  return withAuthorityTraversal(loader.session, binding, () => {
    if (expectedDigest !== null && expectedDigest !== observedDigest) {
      throw new Error(`${reference.path}: queue state semantic digest mismatch`);
    }
    const value = validateQueueStateFromAuthority(
      raw,
      loader.control,
      (sealReference) => loadQueueSeal(loader.seals, sealReference),
      (predecessor) => loadQueueState(loaderValue, {
        path: predecessor.state_path,
        digest: predecessor.state_digest,
      }).value,
      loader.validatePilotTransition,
    );
    const loaded = Object.freeze({
      path: binding.path,
      semanticDigest: binding.semanticDigest,
      contentDigest: artifact.contentDigest,
      value,
    });
    loader.states.set(binding.path, loaded);
    return loaded;
  });
}

function loaderRecord(value) {
  const record = LOADERS.get(value);
  if (!record) throw new Error("operation requires an exact queue state loader");
  return record;
}
