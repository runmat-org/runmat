import {
  assertAuthorityLoadSession, assertSessionArtifact, loadJsonArtifact,
  loadedJsonValue, withAuthorityTraversal,
} from "../authority-loading/index.mjs";
import { assertValidatedControl } from "../control.mjs";
import { evidenceDigest } from "../evidence.mjs";
import { validateQueueSeal } from "../queue-seal.mjs";
import { bindQueueReference, assertQueueReferenceIndex } from "./reference.mjs";

const LOADERS = new WeakMap();

export function openQueueSealLoader({ session, references, control }) {
  const authoritySession = assertAuthorityLoadSession(session);
  assertQueueReferenceIndex(references, authoritySession);
  const validatedControl = assertValidatedControl(control);
  const loader = Object.freeze({});
  LOADERS.set(loader, {
    session: authoritySession,
    references,
    control: validatedControl,
    seals: new Map(),
  });
  return loader;
}

export function loadQueueSeal(loaderValue, reference) {
  const loader = loaderRecord(loaderValue);
  const artifact = loadJsonArtifact(loader.session, reference.path, "queue seal");
  assertSessionArtifact(loader.session, artifact, reference.path);
  const raw = loadedJsonValue(artifact, loader.session.root);
  const observedDigest = evidenceDigest(raw);
  const binding = bindQueueReference(
    loader.references, "queue-seal", reference.path, observedDigest,
  );
  if (observedDigest !== reference.digest) {
    throw new Error(`queue seal ${reference.artifact_id} digest mismatch`);
  }
  const existing = loader.seals.get(binding.path);
  if (existing) {
    if (JSON.stringify(existing.reference) !== JSON.stringify(reference)) {
      throw new Error(`${binding.path}: queue seal reference controls were rebound`);
    }
    return existing.value;
  }
  return withAuthorityTraversal(loader.session, binding, () => {
    const value = validateQueueSeal(raw, reference, loader.control);
    loader.seals.set(binding.path, Object.freeze({
      reference: Object.freeze(structuredClone(reference)), value,
    }));
    return value;
  });
}

function loaderRecord(value) {
  const record = LOADERS.get(value);
  if (!record) throw new Error("operation requires an exact queue seal loader");
  return record;
}
