import fs from "node:fs";
import path from "node:path";

import { canonicalJson, contentDigest } from "./evidence.mjs";
import { parseExampleEvidenceRoot, parseExampleGateManifest } from "./example-gate.mjs";
import { absolutePath, digest, exact, integer, kind } from "./schema.mjs";

const INPUT_KIND = "runmat-builtin-example-gate-input";
export const EXAMPLE_GATE_MANIFEST_FILENAME = "example-gate-manifest.json";

export function stageExampleGateInput(inputs, temporaryDirectory, expected) {
  exact(inputs, ["artifact_output", "evidence_root", "example_manifest"], "example gate producer inputs");
  const evidenceRoot = parseExampleEvidenceRoot({
    path: absolutePath(inputs.evidence_root, "example evidence root"),
    filesystem_id: expected.evidence_storage.filesystem_id,
  }, expected.evidence_storage);
  const manifest = parseExampleGateManifest(inputs.example_manifest, evidenceRoot);
  if (manifest.source_revision !== expected.source_revision
    || JSON.stringify(manifest.identities) !== JSON.stringify([...expected.identities].sort())) {
    throw new Error("example gate manifest has stale or mismatched bundle provenance");
  }
  const directory = fs.realpathSync(absolutePath(temporaryDirectory, "example gate temporary directory"));
  const manifestPath = path.join(directory, EXAMPLE_GATE_MANIFEST_FILENAME);
  const bytes = Buffer.from(`${canonicalJson(manifest)}\n`);
  fs.writeFileSync(manifestPath, bytes, { flag: "wx" });
  const manifestEvidence = {
    path: fs.realpathSync(manifestPath),
    byte_length: bytes.length,
    content_digest: contentDigest(bytes),
  };
  const envelope = {
    schema_version: 1,
    kind: INPUT_KIND,
    evidence_root: evidenceRoot,
    manifest: manifestEvidence,
  };
  return {
    stdin: `${canonicalJson(envelope)}\n`,
    manifest_evidence: manifestEvidence,
    evidence_root: evidenceRoot,
  };
}

export function readExampleGateInput(encoded) {
  let value;
  try { value = JSON.parse(encoded); } catch (error) { throw new Error(`example gate input is not valid JSON: ${error.message}`); }
  kind(value, 1, INPUT_KIND, "example gate input");
  exact(value, ["schema_version", "kind", "evidence_root", "manifest"], "example gate input");
  const evidenceRoot = parseExampleEvidenceRoot(value.evidence_root);
  const manifestEvidence = parseManifestEvidence(value.manifest);
  const stat = fs.lstatSync(manifestEvidence.path);
  if (!stat.isFile() || fs.realpathSync(manifestEvidence.path) !== manifestEvidence.path) {
    throw new Error("example gate manifest must be a canonical regular file");
  }
  const bytes = fs.readFileSync(manifestEvidence.path);
  if (bytes.length !== manifestEvidence.byte_length || contentDigest(bytes) !== manifestEvidence.content_digest) {
    throw new Error("example gate manifest bytes differ from the adapter-owned input evidence");
  }
  let manifest;
  try { manifest = JSON.parse(bytes); } catch (error) { throw new Error(`example gate manifest is not valid JSON: ${error.message}`); }
  return {
    manifest: parseExampleGateManifest(manifest, evidenceRoot),
    manifest_evidence: manifestEvidence,
    evidence_root: evidenceRoot,
  };
}

function parseManifestEvidence(value) {
  exact(value, ["path", "byte_length", "content_digest"], "example gate manifest evidence");
  return {
    path: absolutePath(value.path, "example gate manifest evidence path"),
    byte_length: validateByteLength(value.byte_length),
    content_digest: validateDigest(value.content_digest),
  };
}

function validateByteLength(value) {
  integer(value, "example gate manifest evidence byte length", 0);
  return value;
}

function validateDigest(value) {
  digest(value, "example gate manifest evidence content digest");
  return value;
}
