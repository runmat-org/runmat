#!/usr/bin/env node
import path from "node:path";
import process from "node:process";

import {
  canonicalEvidencePath, publishEvidenceBytes,
} from "./builtin-migration/atomic-evidence-publication.mjs";
import { deriveModuleCompositionBaselineCandidate } from "./builtin-migration/module-composition/baseline-candidate.mjs";
import { buildModuleCompositionBaselineReviewTemplate } from "./builtin-migration/module-composition/baseline-review.mjs";

try {
  const options = parse(process.argv.slice(2));
  if (options.help) process.stdout.write("Usage: scaffold-builtin-module-composition-baseline.mjs --repository PATH --trusted-source-signer FINGERPRINT --candidate PATH --review PATH\n");
  else {
    const candidate = deriveModuleCompositionBaselineCandidate(options.repository, options.trustedSourceSigner);
    writeNewJson(options.candidate, candidate);
    writeNewJson(options.review, buildModuleCompositionBaselineReviewTemplate(candidate));
    process.stdout.write(`${candidate.digest}\n`);
  }
} catch (error) {
  process.stderr.write(`module composition baseline scaffold: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function parse(values) {
  if (values.includes("--help") || values.includes("-h")) return { help: true };
  const result = { repository: null, trustedSourceSigner: null, candidate: null, review: null, help: false };
  const fields = { "--repository": "repository", "--trusted-source-signer": "trustedSourceSigner", "--candidate": "candidate", "--review": "review" };
  while (values.length) {
    const option = values.shift(); const field = fields[option];
    if (!field) throw new Error(`unknown option ${option}`);
    const value = values.shift(); if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
    result[field] = value;
  }
  if (Object.entries(result).some(([key, value]) => key !== "help" && value === null)) throw new Error("--repository, --trusted-source-signer, --candidate, and --review are required");
  return result;
}

function writeNewJson(target, value) {
  const resolved = path.resolve(target);
  publishEvidenceBytes(canonicalEvidencePath(resolved), `${JSON.stringify(value, null, 2)}\n`, {
    createParentDirectories: true,
  });
}
