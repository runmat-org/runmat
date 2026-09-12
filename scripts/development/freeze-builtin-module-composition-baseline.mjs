#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { parseModuleCompositionBaselineCandidate } from "./builtin-migration/module-composition/baseline-candidate.mjs";
import { freezeReviewedModuleCompositionBaseline } from "./builtin-migration/module-composition/baseline-authority.mjs";
import { sealModuleCompositionBaselineReview } from "./builtin-migration/module-composition/baseline-review.mjs";

try {
  const options = parse(process.argv.slice(2));
  if (options.help) process.stdout.write("Usage: freeze-builtin-module-composition-baseline.mjs --repository PATH --trusted-source-signer FINGERPRINT --candidate PATH --review PATH --output PATH\n");
  else {
    const candidate = parseModuleCompositionBaselineCandidate(read(options.candidate), options.repository, options.trustedSourceSigner);
    const review = sealModuleCompositionBaselineReview(read(options.review), candidate);
    const baseline = freezeReviewedModuleCompositionBaseline(candidate, review);
    fs.writeFileSync(path.resolve(options.output), `${JSON.stringify(baseline, null, 2)}\n`, { flag: "wx" });
    process.stdout.write(`${baseline.digest}\n`);
  }
} catch (error) {
  process.stderr.write(`module composition baseline freeze: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function parse(values) {
  if (values.includes("--help") || values.includes("-h")) return { help: true };
  const result = { repository: null, trustedSourceSigner: null, candidate: null, review: null, output: null, help: false };
  const fields = { "--repository": "repository", "--trusted-source-signer": "trustedSourceSigner", "--candidate": "candidate", "--review": "review", "--output": "output" };
  while (values.length) {
    const option = values.shift(); const field = fields[option];
    if (!field) throw new Error(`unknown option ${option}`);
    const value = values.shift(); if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
    result[field] = value;
  }
  if (Object.entries(result).some(([key, value]) => key !== "help" && value === null)) throw new Error("--repository, --trusted-source-signer, --candidate, --review, and --output are required");
  return result;
}

function read(target) { return JSON.parse(fs.readFileSync(path.resolve(target), "utf8")); }
