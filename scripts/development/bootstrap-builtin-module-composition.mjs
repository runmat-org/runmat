#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { bootstrapModuleComposition } from "./builtin-migration/module-composition/bootstrap.mjs";

try {
  const options = parse(process.argv.slice(2));
  if (options.help) {
    process.stdout.write("Usage: bootstrap-builtin-module-composition.mjs --repository PATH --reviewed-baseline PATH --trusted-reviewed-baseline-digest SHA256 --trusted-source-signer FINGERPRINT\n");
  } else {
    const reviewedBaseline = JSON.parse(fs.readFileSync(path.resolve(options.reviewedBaseline), "utf8"));
    const result = bootstrapModuleComposition({
      repository: options.repository,
      reviewedBaseline,
      trustedReviewedBaselineDigest: options.trustedReviewedBaselineDigest,
      trustedSignerFingerprint: options.trustedSourceSigner,
    });
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  }
} catch (error) {
  process.stderr.write(`module composition bootstrap: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function parse(arguments_) {
  if (arguments_.includes("--help") || arguments_.includes("-h")) return { help: true };
  const options = {
    repository: null, reviewedBaseline: null, trustedReviewedBaselineDigest: null,
    trustedSourceSigner: null, help: false,
  };
  while (arguments_.length) {
    const option = arguments_.shift();
    if (!["--repository", "--reviewed-baseline", "--trusted-reviewed-baseline-digest", "--trusted-source-signer"].includes(option)) throw new Error(`unknown option ${option}`);
    const value = arguments_.shift();
    if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
    const key = { "--reviewed-baseline": "reviewedBaseline", "--trusted-reviewed-baseline-digest": "trustedReviewedBaselineDigest", "--trusted-source-signer": "trustedSourceSigner" }[option] ?? option.slice(2);
    options[key] = value;
  }
  if (!options.repository || !options.reviewedBaseline || !options.trustedReviewedBaselineDigest || !options.trustedSourceSigner) {
    throw new Error("--repository, --reviewed-baseline, --trusted-reviewed-baseline-digest, and --trusted-source-signer are required");
  }
  return options;
}
