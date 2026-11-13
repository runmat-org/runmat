#!/usr/bin/env node
import fs from "node:fs";
import process from "node:process";

import { buildExampleGateProof } from "./example-gate.mjs";
import { readExampleGateInput } from "./example-gate-input.mjs";

try {
  if (process.argv.length !== 2) throw new Error("this reviewed producer accepts no arguments");
  const input = readExampleGateInput(fs.readFileSync(0, "utf8"));
  const proof = buildExampleGateProof(input.manifest, input.manifest_evidence, input.evidence_root);
  process.stdout.write(`${JSON.stringify(proof)}\n`);
  if (proof.result !== "pass") process.exitCode = 1;
} catch (error) {
  process.stderr.write(`builtin example gate: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}
