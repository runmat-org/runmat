#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

import { buildExampleGateProof } from "./example-gate.mjs";

try {
  const args = process.argv.slice(2);
  if (args.length !== 2 || args[0] !== "--manifest" || !args[1]) throw new Error("usage: example-gate-cli.mjs --manifest PATH");
  const manifest = JSON.parse(fs.readFileSync(path.resolve(args[1]), "utf8"));
  const proof = buildExampleGateProof(manifest);
  process.stdout.write(`${JSON.stringify(proof)}\n`);
  if (proof.result !== "pass") process.exitCode = 1;
} catch (error) {
  process.stderr.write(`builtin example gate: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}
