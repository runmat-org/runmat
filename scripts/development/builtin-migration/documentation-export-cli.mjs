#!/usr/bin/env node
import process from "node:process";

import { runDocumentationExport } from "./documentation-export.mjs";

try {
  if (process.argv.length !== 2) throw new Error("this reviewed producer accepts no arguments");
  const result = runDocumentationExport();
  process.stdout.write(result.stdout);
  process.stderr.write(result.stderr);
  process.exitCode = result.status;
} catch (error) {
  process.stderr.write(`builtin documentation export: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}
