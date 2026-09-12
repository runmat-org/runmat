#!/usr/bin/env node
import process from "node:process";

import { recoverCompositionAfterStoppedOwner } from "./builtin-migration/module-composition/transaction.mjs";

try {
  const options = parse(process.argv.slice(2));
  if (options.help) {
    process.stdout.write("Usage: recover-builtin-module-composition.mjs --repository PATH --owner-token TOKEN --confirm-owner-cannot-resume\n");
  } else {
    const result = recoverCompositionAfterStoppedOwner(options.repository, {
      token: options.ownerToken, ownerCannotResume: options.confirmOwnerCannotResume,
    });
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  }
} catch (error) {
  process.stderr.write(`module composition recovery: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function parse(arguments_) {
  if (arguments_.includes("--help") || arguments_.includes("-h")) return { help: true };
  const options = { repository: null, ownerToken: null, confirmOwnerCannotResume: false, help: false };
  while (arguments_.length) {
    const option = arguments_.shift();
    if (option === "--confirm-owner-cannot-resume") {
      if (options.confirmOwnerCannotResume) throw new Error(`${option} may appear only once`);
      options.confirmOwnerCannotResume = true;
      continue;
    }
    if (!["--repository", "--owner-token"].includes(option)) throw new Error(`unknown option ${option}`);
    const value = arguments_.shift();
    if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
    if (option === "--repository") options.repository = value;
    else options.ownerToken = value;
  }
  if (!options.repository || !options.ownerToken || !options.confirmOwnerCannotResume) {
    throw new Error("--repository, --owner-token, and --confirm-owner-cannot-resume are required");
  }
  return options;
}
