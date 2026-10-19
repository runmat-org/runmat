#!/usr/bin/env node
import process from "node:process";

import { clearStoppedOwnerRecoveryReservation } from "./builtin-migration/module-composition/transaction-lock-recovery.mjs";

try {
  const options = parse(process.argv.slice(2));
  if (options.help) {
    process.stdout.write("Usage: clear-builtin-module-composition-recovery-reservation.mjs --repository PATH --owner-token TOKEN --reservation-token TOKEN --confirm-owner-cannot-resume --confirm-recovery-process-cannot-resume\n");
  } else {
    const result = clearStoppedOwnerRecoveryReservation(options.repository, options);
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  }
} catch (error) {
  process.stderr.write(`module composition recovery reservation: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function parse(arguments_) {
  if (arguments_.includes("--help") || arguments_.includes("-h")) return { help: true };
  const options = {
    repository: null, ownerToken: null, reservationToken: null,
    ownerCannotResume: false, recoveryProcessCannotResume: false, help: false,
  };
  while (arguments_.length) {
    const option = arguments_.shift();
    const flag = {
      "--confirm-owner-cannot-resume": "ownerCannotResume",
      "--confirm-recovery-process-cannot-resume": "recoveryProcessCannotResume",
    }[option];
    if (flag) { if (options[flag]) throw new Error(`${option} may appear only once`); options[flag] = true; continue; }
    if (!["--repository", "--owner-token", "--reservation-token"].includes(option)) throw new Error(`unknown option ${option}`);
    const value = arguments_.shift();
    if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
    options[{ "--repository": "repository", "--owner-token": "ownerToken", "--reservation-token": "reservationToken" }[option]] = value;
  }
  if (!options.repository || !options.ownerToken || !options.reservationToken
    || !options.ownerCannotResume || !options.recoveryProcessCannotResume) throw new Error("all cleanup options and confirmations are required");
  return options;
}
