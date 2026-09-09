#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { auditInventory, parseBatch } from "./builtin-migration/audit.mjs";
import { buildDispositionSeed, buildInventory, emptyDispositionInput } from "./builtin-migration/inventory.mjs";
import { prepareIdentity } from "./builtin-migration/prepare.mjs";
import { buildQueue } from "./builtin-migration/queue.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

try {
  const options = parse(process.argv.slice(2));
  if (options.help) {
    process.stdout.write(help());
    process.exit(0);
  }
  const dispositions = options.dispositions ? readJson(options.dispositions) : emptyDispositionInput();
  const inventory = buildInventory(repository, dispositions);
  let output;
  if (options.command === "inventory") output = inventory;
  else if (options.command === "queue") output = buildQueue(inventory);
  else if (options.command === "seed-dispositions") output = buildDispositionSeed(inventory);
  else if (options.command === "audit") {
    const identities = options.batch ? parseBatch(readJson(options.batch)) : options.identities;
    output = auditInventory(inventory, identities);
    if (output.result !== "pass") process.exitCode = 1;
  } else output = prepareIdentity(repository, inventory, options.identity, options.workspace);
  emit(output, options.output);
  if (inventory.diagnostics.some((entry) => entry.severity === "error")) process.exitCode = 1;
} catch (error) {
  process.stderr.write(`builtin migration factory: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function parse(arguments_) {
  if (arguments_.includes("--help") || arguments_.includes("-h")) return { help: true };
  const command = arguments_.shift();
  if (!["inventory", "queue", "seed-dispositions", "audit", "prepare"].includes(command)) throw new Error("expected inventory, queue, seed-dispositions, audit, or prepare; use --help");
  const options = { command, dispositions: null, output: null, batch: null, identities: [], identity: null, workspace: null, help: false };
  if (command === "prepare") options.identity = requireValue(arguments_, "prepare identity");
  while (arguments_.length) {
    const option = arguments_.shift();
    if (option === "--dispositions") options.dispositions = requireValue(arguments_, option);
    else if (option === "--output") options.output = requireValue(arguments_, option);
    else if (option === "--batch") options.batch = requireValue(arguments_, option);
    else if (option === "--identity") options.identities.push(requireValue(arguments_, option));
    else if (option === "--workspace") options.workspace = requireValue(arguments_, option);
    else throw new Error(`unknown option ${option}`);
  }
  if (command === "audit" && Boolean(options.batch) === Boolean(options.identities.length)) throw new Error("audit requires either one --batch or one or more --identity options");
  if (command === "prepare" && !options.workspace) throw new Error("prepare requires --workspace outside the repository");
  if (command !== "audit" && (options.batch || options.identities.length)) throw new Error(`${command} does not accept audit selection options`);
  if (command !== "prepare" && options.workspace) throw new Error(`${command} does not accept --workspace`);
  return options;
}

function requireValue(arguments_, option) {
  const value = arguments_.shift();
  if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`);
  return value;
}

function readJson(sourcePath) { return JSON.parse(fs.readFileSync(path.resolve(sourcePath), "utf8")); }
function emit(value, outputPath) {
  const encoded = `${JSON.stringify(value, null, 2)}\n`;
  if (outputPath) fs.writeFileSync(path.resolve(outputPath), encoded, "utf8");
  else process.stdout.write(encoded);
}

function help() {
  return `Usage:\n` +
    `  node scripts/development/builtin-migration-factory.mjs inventory|queue [--dispositions PATH] [--output PATH]\n` +
    `  node scripts/development/builtin-migration-factory.mjs seed-dispositions [--output PATH]\n` +
    `  node scripts/development/builtin-migration-factory.mjs prepare NAME --workspace PATH [--dispositions PATH] [--output PATH]\n` +
    `  node scripts/development/builtin-migration-factory.mjs audit (--identity NAME...|--batch PATH) [--dispositions PATH] [--output PATH]\n\n` +
    `Generated JSON and prepare workspaces are development evidence, never production authority.\n` +
    `prepare requires a workspace outside the repository and never edits RunMat source.\n` +
    `seed-dispositions emits explicit unreviewed rows. A reviewed row must set review.status\n` +
    `to reviewed, provide review.evidence, and choose canonical, alias, or internal.\n`;
}
