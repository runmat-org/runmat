#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { auditMigration, parseBatch } from "./builtin-migration/audit.mjs";
import { parseControlManifest } from "./builtin-migration/control.mjs";
import { buildControlDraft, freezeReviewedControl } from "./builtin-migration/control-draft.mjs";
import { dispositionInputFromControl } from "./builtin-migration/dispositions.mjs";
import { compileDispositionReview } from "./builtin-migration/disposition-review.mjs";
import { buildDispositionSeed, buildInventory, emptyDispositionInput, parseInventoryEvidence } from "./builtin-migration/inventory.mjs";
import { issueLease, parseLease } from "./builtin-migration/lease.mjs";
import { runGateProducer } from "./builtin-migration/gate-adapter.mjs";
import { prepareIdentity } from "./builtin-migration/prepare.mjs";
import { buildQueue, emptyQueueState } from "./builtin-migration/queue.mjs";
import { sealBundle, parseSealManifest } from "./builtin-migration/seal.mjs";
import { exact, kind } from "./builtin-migration/schema.mjs";
import { parseVerificationManifest } from "./builtin-migration/verify-schema.mjs";
import { verifyBatch } from "./builtin-migration/verify.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

try {
  const options = parse(process.argv.slice(2));
  if (options.help) { process.stdout.write(help()); process.exit(0); }
  if (options.command === "draft-control") {
    emit(buildControlDraft(readJson(options.baselineInventory)), options.output);
    process.exit(0);
  }
  if (options.command === "compile-dispositions") {
    emit(compileDispositionReview(readJson(options.review), readJson(options.baselineInventory)), options.output);
    process.exit(0);
  }
  if (options.command === "freeze-control") {
    emit(freezeReviewedControl(readJson(options.draft), readJson(options.control), readJson(options.baselineInventory)).value, options.output);
    process.exit(0);
  }
  if (options.command === "issue-lease") {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlManifest(readJson(options.control), baseline);
    emit(issueLease(readJson(options.request), control), options.output);
    process.exit(0);
  }
  if (options.command === "validate-control") {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    emit(parseControlManifest(readJson(options.control), baseline).value, options.output);
    process.exit(0);
  }
  if (options.command === "produce-gate") {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlManifest(readJson(options.control), baseline);
    const subject = buildInventory(repository, dispositionInputFromControl(control), {
      compiledInventory: readJson(options.compiledInventory),
    });
    emit(runGateProducer({ control, baseline_inventory: baseline, subject_inventory: subject, bundle_id: options.bundle, gate: options.gate, artifact_id: options.artifact, inputs: options.inputs ? readJson(options.inputs) : null }), options.output);
    process.exit(0);
  }
  if (options.command === "verify") { runVerify(options); process.exit(process.exitCode ?? 0); }
  if (options.command === "seal") { runSeal(options); process.exit(process.exitCode ?? 0); }
  const compiledInventory = readJson(options.compiledInventory);
  const dispositions = options.dispositions ? readJson(options.dispositions) : emptyDispositionInput();
  const inventory = buildInventory(repository, dispositions, { compiledInventory });
  if (options.command === "inventory") emit(inventory, options.output);
  else if (options.command === "seed-dispositions") emit(buildDispositionSeed(inventory), options.output);
  else {
    const baselineInventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlManifest(readJson(options.control), baselineInventory);
    if (["queue", "prepare"].includes(options.command) && inventory.digest !== baselineInventory.digest) throw new Error(`${options.command} requires the current inventory to equal the reviewed baseline inventory`);
    if (options.command === "queue") emit(buildQueue(inventory, control, options.state ? readJson(options.state) : emptyQueueState()), options.output);
    else {
      const lease = parseLease(readJson(options.lease), control);
      if (options.command === "prepare") emit(prepareIdentity(repository, inventory, control, lease, options.identity, options.workspace), options.output);
      else runAudit(options, baselineInventory, inventory, control, lease);
    }
  }
  if (inventory.diagnostics.some((entry) => entry.severity === "error")) process.exitCode = 1;
} catch (error) {
  process.stderr.write(`builtin migration factory: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function runAudit(options, baseline, subject, control, lease) {
  const batch = readJson(options.batch);
  parseBatch(batch);
  const evidencePath = path.resolve(options.evidence);
  const evidence = readJson(evidencePath);
  kind(evidence, 1, "runmat-builtin-migration-audit-evidence-manifest", "audit evidence manifest");
  exact(evidence, ["schema_version", "kind", "artifact_id", "prepare_results", "source_dispositions", "gate_results"], "audit evidence manifest");
  const base = path.dirname(evidencePath);
  const load = (paths) => paths.map((entry) => readJson(path.resolve(base, entry)));
  const changedPaths = gitChangedPaths(lease.value.base_revision);
  const output = auditMigration(repository, baseline, subject, control, lease, batch, { artifact_id: evidence.artifact_id, changed_paths: changedPaths, prepare_results: load(evidence.prepare_results), source_dispositions: load(evidence.source_dispositions), gate_results: load(evidence.gate_results) });
  emit(output, options.output);
  if (output.result !== "pass") process.exitCode = 1;
}

function runVerify(options) {
  const manifestPath = path.resolve(options.manifest);
  const manifest = parseVerificationManifest(readJson(manifestPath));
  const base = path.dirname(manifestPath);
  const load = (reference) => ({ reference, value: readJson(path.resolve(base, reference.path)) });
  const output = verifyBatch(manifest, load(manifest.audit), manifest.gate_results.map(load));
  emit(output, options.output);
  if (output.result !== "pass") process.exitCode = 1;
}

function runSeal(options) {
  const manifestPath = path.resolve(options.manifest);
  const manifest = parseSealManifest(readJson(manifestPath));
  const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
  const control = parseControlManifest(readJson(options.control), baseline);
  const base = path.dirname(manifestPath);
  const load = (reference) => ({ reference, value: readJson(path.resolve(base, reference.path)) });
  const verification = load(manifest.verification).value;
  const output = sealBundle(manifest, verification, manifest.integration_gates.map(load), manifest.prerequisite_seals.map(load), control, repository);
  emit(output, options.output);
  if (output.result !== "pass") process.exitCode = 1;
}

function gitChangedPaths(baseRevision) {
  return execFileSync("git", ["diff", "--name-only", baseRevision.slice("git:".length), "--"], { cwd: repository, encoding: "utf8" }).split("\n").filter(Boolean);
}

function parse(arguments_) {
  if (arguments_.includes("--help") || arguments_.includes("-h")) return { help: true };
  const command = arguments_.shift();
  const commands = ["inventory", "queue", "seed-dispositions", "compile-dispositions", "draft-control", "freeze-control", "validate-control", "issue-lease", "produce-gate", "prepare", "audit", "verify", "seal"];
  if (!commands.includes(command)) throw new Error(`expected ${commands.join(", ")}; use --help`);
  const options = { command, output: null, compiledInventory: null, baselineInventory: null, dispositions: null, control: null, draft: null, review: null, request: null, lease: null, state: null, batch: null, evidence: null, identity: null, workspace: null, manifest: null, bundle: null, gate: null, artifact: null, inputs: null, help: false };
  if (command === "prepare") options.identity = requireValue(arguments_, "prepare identity");
  while (arguments_.length) {
    const option = arguments_.shift();
    const fields = { "--output": "output", "--compiled-inventory": "compiledInventory", "--baseline-inventory": "baselineInventory", "--dispositions": "dispositions", "--control": "control", "--draft": "draft", "--review": "review", "--request": "request", "--lease": "lease", "--state": "state", "--batch": "batch", "--evidence": "evidence", "--workspace": "workspace", "--manifest": "manifest", "--bundle": "bundle", "--gate": "gate", "--artifact": "artifact", "--inputs": "inputs" };
    if (!fields[option]) throw new Error(`unknown option ${option}`);
    options[fields[option]] = requireValue(arguments_, option);
  }
  const requiresCompiled = ["inventory", "queue", "seed-dispositions", "prepare", "audit", "produce-gate"].includes(command);
  if (requiresCompiled && !options.compiledInventory) throw new Error(`${command} requires --compiled-inventory`);
  if (["queue", "prepare", "audit", "validate-control", "produce-gate"].includes(command) && !options.control) throw new Error(`${command} requires --control`);
  if (["queue", "prepare", "audit", "compile-dispositions", "draft-control", "freeze-control", "validate-control", "issue-lease", "produce-gate", "seal"].includes(command) && !options.baselineInventory) throw new Error(`${command} requires --baseline-inventory`);
  if (command === "compile-dispositions" && !options.review) throw new Error("compile-dispositions requires --review");
  if (command === "freeze-control" && (!options.control || !options.draft)) throw new Error("freeze-control requires --control and --draft");
  if (command === "issue-lease" && (!options.control || !options.request)) throw new Error("issue-lease requires --control and --request");
  if (command === "produce-gate" && (!options.bundle || !options.gate || !options.artifact)) throw new Error("produce-gate requires --bundle, --gate, and --artifact");
  if (["prepare", "audit"].includes(command) && !options.lease) throw new Error(`${command} requires --lease`);
  if (command === "prepare" && !options.workspace) throw new Error("prepare requires --workspace outside the repository");
  if (command === "audit" && (!options.batch || !options.evidence)) throw new Error("audit requires --batch and --evidence");
  if (["verify", "seal"].includes(command) && !options.manifest) throw new Error(`${command} requires --manifest`);
  if (command === "seal" && !options.control) throw new Error("seal requires --control");
  return options;
}

function requireValue(arguments_, option) { const value = arguments_.shift(); if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`); return value; }
function readJson(sourcePath) { return JSON.parse(fs.readFileSync(path.resolve(sourcePath), "utf8")); }
function emit(value, outputPath) { const encoded = `${JSON.stringify(value, null, 2)}\n`; if (outputPath) fs.writeFileSync(path.resolve(outputPath), encoded); else process.stdout.write(encoded); }
function help() {
  return `Usage:\n` +
    `  builtin-migration-factory.mjs inventory|seed-dispositions --compiled-inventory PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs compile-dispositions --review PATH --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs draft-control --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs freeze-control --draft PATH --control PATH --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs validate-control --control PATH --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs issue-lease --request PATH --control PATH --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs produce-gate --compiled-inventory PATH --control PATH --baseline-inventory PATH --bundle ID --gate NAME --artifact ID [--inputs PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs queue --compiled-inventory PATH --control PATH --baseline-inventory PATH [--state PATH] [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs prepare NAME --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease PATH --workspace PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs audit --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease PATH --batch PATH --evidence PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs verify --manifest PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs seal --manifest PATH --control PATH --baseline-inventory PATH [--output PATH]\n\n` +
    `Generated files are content-addressed development evidence, never production authority.\n`;
}
