#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { auditMigration, parseBatch } from "./builtin-migration/audit.mjs";
import { parseControlManifest } from "./builtin-migration/control.mjs";
import { buildControlDraft, freezeReviewedControl } from "./builtin-migration/control-draft.mjs";
import { validateControlReviewChain } from "./builtin-migration/control-authoring/authority.mjs";
import { buildControlAttestationTemplate, sealControlAttestation } from "./builtin-migration/control-authoring/attestation-template.mjs";
import { composeControlCandidate, parseControlCandidate } from "./builtin-migration/control-authoring/compose.mjs";
import { loadControlReviewSet } from "./builtin-migration/control-authoring/review-set.mjs";
import { buildControlOverlayScaffold, parseControlOverlayScaffold } from "./builtin-migration/control-authoring/scaffold.mjs";
import { indexControlReviews, initializeControlReviewTemplates } from "./builtin-migration/control-authoring/templates.mjs";
import { buildAuthorityComponentGraph } from "./builtin-migration/topology/components.mjs";
import { composeTopologyCandidate } from "./builtin-migration/topology/compose.mjs";
import { freezeReviewedTopology, parseReviewedTopology, reviewedTopologyView } from "./builtin-migration/topology/freeze.mjs";
import { dispositionInputFromControl } from "./builtin-migration/dispositions.mjs";
import { compileDispositionReview } from "./builtin-migration/disposition-review.mjs";
import { buildDispositionSeed, buildInventory, emptyDispositionInput, parseInventoryEvidence } from "./builtin-migration/inventory.mjs";
import { assertLeaseBaseInventory, issueLease, parseLease } from "./builtin-migration/lease.mjs";
import { runGateProducer } from "./builtin-migration/gate-adapter.mjs";
import { prepareIdentity } from "./builtin-migration/prepare.mjs";
import { buildQueue, emptyQueueState, validateQueueState } from "./builtin-migration/queue.mjs";
import { validateQueueCheckpoint } from "./builtin-migration/queue-checkpoint.mjs";
import { sealBundle, parseSealManifest } from "./builtin-migration/seal.mjs";
import { exact, kind } from "./builtin-migration/schema.mjs";
import { parseVerificationManifest } from "./builtin-migration/verify-schema.mjs";
import { verifyBatch } from "./builtin-migration/verify.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

try {
  run(parse(process.argv.slice(2)));
} catch (error) {
  process.stderr.write(`builtin migration factory: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function run(options) {
  if (options.help) { process.stdout.write(help()); return; }
  if (options.command === "draft-control") {
    emit(buildControlDraft(readJson(options.baselineInventory)), options.output);
    return;
  }
  if (options.command === "component-graph") {
    emit(buildAuthorityComponentGraph(readJson(options.baselineInventory)), options.output);
    return;
  }
  if (options.command === "compose-topology") {
    emit(composeTopologyFromOptions(options), options.output);
    return;
  }
  if (options.command === "freeze-topology") {
    const expected = composeTopologyFromOptions(options);
    emit(freezeReviewedTopology(readJson(options.candidate), readJson(options.attestation), expected), options.output);
    return;
  }
  if (options.command === "validate-topology") {
    const expected = composeTopologyFromOptions(options);
    emit(
      parseReviewedTopology(
        readJson(options.topology),
        readJson(options.candidate),
        readJson(options.attestation),
        expected,
      ),
      options.output,
    );
    return;
  }
  if (options.command === "compile-dispositions") {
    emit(compileDispositionReview(readJson(options.review), readJson(options.baselineInventory)), options.output);
    return;
  }
  if (options.command === "scaffold-control") {
    const inventory = parseInventoryEvidence(readJson(options.baselineInventory));
    emit(buildControlOverlayScaffold(inventory, readJson(options.draft), validatedTopologyFromOptions(options)), options.output);
    return;
  }
  if (options.command === "compose-control") {
    const inventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const topology = validatedTopologyFromOptions(options);
    emit(controlInputsFromOptions(options, inventory, topology).expectedCandidate, options.output);
    return;
  }
  if (options.command === "init-control-reviews") {
    const inventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const topology = validatedTopologyFromOptions(options);
    const scaffold = validatedScaffoldFromOptions(options, inventory, topology);
    emit(initializeControlReviewTemplates(options.reviewDirectory, scaffold, topology), options.output);
    return;
  }
  if (options.command === "index-control-reviews") {
    const inventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const topology = validatedTopologyFromOptions(options);
    const scaffold = validatedScaffoldFromOptions(options, inventory, topology);
    emit(indexControlReviews(options.reviewDirectory, options.reviewSetDirectory, { scaffold, topology, inventory }), options.output);
    return;
  }
  if (["scaffold-control-attestation", "seal-control-attestation"].includes(options.command)) {
    const inventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const topology = validatedTopologyFromOptions(options);
    const inputs = controlInputsFromOptions(options, inventory, topology);
    const candidate = parseControlCandidate(readJson(options.controlCandidate), inputs.expectedCandidate);
    const value = options.command === "scaffold-control-attestation"
      ? buildControlAttestationTemplate(candidate)
      : sealControlAttestation(readJson(options.attestationReview), candidate);
    emit(value, options.output);
    return;
  }
  if (options.command === "freeze-control") {
    const inventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const topology = validatedTopologyFromOptions(options);
    const authoring = controlAuthoringFromOptions(options, inventory, topology);
    emit(freezeReviewedControl(readJson(options.draft), inventory, topology, authoring.reviewedControl).value, options.output);
    return;
  }
  if (options.command === "issue-lease") {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlFromOptions(options, baseline);
    const leaseBase = parseInventoryEvidence(readJson(options.leaseBaseInventory));
    const queue = loadQueueAuthority(
      options.state, options.queueCheckpoint, options.trustedQueueCheckpointDigest, control,
    );
    emit(issueLease(
      readJson(options.request), control, repository, leaseBase, queue.state, queue.checkpoint,
    ), options.output);
    return;
  }
  if (options.command === "validate-control") {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    emit(parseControlFromOptions(options, baseline).value, options.output);
    return;
  }
  if (options.command === "produce-gate") {
    const controlBaseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlFromOptions(options, controlBaseline);
    const leaseBase = parseInventoryEvidence(readJson(options.leaseBaseInventory));
    const lease = parseLease(readJson(options.lease), control, repository);
    assertLeaseBaseInventory(lease, control, leaseBase);
    const subject = buildInventory(repository, dispositionInputFromControl(control), {
      compiledInventory: readJson(options.compiledInventory),
    });
    emit(runGateProducer({ control, lease, control_baseline_inventory: controlBaseline, lease_base_inventory: leaseBase, subject_inventory: subject, bundle_id: options.bundle, gate: options.gate, artifact_id: options.artifact, inputs: options.inputs ? readJson(options.inputs) : null }), options.output);
    return;
  }
  if (options.command === "verify") { runVerify(options); return; }
  if (options.command === "seal") { runSeal(options); return; }
  const compiledInventory = readJson(options.compiledInventory);
  const dispositions = options.dispositions ? readJson(options.dispositions) : emptyDispositionInput();
  const inventory = buildInventory(repository, dispositions, { compiledInventory });
  if (options.command === "inventory") emit(inventory, options.output);
  else if (options.command === "seed-dispositions") emit(buildDispositionSeed(inventory), options.output);
  else {
    const baselineInventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlFromOptions(options, baselineInventory);
    if (options.command === "queue" && inventory.digest !== baselineInventory.digest) throw new Error("queue requires the current inventory to equal the reviewed baseline inventory");
    if (options.command === "queue") {
      const queueState = options.state
        ? loadQueueAuthority(
          options.state, options.queueCheckpoint, options.trustedQueueCheckpointDigest, control,
        ).state
        : validateQueueState(emptyQueueState(control), control, () => null);
      emit(buildQueue(inventory, control, queueState), options.output);
    }
    else {
      const lease = parseLease(readJson(options.lease), control, repository);
      const leaseBase = parseInventoryEvidence(readJson(options.leaseBaseInventory));
      assertLeaseBaseInventory(lease, control, leaseBase);
      if (options.command === "prepare") {
        if (inventory.digest !== leaseBase.digest) throw new Error("prepare requires the current inventory to equal the lease base inventory");
        emit(prepareIdentity(repository, inventory, control, lease, options.identity, options.workspace), options.output);
      } else runAudit(options, baselineInventory, leaseBase, inventory, control, lease);
    }
  }
  if (inventory.diagnostics.some((entry) => entry.severity === "error")) process.exitCode = 1;
}

function runAudit(options, controlBaseline, leaseBase, subject, control, lease) {
  const batch = readJson(options.batch);
  parseBatch(batch);
  const evidencePath = path.resolve(options.evidence);
  const evidence = readJson(evidencePath);
  kind(evidence, 2, "runmat-builtin-migration-audit-evidence-manifest", "audit evidence manifest");
  exact(evidence, ["schema_version", "kind", "artifact_id", "authored_revision", "prepare_results", "source_dispositions", "gate_results"], "audit evidence manifest");
  const base = path.dirname(evidencePath);
  const load = (paths) => paths.map((entry) => readJson(path.resolve(base, entry)));
  const output = auditMigration(repository, controlBaseline, leaseBase, subject, control, lease, batch, { artifact_id: evidence.artifact_id, authored_revision: evidence.authored_revision, prepare_results: load(evidence.prepare_results), source_dispositions: load(evidence.source_dispositions), gate_results: load(evidence.gate_results) });
  emit(output, options.output);
  if (output.result !== "pass") process.exitCode = 1;
}

function loadQueueAuthority(statePath, checkpointPath, trustedCheckpointDigest, control) {
  const resolvedState = fs.realpathSync(path.resolve(statePath));
  const base = fs.realpathSync(path.dirname(resolvedState));
  const states = new Map();
  const activeStates = new Set();
  const loadWithinBase = (relativePath, label) => {
    const resolved = fs.realpathSync(path.resolve(base, relativePath));
    const relative = path.relative(base, resolved);
    if (relative === ".." || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
      throw new Error(`${label} escapes the queue authority directory: ${relativePath}`);
    }
    return resolved;
  };
  const loadState = (resolved) => {
    if (activeStates.has(resolved)) throw new Error("queue state predecessor chain contains a cycle");
    const raw = readJson(resolved);
    if (states.has(raw.digest)) return states.get(raw.digest);
    activeStates.add(resolved);
    try {
      const state = validateQueueState(
        raw, control,
        (reference) => readJson(loadWithinBase(reference.path, "queue seal reference")),
        (predecessor) => loadState(loadWithinBase(
          predecessor.state_path, "queue predecessor state reference",
        )),
      );
      states.set(state.stateDigest, state);
      return state;
    } finally { activeStates.delete(resolved); }
  };
  const state = loadState(resolvedState);
  const checkpoints = new Map();
  const activeCheckpoints = new Set();
  const loadCheckpoint = (resolved, expectedDigest, checkpointState) => {
    if (activeCheckpoints.has(resolved)) throw new Error("queue checkpoint predecessor chain contains a cycle");
    if (checkpoints.has(expectedDigest)) return checkpoints.get(expectedDigest);
    activeCheckpoints.add(resolved);
    try {
      const checkpoint = validateQueueCheckpoint(
        readJson(resolved), expectedDigest, checkpointState, control,
        (predecessor) => loadCheckpoint(
          loadWithinBase(predecessor.checkpoint_path, "queue predecessor checkpoint reference"),
          predecessor.checkpoint_digest,
          checkpointState.predecessorState,
        ).value,
      );
      checkpoints.set(expectedDigest, checkpoint);
      return checkpoint;
    } finally { activeCheckpoints.delete(resolved); }
  };
  const checkpoint = loadCheckpoint(
    loadWithinBase(checkpointPath, "queue checkpoint"), trustedCheckpointDigest, state,
  );
  return { state, checkpoint };
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
  const control = parseControlFromOptions(options, baseline);
  const lease = parseLease(readJson(options.lease), control, repository);
  const base = path.dirname(manifestPath);
  const load = (reference) => ({ reference, value: readJson(path.resolve(base, reference.path)) });
  const verification = load(manifest.verification).value;
  const output = sealBundle(
    manifest, verification, manifest.integration_gates.map(load),
    manifest.accepted_seals.map(load), control, repository, lease,
  );
  emit(output, options.output);
  if (output.result !== "pass") process.exitCode = 1;
}

function parse(arguments_) {
  if (arguments_.includes("--help") || arguments_.includes("-h")) return { help: true };
  const command = arguments_.shift();
  const commands = ["inventory", "queue", "seed-dispositions", "compile-dispositions", "draft-control", "component-graph", "compose-topology", "freeze-topology", "validate-topology", "scaffold-control", "init-control-reviews", "index-control-reviews", "compose-control", "scaffold-control-attestation", "seal-control-attestation", "freeze-control", "validate-control", "issue-lease", "produce-gate", "prepare", "audit", "verify", "seal"];
  const controlCommands = ["queue", "prepare", "audit", "freeze-control", "validate-control", "issue-lease", "produce-gate", "seal"];
  const controlAuthoringCommands = ["scaffold-control", "init-control-reviews", "index-control-reviews", "compose-control", "scaffold-control-attestation", "seal-control-attestation"];
  if (!commands.includes(command)) throw new Error(`expected ${commands.join(", ")}; use --help`);
  const options = { command, output: null, compiledInventory: null, baselineInventory: null, leaseBaseInventory: null, dispositions: null, control: null, draft: null, review: null, request: null, lease: null, state: null, queueCheckpoint: null, trustedQueueCheckpointDigest: null, batch: null, evidence: null, identity: null, workspace: null, manifest: null, bundle: null, gate: null, artifact: null, inputs: null, componentGraph: null, c01C03Review: null, c04C05Review: null, c06C07Review: null, reconciliation: null, stabilityCorrections: null, candidate: null, attestation: null, topology: null, controlScaffold: null, controlReviewSet: null, controlCandidate: null, controlAttestation: null, reviewDirectory: null, reviewSetDirectory: null, attestationReview: null, help: false };
  if (command === "prepare") options.identity = requireValue(arguments_, "prepare identity");
  while (arguments_.length) {
    const option = arguments_.shift();
    const fields = { "--output": "output", "--compiled-inventory": "compiledInventory", "--baseline-inventory": "baselineInventory", "--lease-base-inventory": "leaseBaseInventory", "--dispositions": "dispositions", "--control": "control", "--draft": "draft", "--review": "review", "--request": "request", "--lease": "lease", "--state": "state", "--queue-checkpoint": "queueCheckpoint", "--trusted-queue-checkpoint-digest": "trustedQueueCheckpointDigest", "--batch": "batch", "--evidence": "evidence", "--workspace": "workspace", "--manifest": "manifest", "--bundle": "bundle", "--gate": "gate", "--artifact": "artifact", "--inputs": "inputs", "--component-graph": "componentGraph", "--c01-c03-review": "c01C03Review", "--c04-c05-review": "c04C05Review", "--c06-c07-review": "c06C07Review", "--reconciliation": "reconciliation", "--stability-corrections": "stabilityCorrections", "--candidate": "candidate", "--attestation": "attestation", "--topology": "topology", "--control-scaffold": "controlScaffold", "--control-review-set": "controlReviewSet", "--control-candidate": "controlCandidate", "--control-attestation": "controlAttestation", "--review-directory": "reviewDirectory", "--review-set-directory": "reviewSetDirectory", "--attestation-review": "attestationReview" };
    if (!fields[option]) throw new Error(`unknown option ${option}`);
    options[fields[option]] = requireValue(arguments_, option);
  }
  const requiresCompiled = ["inventory", "queue", "seed-dispositions", "prepare", "audit", "produce-gate"].includes(command);
  if (requiresCompiled && !options.compiledInventory) throw new Error(`${command} requires --compiled-inventory`);
  if (["queue", "prepare", "audit", "validate-control", "produce-gate"].includes(command) && !options.control) throw new Error(`${command} requires --control`);
  if (["queue", "prepare", "audit", "compile-dispositions", "draft-control", "component-graph", "compose-topology", "freeze-topology", "validate-topology", ...controlAuthoringCommands, ...controlCommands].includes(command) && !options.baselineInventory) throw new Error(`${command} requires --baseline-inventory`);
  if (["compose-topology", "freeze-topology", "validate-topology", ...controlAuthoringCommands, ...controlCommands].includes(command) && (!options.componentGraph || !options.draft || !options.c01C03Review || !options.c04C05Review || !options.c06C07Review || !options.reconciliation || !options.stabilityCorrections)) {
    throw new Error(`${command} requires --component-graph, --draft, all three cohort reviews, --reconciliation, and --stability-corrections`);
  }
  if (["freeze-topology", "validate-topology", ...controlAuthoringCommands, ...controlCommands].includes(command) && (!options.candidate || !options.attestation)) throw new Error(`${command} requires --candidate and --attestation`);
  if (["validate-topology", ...controlAuthoringCommands, ...controlCommands].includes(command) && !options.topology) throw new Error(`${command} requires --topology`);
  if (command === "compile-dispositions" && !options.review) throw new Error("compile-dispositions requires --review");
  if (["init-control-reviews", "index-control-reviews", "compose-control", "scaffold-control-attestation", "seal-control-attestation", ...controlCommands].includes(command) && !options.controlScaffold) throw new Error(`${command} requires --control-scaffold`);
  if (["compose-control", "scaffold-control-attestation", "seal-control-attestation", ...controlCommands].includes(command) && !options.controlReviewSet) throw new Error(`${command} requires --control-review-set`);
  if (["scaffold-control-attestation", "seal-control-attestation", ...controlCommands].includes(command) && !options.controlCandidate) throw new Error(`${command} requires --control-candidate`);
  if (controlCommands.includes(command) && (!options.controlCandidate || !options.controlAttestation)) throw new Error(`${command} requires --control-candidate and --control-attestation`);
  if (["init-control-reviews", "index-control-reviews"].includes(command) && !options.reviewDirectory) throw new Error(`${command} requires --review-directory`);
  if (command === "index-control-reviews" && !options.reviewSetDirectory) throw new Error("index-control-reviews requires --review-set-directory");
  if (command === "seal-control-attestation" && !options.attestationReview) throw new Error("seal-control-attestation requires --attestation-review");
  if (command === "issue-lease" && (!options.control || !options.request || !options.leaseBaseInventory || !options.state || !options.queueCheckpoint || !options.trustedQueueCheckpointDigest)) throw new Error("issue-lease requires --control, --request, --lease-base-inventory, --state, --queue-checkpoint, and --trusted-queue-checkpoint-digest");
  if (command === "queue" && options.state
    && (!options.queueCheckpoint || !options.trustedQueueCheckpointDigest)) {
    throw new Error("queue with --state requires --queue-checkpoint and --trusted-queue-checkpoint-digest");
  }
  if (command === "produce-gate" && (!options.bundle || !options.gate || !options.artifact)) throw new Error("produce-gate requires --bundle, --gate, and --artifact");
  if (["prepare", "audit"].includes(command) && !options.lease) throw new Error(`${command} requires --lease`);
  if (["prepare", "audit", "produce-gate"].includes(command) && !options.leaseBaseInventory) throw new Error(`${command} requires --lease-base-inventory`);
  if (command === "produce-gate" && !options.lease) throw new Error("produce-gate requires --lease");
  if (command === "prepare" && !options.workspace) throw new Error("prepare requires --workspace outside the repository");
  if (command === "audit" && (!options.batch || !options.evidence)) throw new Error("audit requires --batch and --evidence");
  if (["verify", "seal"].includes(command) && !options.manifest) throw new Error(`${command} requires --manifest`);
  if (command === "seal" && (!options.control || !options.lease)) {
    throw new Error("seal requires --control and --lease");
  }
  return options;
}

function requireValue(arguments_, option) { const value = arguments_.shift(); if (!value || value.startsWith("-")) throw new Error(`${option} requires a value`); return value; }
function readJson(sourcePath) { return JSON.parse(fs.readFileSync(path.resolve(sourcePath), "utf8")); }
function emit(value, outputPath) { const encoded = `${JSON.stringify(value, null, 2)}\n`; if (outputPath) fs.writeFileSync(path.resolve(outputPath), encoded); else process.stdout.write(encoded); }
function help() {
  const topology = "--topology PATH --candidate PATH --attestation PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH";
  const controlReview = "--control-scaffold PATH --control-review-set PATH --control-candidate PATH --control-attestation PATH";
  return `Usage:\n` +
    `  builtin-migration-factory.mjs inventory|seed-dispositions --compiled-inventory PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs compile-dispositions --review PATH --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs draft-control --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs component-graph --baseline-inventory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs compose-topology --baseline-inventory PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs freeze-topology --candidate PATH --attestation PATH --baseline-inventory PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs validate-topology --topology PATH --candidate PATH --attestation PATH --baseline-inventory PATH --component-graph PATH --draft PATH --c01-c03-review PATH --c04-c05-review PATH --c06-c07-review PATH --reconciliation PATH --stability-corrections PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs scaffold-control --baseline-inventory PATH ${topology} [--output PATH]\n` +
    `  builtin-migration-factory.mjs init-control-reviews --baseline-inventory PATH ${topology} --control-scaffold PATH --review-directory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs index-control-reviews --baseline-inventory PATH ${topology} --control-scaffold PATH --review-directory PATH --review-set-directory PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs compose-control --baseline-inventory PATH ${topology} --control-scaffold PATH --control-review-set PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs scaffold-control-attestation --baseline-inventory PATH ${topology} --control-scaffold PATH --control-review-set PATH --control-candidate PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs seal-control-attestation --baseline-inventory PATH ${topology} --control-scaffold PATH --control-review-set PATH --control-candidate PATH --attestation-review PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs freeze-control --baseline-inventory PATH ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs validate-control --control PATH --baseline-inventory PATH ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs issue-lease --request PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH --state PATH --queue-checkpoint PATH --trusted-queue-checkpoint-digest SHA256 ${topology} ${controlReview} [--output PATH]\n` +
    `  builtin-migration-factory.mjs produce-gate --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH --lease PATH ${topology} ${controlReview} --bundle ID --gate NAME --artifact ID [--inputs PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs queue --compiled-inventory PATH --control PATH --baseline-inventory PATH ${topology} ${controlReview} [--state PATH --queue-checkpoint PATH --trusted-queue-checkpoint-digest SHA256] [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs prepare NAME --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH ${topology} ${controlReview} --lease PATH --workspace PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs audit --compiled-inventory PATH --control PATH --baseline-inventory PATH --lease-base-inventory PATH ${topology} ${controlReview} --lease PATH --batch PATH --evidence PATH [--dispositions PATH] [--output PATH]\n` +
    `  builtin-migration-factory.mjs verify --manifest PATH [--output PATH]\n` +
    `  builtin-migration-factory.mjs seal --manifest PATH --lease PATH --control PATH --baseline-inventory PATH ${topology} ${controlReview} [--output PATH]\n\n` +
    `Generated files are content-addressed development evidence, never production authority.\n`;
}

function composeTopologyFromOptions(options) {
  return composeTopologyCandidate({
    baselineInventory: readJson(options.baselineInventory),
    componentGraph: readJson(options.componentGraph),
    controlDraft: readJson(options.draft),
    reviewValues: new Map([
      ["c01_c03", readJson(options.c01C03Review)],
      ["c04_c05", readJson(options.c04C05Review)],
      ["c06_c07", readJson(options.c06C07Review)],
    ]),
    reconciliationValue: readJson(options.reconciliation),
    stabilityCorrectionsValue: readJson(options.stabilityCorrections),
  });
}

function validatedTopologyFromOptions(options) {
  const expected = composeTopologyFromOptions(options);
  const value = parseReviewedTopology(
    readJson(options.topology),
    readJson(options.candidate),
    readJson(options.attestation),
    expected,
  );
  return reviewedTopologyView(value);
}

function parseControlFromOptions(options, inventory) {
  const topology = validatedTopologyFromOptions(options);
  const { reviewedControl } = controlAuthoringFromOptions(options, inventory, topology);
  return parseControlManifest(readJson(options.control), {
    inventory,
    reviewedTopology: topology,
    reviewedControl,
  });
}

function controlAuthoringFromOptions(options, inventory, topology) {
  const inputs = controlInputsFromOptions(options, inventory, topology);
  const reviewedControl = validateControlReviewChain(
    readJson(options.controlCandidate),
    readJson(options.controlAttestation),
    inputs.expectedCandidate,
  );
  return { ...inputs, reviewedControl };
}

function controlInputsFromOptions(options, inventory, topology) {
  const scaffold = validatedScaffoldFromOptions(options, inventory, topology);
  const reviewSet = loadControlReviewSet(options.controlReviewSet, { scaffold, topology, inventory });
  const expectedCandidate = composeControlCandidate({ inventory, topology, scaffold, reviewSet });
  return { scaffold, reviewSet, expectedCandidate };
}

function validatedScaffoldFromOptions(options, inventory, topology) {
  return parseControlOverlayScaffold(
    readJson(options.controlScaffold),
    inventory,
    readJson(options.draft),
    topology,
  );
}
