#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { auditMigration, parseBatch } from "./builtin-migration/audit.mjs";
import {
  canonicalEvidencePath, publishEvidenceBytes,
} from "./builtin-migration/atomic-evidence-publication.mjs";
import { parseControlManifest } from "./builtin-migration/control.mjs";
import { buildControlDraft, freezeReviewedControl } from "./builtin-migration/control-draft.mjs";
import { validateControlReviewChain } from "./builtin-migration/control-authoring/authority.mjs";
import { buildControlAttestationTemplate, sealControlAttestation } from "./builtin-migration/control-authoring/attestation-template.mjs";
import { composeControlCandidate, parseControlCandidate } from "./builtin-migration/control-authoring/compose.mjs";
import { loadControlReviewSet } from "./builtin-migration/control-authoring/review-set.mjs";
import {
  validateBundleControlReviewInput, validateGlobalControlReviewInput,
} from "./builtin-migration/control-authoring/review-validation.mjs";
import { buildControlOverlayScaffold, parseControlOverlayScaffold } from "./builtin-migration/control-authoring/scaffold.mjs";
import { indexControlReviews, initializeControlReviewTemplates } from "./builtin-migration/control-authoring/templates.mjs";
import { buildAuthorityComponentGraph } from "./builtin-migration/topology/components.mjs";
import { composeTopologyCandidate } from "./builtin-migration/topology/compose.mjs";
import { freezeReviewedTopology, parseReviewedTopology, reviewedTopologyView } from "./builtin-migration/topology/freeze.mjs";
import { dispositionInputFromControl } from "./builtin-migration/dispositions.mjs";
import { compileDispositionReview } from "./builtin-migration/disposition-review.mjs";
import { factoryCliHelp } from "./builtin-migration/factory-cli/help.mjs";
import { parseFactoryCliArguments } from "./builtin-migration/factory-cli/parse.mjs";
import { PILOT_COMMANDS } from "./builtin-migration/factory-cli/pilot-contract.mjs";
import { runPilotLifecycleCommand } from "./builtin-migration/factory-cli/pilot-lifecycle.mjs";
import { runInitialQueueCommand } from "./builtin-migration/factory-cli/queue-initialization.mjs";
import { buildDispositionSeed, buildInventory, emptyDispositionInput, parseInventoryEvidence } from "./builtin-migration/inventory.mjs";
import { assertLeaseBaseInventory, issueLease, parseLease } from "./builtin-migration/lease.mjs";
import { runGateProducer } from "./builtin-migration/gate-adapter.mjs";
import { materializeEffectiveModuleComposition } from "./builtin-migration/module-composition/materialize.mjs";
import { prepareIdentity } from "./builtin-migration/prepare.mjs";
import { buildQueue, emptyQueueState, validateQueueState } from "./builtin-migration/queue.mjs";
import {
  loadQueueAuthorityFromCliPaths,
} from "./builtin-migration/queue-authority/index.mjs";
import { sealBundle, parseSealManifest } from "./builtin-migration/seal.mjs";
import { exact, kind } from "./builtin-migration/schema.mjs";
import { parseVerificationManifest } from "./builtin-migration/verify-schema.mjs";
import { verifyBatch } from "./builtin-migration/verify.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

try {
  run(parseFactoryCliArguments(process.argv.slice(2)));
} catch (error) {
  process.stderr.write(`builtin migration factory: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 2;
}

function run(options) {
  if (options.help) { process.stdout.write(factoryCliHelp()); return; }
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
  if (["validate-global-control-review", "validate-bundle-control-review"].includes(options.command)) {
    const inventory = parseInventoryEvidence(readJson(options.baselineInventory));
    const topology = validatedTopologyFromOptions(options);
    const scaffold = validatedScaffoldFromOptions(options, inventory, topology);
    const context = { scaffold, topology, inventory };
    const report = options.command === "validate-global-control-review"
      ? validateGlobalControlReviewInput(readJson(options.globalReview), context)
      : validateBundleControlReviewInput(
        readJson(options.bundleReview), options.bundle, readJson(options.globalReview), context,
      );
    emit(report, null);
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
    const queue = queueAuthorityFromOptions(options, control);
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
  if (options.command === "initialize-queue") {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlFromOptions(options, baseline);
    emit(runInitialQueueCommand({ options, control }), options.output);
    return;
  }
  if (PILOT_COMMANDS.includes(options.command)) {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlFromOptions(options, baseline);
    emit(runPilotLifecycleCommand({ options, control, repository }), options.output);
    return;
  }
  if (options.command === "produce-gate") {
    const controlBaseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlFromOptions(options, controlBaseline);
    const leaseBase = parseInventoryEvidence(readJson(options.leaseBaseInventory));
    const lease = parseLease(readJson(options.lease), control, repository);
    assertLeaseBaseInventory(lease, control, leaseBase);
    const queue = queueAuthorityFromOptions(options, control);
    const subjectCompiledInventory = readJson(options.compiledInventory);
    const subject = buildInventory(repository, dispositionInputFromControl(control), {
      compiledInventory: subjectCompiledInventory,
    });
    emit(runGateProducer({ control, lease, queue_state: queue.state, queue_checkpoint: queue.checkpoint, control_baseline_inventory: controlBaseline, lease_base_inventory: leaseBase, subject_inventory: subject, subject_compiled_inventory: subjectCompiledInventory, bundle_id: options.bundle, gate: options.gate, artifact_id: options.artifact, inputs: options.inputs ? readJson(options.inputs) : null }), options.output);
    return;
  }
  if (options.command === "materialize-composition") {
    const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
    const control = parseControlFromOptions(options, baseline);
    const lease = parseLease(readJson(options.lease), control, repository);
    const leaseBase = parseInventoryEvidence(readJson(options.leaseBaseInventory));
    assertLeaseBaseInventory(lease, control, leaseBase);
    const queue = queueAuthorityFromOptions(options, control);
    emit(materializeEffectiveModuleComposition({
      repository, control, queueState: queue.state, queueCheckpoint: queue.checkpoint, lease,
    }, { productIds: options.authorityProducts ? null : options.productIds }), options.output);
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
        ? queueAuthorityFromOptions(options, control).state
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

function queueAuthorityFromOptions(options, control) {
  return loadQueueAuthorityFromCliPaths({
    statePath: options.state,
    checkpointPath: options.queueCheckpoint,
    trustedCheckpointDigest: options.trustedQueueCheckpointDigest,
    control,
  });
}

function runVerify(options) {
  const manifestPath = path.resolve(options.manifest);
  const baseline = parseInventoryEvidence(readJson(options.baselineInventory));
  const control = parseControlFromOptions(options, baseline);
  const manifest = parseVerificationManifest(readJson(manifestPath), control);
  const base = path.dirname(manifestPath);
  const load = (reference) => ({ reference, value: readJson(path.resolve(base, reference.path)) });
  const output = verifyBatch(
    manifest, load(manifest.audit), manifest.gate_results.map(load), control,
  );
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

function readJson(sourcePath) { return JSON.parse(fs.readFileSync(path.resolve(sourcePath), "utf8")); }
function emit(value, outputPath) {
  const encoded = `${JSON.stringify(value, null, 2)}\n`;
  if (outputPath) publishEvidenceBytes(canonicalEvidencePath(path.resolve(outputPath)), encoded);
  else process.stdout.write(encoded);
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
