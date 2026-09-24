import {
  PILOT_COMMANDS, PILOT_OPTION_FIELDS, validatePilotCommandOptions,
} from "./pilot-contract.mjs";

export const COMMANDS = Object.freeze([
  "inventory", "queue", "seed-dispositions", "compile-dispositions", "draft-control",
  "component-graph", "compose-topology", "freeze-topology", "validate-topology",
  "scaffold-control", "init-control-reviews", "validate-global-control-review",
  "validate-bundle-control-review", "index-control-reviews", "compose-control",
  "scaffold-control-attestation", "seal-control-attestation", "freeze-control",
  "validate-control", "issue-lease", "produce-gate", "materialize-composition",
  "initialize-queue",
  "prepare", "audit", "verify", "seal", ...PILOT_COMMANDS,
]);

export const OPTION_FIELDS = Object.freeze({
  "--output": "output",
  "--authority-root": "authorityRoot",
  "--compiled-inventory": "compiledInventory",
  "--baseline-inventory": "baselineInventory",
  "--lease-base-inventory": "leaseBaseInventory",
  "--dispositions": "dispositions",
  "--control": "control",
  "--draft": "draft",
  "--review": "review",
  "--request": "request",
  "--lease": "lease",
  "--state": "state",
  "--queue-checkpoint": "queueCheckpoint",
  "--trusted-queue-checkpoint-digest": "trustedQueueCheckpointDigest",
  "--batch": "batch",
  "--evidence": "evidence",
  "--workspace": "workspace",
  "--manifest": "manifest",
  "--bundle": "bundle",
  "--gate": "gate",
  "--artifact": "artifact",
  "--inputs": "inputs",
  "--component-graph": "componentGraph",
  "--c01-c03-review": "c01C03Review",
  "--c04-c05-review": "c04C05Review",
  "--c06-c07-review": "c06C07Review",
  "--reconciliation": "reconciliation",
  "--stability-corrections": "stabilityCorrections",
  "--candidate": "candidate",
  "--attestation": "attestation",
  "--topology": "topology",
  "--control-scaffold": "controlScaffold",
  "--control-review-set": "controlReviewSet",
  "--control-candidate": "controlCandidate",
  "--control-attestation": "controlAttestation",
  "--initial-queue-review": "initialQueueReview",
  "--initial-queue-review-digest": "initialQueueReviewDigest",
  "--review-directory": "reviewDirectory",
  "--review-set-directory": "reviewSetDirectory",
  "--attestation-review": "attestationReview",
  "--global-review": "globalReview",
  "--bundle-review": "bundleReview",
  ...PILOT_OPTION_FIELDS,
});

export const REPEATABLE_OPTION_FIELDS = Object.freeze({
  "--product": "productIds",
});

export const FLAG_FIELDS = Object.freeze({
  "--no-products": "noProducts",
  "--authority-products": "authorityProducts",
});

const CONTROL_COMMANDS = Object.freeze([
  "queue", "prepare", "audit", "freeze-control", "validate-control", "issue-lease",
  "initialize-queue",
  "produce-gate", "materialize-composition", "verify", "seal", ...PILOT_COMMANDS,
]);
const CONTROL_AUTHORING_COMMANDS = Object.freeze([
  "scaffold-control", "init-control-reviews", "validate-global-control-review",
  "validate-bundle-control-review", "index-control-reviews", "compose-control",
  "scaffold-control-attestation", "seal-control-attestation",
]);
const REVIEW_VALIDATION_COMMANDS = Object.freeze([
  "validate-global-control-review", "validate-bundle-control-review",
]);
const TOPOLOGY_OPTIONS = Object.freeze([
  "--baseline-inventory", "--topology", "--candidate", "--attestation",
  "--component-graph", "--draft", "--c01-c03-review", "--c04-c05-review",
  "--c06-c07-review", "--reconciliation", "--stability-corrections",
]);
const COMMAND_OWNED_OPTIONS = Object.freeze(new Map([
  ["--initial-queue-review", new Set(["initialize-queue"])],
  ["--initial-queue-review-digest", new Set(["initialize-queue"])],
]));

export function newCommandOptions(command) {
  return {
    command,
    ...Object.fromEntries(Object.values(OPTION_FIELDS).map((field) => [field, null])),
    identity: null,
    productIds: [],
    noProducts: false,
    authorityProducts: false,
    help: false,
  };
}

export function validateCommandOptions(options, suppliedOptions) {
  const { command } = options;
  assertCommandOwnedOptions(command, suppliedOptions);
  assertInitialQueueOptions(command, suppliedOptions);
  assertReviewValidationOptions(command, suppliedOptions);
  const requiresCompiled = [
    "inventory", "queue", "seed-dispositions", "prepare", "audit", "produce-gate",
  ].includes(command);
  if (requiresCompiled && !options.compiledInventory) throw new Error(`${command} requires --compiled-inventory`);
  if (["queue", "prepare", "audit", "validate-control", "initialize-queue", "produce-gate", "verify"].includes(command) && !options.control) throw new Error(`${command} requires --control`);
  if (["queue", "prepare", "audit", "compile-dispositions", "draft-control", "component-graph", "compose-topology", "freeze-topology", "validate-topology", ...CONTROL_AUTHORING_COMMANDS, ...CONTROL_COMMANDS].includes(command) && !options.baselineInventory) throw new Error(`${command} requires --baseline-inventory`);
  if (["compose-topology", "freeze-topology", "validate-topology", ...CONTROL_AUTHORING_COMMANDS, ...CONTROL_COMMANDS].includes(command) && (!options.componentGraph || !options.draft || !options.c01C03Review || !options.c04C05Review || !options.c06C07Review || !options.reconciliation || !options.stabilityCorrections)) {
    throw new Error(`${command} requires --component-graph, --draft, all three cohort reviews, --reconciliation, and --stability-corrections`);
  }
  if (["freeze-topology", "validate-topology", ...CONTROL_AUTHORING_COMMANDS, ...CONTROL_COMMANDS].includes(command) && (!options.candidate || !options.attestation)) throw new Error(`${command} requires --candidate and --attestation`);
  if (["validate-topology", ...CONTROL_AUTHORING_COMMANDS, ...CONTROL_COMMANDS].includes(command) && !options.topology) throw new Error(`${command} requires --topology`);
  if (command === "compile-dispositions" && !options.review) throw new Error("compile-dispositions requires --review");
  if (["init-control-reviews", "validate-global-control-review", "validate-bundle-control-review", "index-control-reviews", "compose-control", "scaffold-control-attestation", "seal-control-attestation", ...CONTROL_COMMANDS].includes(command) && !options.controlScaffold) throw new Error(`${command} requires --control-scaffold`);
  if (REVIEW_VALIDATION_COMMANDS.includes(command) && !options.globalReview) throw new Error(`${command} requires --global-review`);
  if (command === "validate-bundle-control-review" && (!options.bundleReview || !options.bundle)) throw new Error("validate-bundle-control-review requires --bundle-review and --bundle");
  if (["compose-control", "scaffold-control-attestation", "seal-control-attestation", ...CONTROL_COMMANDS].includes(command) && !options.controlReviewSet) throw new Error(`${command} requires --control-review-set`);
  if (["scaffold-control-attestation", "seal-control-attestation", ...CONTROL_COMMANDS].includes(command) && !options.controlCandidate) throw new Error(`${command} requires --control-candidate`);
  if (CONTROL_COMMANDS.includes(command) && (!options.controlCandidate || !options.controlAttestation)) throw new Error(`${command} requires --control-candidate and --control-attestation`);
  if (["init-control-reviews", "index-control-reviews"].includes(command) && !options.reviewDirectory) throw new Error(`${command} requires --review-directory`);
  if (command === "index-control-reviews" && !options.reviewSetDirectory) throw new Error("index-control-reviews requires --review-set-directory");
  if (command === "seal-control-attestation" && !options.attestationReview) throw new Error("seal-control-attestation requires --attestation-review");
  validateQueueAndLeaseOptions(options);
  validateMaterializationSelection(options);
  validatePilotCommandOptions(options);
  if (["prepare", "audit"].includes(command) && !options.lease) throw new Error(`${command} requires --lease`);
  if (["prepare", "audit", "produce-gate"].includes(command) && !options.leaseBaseInventory) throw new Error(`${command} requires --lease-base-inventory`);
  if (command === "produce-gate" && !options.lease) throw new Error("produce-gate requires --lease");
  if (command === "prepare" && !options.workspace) throw new Error("prepare requires --workspace outside the repository");
  if (command === "audit" && (!options.batch || !options.evidence)) throw new Error("audit requires --batch and --evidence");
  if (["verify", "seal"].includes(command) && !options.manifest) throw new Error(`${command} requires --manifest`);
  if (command === "seal" && (!options.control || !options.lease)) throw new Error("seal requires --control and --lease");
}

function validateMaterializationSelection(options) {
  const selected = options.productIds.length > 0;
  const modes = Number(selected) + Number(options.noProducts)
    + Number(options.authorityProducts);
  if (options.command !== "materialize-composition") {
    if (modes > 0) {
      throw new Error("--product, --no-products, and --authority-products are accepted only by materialize-composition");
    }
    return;
  }
  if (modes !== 1) {
    throw new Error("materialize-composition requires exactly one of --product, --no-products, or --authority-products");
  }
  if (new Set(options.productIds).size !== options.productIds.length) {
    throw new Error("materialize-composition does not accept duplicate --product values");
  }
}

function assertCommandOwnedOptions(command, suppliedOptions) {
  for (const option of suppliedOptions) {
    const owners = COMMAND_OWNED_OPTIONS.get(option);
    if (owners && !owners.has(command)) {
      throw new Error(`${option} is not accepted by ${command}`);
    }
  }
}

function validateQueueAndLeaseOptions(options) {
  const { command } = options;
  if (command === "issue-lease" && (!options.control || !options.request || !options.leaseBaseInventory || !options.state || !options.queueCheckpoint || !options.trustedQueueCheckpointDigest)) throw new Error("issue-lease requires --control, --request, --lease-base-inventory, --state, --queue-checkpoint, and --trusted-queue-checkpoint-digest");
  if (command === "initialize-queue" && (!options.authorityRoot || !options.initialQueueReview || !options.initialQueueReviewDigest)) throw new Error("initialize-queue requires --authority-root, --initial-queue-review, and --initial-queue-review-digest");
  if (command === "queue" && options.state && (!options.queueCheckpoint || !options.trustedQueueCheckpointDigest)) throw new Error("queue with --state requires --queue-checkpoint and --trusted-queue-checkpoint-digest");
  if (command === "produce-gate" && (!options.bundle || !options.gate || !options.artifact || !options.state || !options.queueCheckpoint || !options.trustedQueueCheckpointDigest)) throw new Error("produce-gate requires --bundle, --gate, --artifact, --state, --queue-checkpoint, and --trusted-queue-checkpoint-digest");
  if (command === "materialize-composition" && (!options.control || !options.lease || !options.leaseBaseInventory || !options.state || !options.queueCheckpoint || !options.trustedQueueCheckpointDigest)) throw new Error("materialize-composition requires --control, --lease-base-inventory, --lease, --state, --queue-checkpoint, and --trusted-queue-checkpoint-digest");
}

function assertInitialQueueOptions(command, suppliedOptions) {
  if (command !== "initialize-queue") return;
  const allowed = new Set([
    ...TOPOLOGY_OPTIONS,
    "--control", "--control-scaffold", "--control-review-set",
    "--control-candidate", "--control-attestation", "--authority-root",
    "--initial-queue-review", "--initial-queue-review-digest", "--output",
  ]);
  for (const option of suppliedOptions) {
    if (!allowed.has(option)) throw new Error(`${command} does not accept ${option}`);
  }
}

function assertReviewValidationOptions(command, suppliedOptions) {
  if (!REVIEW_VALIDATION_COMMANDS.includes(command)) return;
  const common = [...TOPOLOGY_OPTIONS, "--control-scaffold", "--global-review"];
  const allowed = new Set(command === "validate-global-control-review"
    ? common
    : [...common, "--bundle-review", "--bundle"]);
  for (const option of suppliedOptions) {
    if (option === "--output") throw new Error(`${command} writes deterministic validation output only to stdout; --output is not accepted`);
    if (!allowed.has(option)) throw new Error(`${command} does not accept ${option}`);
  }
}
