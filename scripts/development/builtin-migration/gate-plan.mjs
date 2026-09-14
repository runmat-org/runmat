import { GATE_PRODUCERS } from "./gate-kinds.mjs";
import path from "node:path";
import { array, digest, enumValue, exact, nonempty, repositoryPath, uniqueStrings } from "./schema.mjs";

export const GATE_PARSERS = Object.freeze([
  "compiled_inventory", "documentation_cutover", "example_reconciliation", "exit_status",
  "generated_products", "inventory_delta",
]);

const PARSERS_BY_GATE = Object.freeze({
  "catalog-contract": ["compiled_inventory"],
  "runtime-binding": ["compiled_inventory"],
  "documentation-cutover": ["documentation_cutover"],
  "native-link": ["compiled_inventory"],
  "wasm-registry": ["exit_status"],
  architecture: ["exit_status"],
  "focused-tests": ["exit_status"],
  "strict-clippy": ["exit_status"],
  "format-diff": ["exit_status"],
  "native-examples": ["example_reconciliation"],
  "browser-examples": ["example_reconciliation"],
  "provider-tests": ["exit_status"],
  "host-tests": ["exit_status"],
  "foreign-tests": ["exit_status"],
  "parallel-tests": ["exit_status"],
  "deterministic-products": ["generated_products"],
  "inventory-delta": ["inventory_delta"],
});

const ARTIFACT_ROLES_BY_PARSER = Object.freeze({
  compiled_inventory: ["compiled-inventory"],
  documentation_cutover: ["documentation-reconciliation"],
  example_reconciliation: ["example-reconciliation"],
  exit_status: [],
  generated_products: ["generated-products"],
  inventory_delta: ["inventory-delta"],
});

export function parseGatePlans(value, bundleId, sourceInventory = null) {
  const rows = array(value, `${bundleId} gate plans`, { empty: true })
    .map((entry) => parseGatePlan(entry, bundleId, sourceInventory));
  const keys = rows.map((entry) => entry.gate);
  if (new Set(keys).size !== keys.length) throw new Error(`${bundleId}: gate plans must be unique by gate`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort())) throw new Error(`${bundleId}: gate plans must use canonical gate ordering`);
  return new Map(rows.map((entry) => [entry.gate, entry]));
}

export function gatePlanEvidence(plan, build, repository) {
  const tools = approvedTools(plan.program, build, plan.gate);
  if (plan.program.kind === "repository_script") return {
    primary_tool: "node", tools, source_digest: plan.program.content_digest,
    arguments: [path.join(repository, plan.program.path), ...plan.arguments], cwd: repository,
  };
  if (plan.program.kind === "cargo_operation") return {
    primary_tool: "cargo", tools, source_digest: plan.program.manifest_digest,
    arguments: [plan.program.operation, "--manifest-path", path.join(repository, plan.program.manifest_path), ...plan.arguments], cwd: repository,
  };
  return {
    primary_tool: "cargo", tools, source_digest: plan.program.manifest_digest,
    arguments: ["run", "--manifest-path", path.join(repository, plan.program.manifest_path), "--quiet", "-p", plan.program.package, "--bin", plan.program.binary, "--", ...plan.arguments], cwd: repository,
  };
}

export function validateGatePlanTargetCoverage(plans, executionTargets, bundleId) {
  const expected = executionTargets
    .map((entry) => `${entry.operating_system}\0${entry.architecture}`)
    .sort();
  for (const plan of plans.values()) {
    const observed = plan.program.approved_toolchains
      .map((entry) => `${entry.operating_system}\0${entry.architecture}`)
      .sort();
    if (JSON.stringify(observed) !== JSON.stringify(expected)) {
      throw new Error(`${bundleId}/${plan.gate}: reviewed toolchains must exactly cover every execution target`);
    }
  }
}

function parseGatePlan(value, bundleId, sourceInventory) {
  exact(value, ["gate", "program", "arguments", "working_directory", "parser", "expected_artifact_roles"], `${bundleId} gate plan`);
  const gate = enumValue(value.gate, Object.keys(GATE_PRODUCERS), `${bundleId} gate`);
  const parser = enumValue(value.parser, GATE_PARSERS, `${bundleId} gate parser`);
  if (!PARSERS_BY_GATE[gate].includes(parser)) throw new Error(`${bundleId}: ${parser} is not an approved parser for ${gate}`);
  if (value.working_directory !== "repository") throw new Error(`${bundleId}: gate working directory must be repository`);
  const argumentsList = array(value.arguments, `${bundleId} gate arguments`, { empty: true });
  argumentsList.forEach((entry) => {
    nonempty(entry, `${bundleId} gate argument`);
    if (entry.includes("\0")) throw new Error(`${bundleId}: gate arguments cannot contain NUL bytes`);
  });
  const roles = uniqueStrings(value.expected_artifact_roles, `${bundleId} gate artifact roles`, { empty: true });
  if (JSON.stringify(roles) !== JSON.stringify([...roles].sort())) throw new Error(`${bundleId}: gate artifact roles must use canonical ordering`);
  if (JSON.stringify(roles) !== JSON.stringify(ARTIFACT_ROLES_BY_PARSER[parser])) throw new Error(`${bundleId}: ${parser} must emit its exact typed artifact role set`);
  parseGateProgram(value.program, bundleId, sourceInventory);
  if (value.program.kind !== "repository_script") validateCargoArguments(argumentsList, bundleId);
  if (parser === "documentation_cutover") requireRepositoryProducer(
    value, bundleId, "documentation cutover",
    "scripts/development/builtin-migration/documentation-export-cli.mjs",
    ["cargo", "git", "node", "rustc"],
  );
  if (parser === "example_reconciliation") requireRepositoryProducer(
    value, bundleId, "example reconciliation",
    "scripts/development/builtin-migration/example-gate-cli.mjs",
    ["node"],
  );
  if (parser === "generated_products") requireProgramTools(value.program, ["cargo", "rustc"], bundleId, "generated product verification");
  return value;
}

export function parseGateProgram(value, bundleId, sourceInventory = null) {
  if (value?.kind === "repository_script") {
    exact(value, ["kind", "path", "content_digest", "approved_toolchains"], `${bundleId} repository script`);
    repositoryPath(value.path, `${bundleId} repository script path`);
    validateSourceDigest(value.path, value.content_digest, sourceInventory, `${bundleId} repository script`);
    parseApprovedToolchains(value.approved_toolchains, bundleId, ["node"], [
      "cargo", "cargo-clippy", "cargo-fmt", "clippy-driver", "git", "node", "rustc", "rustdoc", "rustfmt",
    ]);
    return value;
  } else if (value?.kind === "cargo_binary") {
    exact(value, ["kind", "package", "binary", "manifest_path", "manifest_digest", "approved_toolchains"], `${bundleId} cargo binary`);
    cargoName(value.package, `${bundleId} cargo package`); cargoName(value.binary, `${bundleId} cargo binary`);
    repositoryPath(value.manifest_path, `${bundleId} Cargo manifest path`);
    validateSourceDigest(value.manifest_path, value.manifest_digest, sourceInventory, `${bundleId} Cargo manifest`);
    parseApprovedToolchains(value.approved_toolchains, bundleId, requiredToolRoles("run"));
    return value;
  } else if (value?.kind === "cargo_operation") {
    exact(value, ["kind", "operation", "manifest_path", "manifest_digest", "approved_toolchains"], `${bundleId} cargo operation`);
    const operation = enumValue(value.operation, ["check", "clippy", "fmt", "test"], `${bundleId} Cargo operation`);
    repositoryPath(value.manifest_path, `${bundleId} Cargo manifest path`);
    validateSourceDigest(value.manifest_path, value.manifest_digest, sourceInventory, `${bundleId} Cargo manifest`);
    parseApprovedToolchains(value.approved_toolchains, bundleId, requiredToolRoles(operation));
    return value;
  } else throw new Error(`${bundleId}: unsupported gate program kind`);
}

function parseApprovedToolchains(value, bundleId, requiredRoles, allowedRoles = requiredRoles) {
  const rows = array(value, `${bundleId} approved gate toolchains`);
  const keys = rows.map((entry) => {
    exact(entry, ["operating_system", "architecture", "tools"], `${bundleId} approved gate toolchain`);
    const key = `${nonempty(entry.operating_system, `${bundleId} toolchain operating system`)}\0${nonempty(entry.architecture, `${bundleId} toolchain architecture`)}`;
    const roles = array(entry.tools, `${bundleId} approved toolchain tools`).map((tool) => {
      exact(tool, ["role", "content_digest"], `${bundleId} approved tool`);
      const role = enumValue(tool.role, ["cargo", "cargo-clippy", "cargo-fmt", "clippy-driver", "git", "node", "rustc", "rustdoc", "rustfmt"], `${bundleId} approved tool role`);
      digest(tool.content_digest, `${bundleId} ${role} digest`);
      return role;
    });
    if (new Set(roles).size !== roles.length || JSON.stringify(roles) !== JSON.stringify([...roles].sort())) {
      throw new Error(`${bundleId}: approved tool roles must be unique and canonical`);
    }
    if (roles.some((role) => !allowedRoles.includes(role)) || requiredRoles.some((role) => !roles.includes(role))) {
      throw new Error(`${bundleId}: approved toolchain does not contain its exact allowed and required tools`);
    }
    return key;
  });
  if (new Set(keys).size !== keys.length) throw new Error(`${bundleId}: approved gate toolchains must be unique by platform`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort())) throw new Error(`${bundleId}: approved gate toolchains must use canonical platform ordering`);
}

function approvedTools(program, build, gate) {
  const toolchain = program.approved_toolchains.find((entry) =>
    entry.operating_system === build.operating_system && entry.architecture === build.architecture);
  if (!toolchain) throw new Error(`${gate}: no reviewed toolchain for the evidence platform`);
  return toolchain.tools;
}

function requireProgramTools(program, requiredRoles, bundleId, label) {
  for (const toolchain of program.approved_toolchains) {
    const roles = toolchain.tools.map((tool) => tool.role);
    if (requiredRoles.some((role) => !roles.includes(role))) {
      throw new Error(`${bundleId}: ${label} requires reviewed ${requiredRoles.join(" and ")} tools on every execution target`);
    }
  }
}

function requireRepositoryProducer(plan, bundleId, label, sourcePath, toolRoles) {
  if (plan.program.kind !== "repository_script" || plan.program.path !== sourcePath) {
    throw new Error(`${bundleId}: ${label} must use its reviewed repository producer`);
  }
  if (plan.arguments.length !== 0) throw new Error(`${bundleId}: ${label} producer accepts no arguments`);
  for (const toolchain of plan.program.approved_toolchains) {
    const observed = toolchain.tools.map((tool) => tool.role);
    if (JSON.stringify(observed) !== JSON.stringify(toolRoles)) {
      throw new Error(`${bundleId}: ${label} requires exact reviewed ${toolRoles.join(", ")} tools`);
    }
  }
}

function cargoName(value, label) {
  const result = nonempty(value, label);
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(result)) throw new Error(`${label} is not a valid Cargo target name`);
  return result;
}

function validateCargoArguments(argumentsList, bundleId) {
  const forbidden = ["--config", "--manifest-path", "--target-dir"];
  for (const argument of argumentsList) {
    if (forbidden.some((flag) => argument === flag || argument.startsWith(`${flag}=`))
      || argument === "-Z" || argument.startsWith("-Z")) {
      throw new Error(`${bundleId}: Cargo gate arguments cannot override manifest, storage, or configuration authority`);
    }
  }
}

function requiredToolRoles(operation) {
  if (operation === "clippy") return ["cargo", "cargo-clippy", "clippy-driver", "rustc"];
  if (operation === "fmt") return ["cargo", "cargo-fmt", "rustfmt"];
  if (operation === "test") return ["cargo", "rustc", "rustdoc"];
  return ["cargo", "rustc"];
}

function validateSourceDigest(sourcePath, expectedDigest, sourceInventory, label) {
  digest(expectedDigest, `${label} digest`);
  if (!sourceInventory) return;
  const observed = sourceInventory.source.files.find((entry) => entry.path === sourcePath);
  if (!observed || observed.content_digest !== expectedDigest) throw new Error(`${label} is absent from or differs from the frozen source snapshot`);
}
