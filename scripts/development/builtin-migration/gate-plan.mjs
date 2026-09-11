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
  "native-link": ["exit_status"],
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
  "deterministic-products": ["generated_products"],
  "inventory-delta": ["inventory_delta"],
});

export function parseGatePlans(value, bundleId, current = null) {
  const rows = array(value, `${bundleId} gate plans`, { empty: true }).map((entry) => parseGatePlan(entry, bundleId, current));
  const keys = rows.map((entry) => entry.gate);
  if (new Set(keys).size !== keys.length) throw new Error(`${bundleId}: gate plans must be unique by gate`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort())) throw new Error(`${bundleId}: gate plans must use canonical gate ordering`);
  return new Map(rows.map((entry) => [entry.gate, entry]));
}

export function gatePlanEvidence(plan, build, repository) {
  const executable = plan.program.approved_executables.find((entry) => entry.operating_system === build.operating_system && entry.architecture === build.architecture);
  if (!executable) throw new Error(`${plan.gate}: no reviewed executable for the evidence platform`);
  if (plan.program.kind === "repository_script") return {
    executable_digest: executable.content_digest, source_digest: plan.program.content_digest,
    arguments: [path.join(repository, plan.program.path), ...plan.arguments], cwd: repository,
  };
  return {
    executable_digest: executable.content_digest, source_digest: plan.program.manifest_digest,
    arguments: ["run", "--quiet", "-p", plan.program.package, "--bin", plan.program.binary, "--", ...plan.arguments], cwd: repository,
  };
}

function parseGatePlan(value, bundleId, current) {
  exact(value, ["gate", "program", "arguments", "working_directory", "parser", "expected_artifact_roles"], `${bundleId} gate plan`);
  const gate = enumValue(value.gate, Object.keys(GATE_PRODUCERS), `${bundleId} gate`);
  const parser = enumValue(value.parser, GATE_PARSERS, `${bundleId} gate parser`);
  if (!PARSERS_BY_GATE[gate].includes(parser)) throw new Error(`${bundleId}: ${parser} is not an approved parser for ${gate}`);
  if (value.working_directory !== "repository") throw new Error(`${bundleId}: gate working directory must be repository`);
  uniqueStrings(value.arguments, `${bundleId} gate arguments`, { empty: true });
  const roles = uniqueStrings(value.expected_artifact_roles, `${bundleId} gate artifact roles`, { empty: true });
  if (JSON.stringify(roles) !== JSON.stringify([...roles].sort())) throw new Error(`${bundleId}: gate artifact roles must use canonical ordering`);
  parseProgram(value.program, bundleId, current);
  return value;
}

function parseProgram(value, bundleId, current) {
  if (value?.kind === "repository_script") {
    exact(value, ["kind", "path", "content_digest", "approved_executables"], `${bundleId} repository script`);
    repositoryPath(value.path, `${bundleId} repository script path`);
    validateSourceDigest(value.path, value.content_digest, current, `${bundleId} repository script`);
  } else if (value?.kind === "cargo_binary") {
    exact(value, ["kind", "package", "binary", "manifest_path", "manifest_digest", "approved_executables"], `${bundleId} cargo binary`);
    nonempty(value.package, `${bundleId} cargo package`); nonempty(value.binary, `${bundleId} cargo binary`);
    repositoryPath(value.manifest_path, `${bundleId} Cargo manifest path`);
    validateSourceDigest(value.manifest_path, value.manifest_digest, current, `${bundleId} Cargo manifest`);
  } else throw new Error(`${bundleId}: unsupported gate program kind`);
  parseApprovedExecutables(value.approved_executables, bundleId, current);
}

function parseApprovedExecutables(value, bundleId, current) {
  const rows = array(value, `${bundleId} approved gate executables`);
  const keys = rows.map((entry) => {
    exact(entry, ["operating_system", "architecture", "content_digest"], `${bundleId} approved gate executable`);
    nonempty(entry.operating_system, `${bundleId} executable operating system`); nonempty(entry.architecture, `${bundleId} executable architecture`); digest(entry.content_digest, `${bundleId} executable digest`);
    return `${entry.operating_system}\0${entry.architecture}`;
  });
  if (new Set(keys).size !== keys.length) throw new Error(`${bundleId}: approved gate executables must be unique by platform`);
  if (JSON.stringify(keys) !== JSON.stringify([...keys].sort())) throw new Error(`${bundleId}: approved gate executables must use canonical platform ordering`);
  if (current) {
    const build = current.compiled_inventory.build;
    if (!rows.some((entry) => entry.operating_system === build.operating_system && entry.architecture === build.architecture)) throw new Error(`${bundleId}: gate plan has no approved executable for the compiled baseline platform`);
  }
}

function validateSourceDigest(sourcePath, expectedDigest, current, label) {
  digest(expectedDigest, `${label} digest`);
  if (!current) return;
  const observed = current.source.files.find((entry) => entry.path === sourcePath);
  if (!observed || observed.content_digest !== expectedDigest) throw new Error(`${label} is absent from or differs from the frozen source snapshot`);
}
