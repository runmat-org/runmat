#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../..");
const outputPath = path.join(repo, "target/rm1064-inventory/report.json");
const summaryPath = path.join(repo, "docs/development/rm1064-inventory.md");
const mode = process.argv[2] ?? "--check";

if (!["--check", "--write", "--stdout"].includes(mode)) {
  console.error("usage: rm1064-inventory.mjs [--check|--write|--stdout]");
  process.exit(2);
}

function walk(directory) {
  const files = [];
  for (const item of fs.readdirSync(directory, { withFileTypes: true })) {
    if (["target", ".git", "node_modules"].includes(item.name)) continue;
    const absolute = path.join(directory, item.name);
    if (item.isDirectory()) files.push(...walk(absolute));
    else if (item.isFile() && item.name.endsWith(".rs")) files.push(absolute);
  }
  return files.sort();
}

const files = walk(path.join(repo, "crates"));
const entries = [];
const providerMethods = new Set();

function relative(file) {
  return path.relative(repo, file).split(path.sep).join("/");
}

function lineAt(text, offset) {
  return text.slice(0, offset).split("\n").length;
}

function cleanIdentity(value) {
  return value.replace(/\s+/g, " ").trim();
}

function enclosingFunction(text, offset) {
  const prefix = text.slice(0, offset);
  const functions = /\bfn\s+([A-Za-z0-9_]+)\s*\(/g;
  let symbol = "module";
  for (let match; (match = functions.exec(prefix)); ) symbol = match[1];
  return symbol;
}

function cohortFor(file, identity = "", category = "") {
  const value = `${file} ${identity} ${category}`.toLowerCase();
  if (/plot|figure|graphics|\bui\b|timing|workspace|diagnostic|introspection|console/.test(value)) return "G";
  if (/\bio\b|image|geometry|meshing|fea|filesystem|archive|http|network|package|foreign|mex|ffi|java/.test(value)) return "F";
  if (/stats|signal|control|optim|finance|comms|deep_learning|machine_learning/.test(value)) return "E";
  if (/string|cell|struct|symbolic|container|table|categorical|datetime|duration|object|class/.test(value)) return "D";
  if (/index|reduction|linalg|sorting|set|matrix|sparse|fft|polynomial/.test(value)) return "C";
  if (/creation|random|shape|reshape|cat|grid|range|array/.test(value)) return "B";
  return "A";
}

const policies = {
  catalog_builtin: ["static", "R06/C00-C07", "The canonical runtime-independent catalog retains this identity exactly once and every executable binding joins it by stable identity."],
  runtime_builtin: ["runtime", "C00-C07", "Migrate the identity atomically to the canonical catalog/runtime binding; remove this legacy registration authority."],
  builtin_descriptor: ["static", "R06/C00-C07", "Canonical catalog owns the descriptor; no runtime-owned duplicate remains."],
  type_resolver: ["static", "R05-R08/C00-C07", "Shared facts/call contracts replace the resolver and the old resolver attachment is removed."],
  semantic_fallback: ["static", "R04-R08", "Classify the fallback explicitly in shared facts or remove it; no unclassified static fallback remains."],
  gpu_spec: ["placement", "R15-R20/C00-C07", "Canonical placement/provider contract owns this identity; no parallel GPU registry remains."],
  fusion_spec: ["placement", "R15-R20/C00-C07", "Canonical region/placement contract owns fusion semantics; no parallel fusion registry remains."],
  residency_policy: ["placement", "R15-R20/C00-C07", "Residency is expressed once in the placement contract and this legacy attachment is removed."],
  sink_policy: ["static", "R06/C00-C07", "Effects/output behavior live in the canonical contract; no independent sink flag remains."],
  vm_private_builtin: ["vm", "R10/C00-C07", "Shared runtime semantics own the behavior; VM retains only executor dispatch with no name-specific semantic authority."],
  turbine_host_bridge: ["native", "R12-R14/R30", "The production native ABI/runtime helper replaces this Turbine bridge and Turbine is removed."],
  lsp_shape_rule: ["static", "R07-R08", "LSP consumes shared program-point facts and contains no independent shape rule."],
  provider_method: ["placement", "R15-R17", "Retain only as a versioned provider capability contracted through the unified planner."],
  provider_direct_call: ["placement", "R15-R17/C00-C07", "Route through placement feasibility/dispatch or explicitly retain below policy; no unclassified bypass remains."],
  parallel_surface: ["parallel", "R26-R28", "Canonical parallel/distributed schema and runtime own this surface once."],
  package_surface: ["package", "R20/R28", "Retain under the resolved package graph and executable-product contracts with one stable identity."],
  interop_surface: ["interop", "R21-R25A", "Canonical foreign/extension ABI owns this surface; prototypes and duplicates are removed."],
  runtime_ambient: ["runtime", "R09/R10/R29", "Session semantics route through the explicit RuntimeContext; only the classified fallback adapter remains until its assigned removal slice."],
  relevant_test: ["test", "owning slice", "Retain or replace with parity/conformance coverage against the final authority; no test targets removed legacy behavior."],
};

const runtimeAmbientAuthorities = [
  ["crates/runmat-runtime/src/source_context.rs", "CURRENT_SOURCE|SOURCE_CATALOG", "R29"],
  ["crates/runmat-runtime/src/callsite.rs", "CALLSITE_STACK|FUNCTION_INPUT_CALLSITE_STACK", "R29"],
  ["crates/runmat-runtime/src/debug_context.rs", "DEBUG_STACK", "R29"],
  ["crates/runmat-runtime/src/output_context.rs", "REQUESTED_OUTPUTS", "R29"],
  ["crates/runmat-runtime/src/output_count.rs", "OUTPUT_COUNT_STACK", "R29"],
  ["crates/runmat-runtime/src/interrupt.rs", "INTERRUPT_HANDLE", "R29"],
  ["crates/runmat-runtime/src/warning_store.rs", "WARNINGS", "R29"],
  ["crates/runmat-runtime/src/compatibility.rs", "RUNMAT_EXTENSIONS_ENABLED", "R29"],
  ["crates/runmat-runtime/src/console.rs", "FALLBACK_STATE|FORWARDER", "R29"],
  ["crates/runmat-runtime/src/interaction.rs", "QUEUED_RESPONSE|ASYNC_HANDLER|EVAL_HOOK", "R29"],
  ["crates/runmat-runtime/src/workspace.rs", "RESOLVER", "R10/R29"],
  ["crates/runmat-runtime/src/class_registry.rs", "FALLBACK_STATE|CONTEXT_STATES", "R29"],
  ["crates/runmat-runtime/src/builtins/common/path_state.rs", "PATH_STATE", "R29"],
  ["crates/runmat-runtime/src/user_functions.rs", "SEMANTIC_FUNCTION_INVOKER|SEMANTIC_FUNCTION_RESOLVER|SOURCE_FUNCTION_CATALOG|ACTIVE_SEMANTIC_FUNCTION_STACK", "R10/R29"],
  ["crates/runmat-runtime/src/testing/context.rs", "TEST_CONTEXT_STACK", "R29"],
  ["crates/runmat-runtime/src/testing/services.rs", "TEST_SERVICES_STACK", "R29"],
  ["crates/runmat-runtime/src/dispatcher.rs", "CLASS_ACCESS_CONTEXT", "R10/R29"],
  ["crates/runmat-runtime/src/lib.rs", "CONSTRUCTOR_RECEIVER_STACK|EVENT_REGISTRY", "R10/R29"],
  ["crates/runmat-runtime/src/context/scope.rs", "ACTIVE_CONTEXTS", "R29"],
  ["crates/runmat-vm/src/runtime/workspace.rs", "WORKSPACE_STACK|PENDING_WORKSPACE|LAST_WORKSPACE_STATE|LAST_WORKSPACE_ASSIGNED_REPORT", "R10"],
  ["crates/runmat-vm/src/runtime/globals.rs", "GLOBALS|PERSISTENTS|PERSISTENTS_BY_NAME", "R10"],
  ["crates/runmat-vm/src/runtime/call_stack.rs", "CALL_STACK|CALL_STACK_LIMIT|ERROR_NAMESPACE", "R10/R29"],
  ["crates/runmat-vm/src/interpreter/runner.rs", "CALL_COUNTS", "R10"],
  ["crates/runmat-vm/src/interpreter/errors.rs", "CURRENT_PC", "R10"],
  ["crates/runmat-vm/src/call/builtins.rs", "DYNAMIC_EVAL_OPTIONS", "R10/R29"],
  ["crates/runmat-vm/src/coverage.rs", "ACTIVE", "R29"],
];
const allowedLegacyContextConsumers = new Set(
  runtimeAmbientAuthorities.map(([source]) => source),
);
allowedLegacyContextConsumers.add("crates/runmat-runtime/src/runtime_error.rs");

const runtimeIndexingModules = [
  "end_expr.rs",
  "integer_assignment.rs",
  "plan.rs",
  "read_linear.rs",
  "read_slice.rs",
  "selectors.rs",
  "write_linear.rs",
  "write_slice.rs",
];
for (const module of runtimeIndexingModules) {
  const owner = path.join(repo, "crates/runmat-runtime/src/indexing", module);
  if (!fs.existsSync(owner)) throw new Error(`missing R10 runtime indexing owner ${relative(owner)}`);
  const legacy = path.join(repo, "crates/runmat-vm/src/indexing", module);
  if (fs.existsSync(legacy)) throw new Error(`legacy VM indexing authority remains at ${relative(legacy)}`);
}
const endExprOwners = files.filter((file) =>
  /\bpub\s+enum\s+EndExpr\b/.test(fs.readFileSync(file, "utf8")),
);
if (endExprOwners.length !== 1 || relative(endExprOwners[0]) !== "crates/runmat-runtime/src/indexing/end_expr.rs") {
  throw new Error(`EndExpr must have exactly one runtime owner; found ${endExprOwners.map(relative).join(", ")}`);
}

const runtimeCellOwner = path.join(repo, "crates/runmat-runtime/src/object/cell.rs");
if (!fs.existsSync(runtimeCellOwner)) {
  throw new Error(`missing R10 runtime cell-semantics owner ${relative(runtimeCellOwner)}`);
}
const legacyVmCellOwner = path.join(repo, "crates/runmat-vm/src/ops/cells.rs");
if (fs.existsSync(legacyVmCellOwner)) {
  throw new Error(`legacy VM cell-semantics authority remains at ${relative(legacyVmCellOwner)}`);
}

const argumentSpecOwners = files.filter((file) =>
  /\bpub\s+struct\s+ArgumentSpec\b/.test(fs.readFileSync(file, "utf8")),
);
if (
  argumentSpecOwners.length !== 1 ||
  relative(argumentSpecOwners[0]) !== "crates/runmat-runtime/src/call/arguments.rs"
) {
  throw new Error(
    `ArgumentSpec must have exactly one runtime owner; found ${argumentSpecOwners.map(relative).join(", ")}`,
  );
}

for (const [runtimeOwner, legacyOwner, label] of [
  ["crates/runmat-runtime/src/call/descriptor.rs", "crates/runmat-vm/src/call/descriptor.rs", "callable descriptor"],
  ["crates/runmat-runtime/src/call/identity.rs", null, "callable identity operations"],
  ["crates/runmat-runtime/src/object/indexing.rs", null, "object indexing protocol"],
]) {
  const owner = path.join(repo, runtimeOwner);
  if (!fs.existsSync(owner)) throw new Error(`missing R10 runtime ${label} owner ${relative(owner)}`);
  if (legacyOwner) {
    const legacy = path.join(repo, legacyOwner);
    if (fs.existsSync(legacy)) throw new Error(`legacy VM ${label} authority remains at ${relative(legacy)}`);
  }
}

for (const [source, identities, targetSlice] of runtimeAmbientAuthorities) {
  const file = path.join(repo, source);
  const text = fs.readFileSync(file, "utf8");
  for (const identity of identities.split("|")) {
    const marker = new RegExp(`\\bstatic\\s+${identity}\\b`);
    const match = marker.exec(text);
    if (!match) throw new Error(`missing classified runtime ambient authority ${identity} in ${source}`);
    add("runtime_ambient", identity, file, text, match.index, {
      migration: {
        target_slice: targetSlice,
        zero_state: `Explicit RuntimeContext state/services own ${identity}; the fallback declaration and every direct consumer are removed by ${targetSlice}.`,
      },
    });
  }
}

function add(kind, identity, file, text, offset, detail = {}) {
  const [owner, targetSlice, zeroState] = policies[kind];
  const source = relative(file);
  const symbol = cleanIdentity(identity);
  const baseId = `${kind}:${source}:${symbol}`;
  let id = baseId;
  let suffix = 2;
  while (entries.some((entry) => entry.id === id)) id = `${baseId}#${suffix++}`;
  entries.push({
    id,
    kind,
    identity: symbol,
    cohort: cohortFor(source, symbol, detail.category ?? ""),
    owner,
    source: { path: source, line: lineAt(text, offset) },
    migration: { target_slice: targetSlice, zero_state: zeroState },
    ...detail,
  });
}

function findBalanced(text, start, open = "(", close = ")") {
  let depth = 0;
  let string = false;
  let escaped = false;
  for (let i = start; i < text.length; i += 1) {
    const char = text[i];
    if (string) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') string = false;
      continue;
    }
    if (char === '"') string = true;
    else if (char === open) depth += 1;
    else if (char === close && --depth === 0) return i;
  }
  return -1;
}

const catalogPath = path.join(repo, "docs/builtins/meta.json");
const catalogText = fs.readFileSync(catalogPath, "utf8");
const catalog = JSON.parse(catalogText);
let catalogOffset = 0;
for (const builtin of catalog.builtins) {
  const needle = `"name": "${builtin.name.replaceAll('"', '\\"')}"`;
  const offset = catalogText.indexOf(needle, catalogOffset);
  if (offset < 0) throw new Error(`cannot locate catalog builtin ${builtin.name}`);
  add("catalog_builtin", builtin.name, catalogPath, catalogText, offset, { builtin: builtin.name, category: builtin.category ?? "" });
  catalogOffset = offset + needle.length;
}

function stringField(body, name) {
  return body.match(new RegExp(`\\b${name}\\s*=\\s*"([^"]+)"`))?.[1] ?? null;
}

for (const file of files) {
  const source = relative(file);
  const text = fs.readFileSync(file, "utf8");

  const legacyContextAccess = /context::legacy::(?:active|error_namespace)\s*\(/g;
  for (let match; (match = legacyContextAccess.exec(text)); ) {
    if (!allowedLegacyContextConsumers.has(source)) {
      throw new Error(
        `unapproved direct runtime-context bridge consumer in ${source}:${lineAt(text, match.index)}`,
      );
    }
    add("runtime_ambient", `legacy-adapter/${enclosingFunction(text, match.index)}`, file, text, match.index, {
      migration: {
        target_slice: source.startsWith("crates/runmat-vm/") ? "R10/R29" : "R29",
        zero_state: "The legacy active-context adapter has no direct semantic consumers and is deleted in R29.",
      },
    });
  }

  if (source.startsWith("crates/runmat-runtime/src/")) {
    const marker = /#\[runtime_builtin\s*\(/g;
    for (let match; (match = marker.exec(text)); ) {
      const open = text.indexOf("(", match.index);
      const close = findBalanced(text, open);
      if (close < 0) throw new Error(`unclosed runtime_builtin attribute in ${source}:${lineAt(text, match.index)}`);
      const body = text.slice(open + 1, close);
      const name = stringField(body, "name") ?? cleanIdentity(body.match(/\bname\s*=\s*([^,\n]+)/)?.[1] ?? "");
      const tail = text.slice(close + 2, close + 700);
      const symbol = tail.match(/(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\$?[A-Za-z0-9_]+)/)?.[1];
      if (!name || !symbol) throw new Error(`incomplete runtime_builtin in ${source}:${lineAt(text, match.index)}`);
      const category = stringField(body, "category") ?? "";
      add("runtime_builtin", `${name}/${symbol}`, file, text, match.index, { builtin: name, category });
      if (/parallel|distributed|spmd|pool|future|batch/i.test(`${source} ${name} ${category}`)) {
        add("parallel_surface", name, file, text, match.index, { builtin: name, category });
      }
      if (/\bdescriptor\s*\(/.test(body)) add("builtin_descriptor", name, file, text, match.index, { builtin: name, category });
      if (/\btype_resolver\s*\(/.test(body)) add("type_resolver", name, file, text, match.index, { builtin: name, category });
      if (/\bsink\s*=\s*true/.test(body)) add("sink_policy", name, file, text, match.index, { builtin: name, category });
      marker.lastIndex = close + 2;
    }

    for (const [attribute, kind] of [["register_gpu_spec", "gpu_spec"], ["register_fusion_spec", "fusion_spec"]]) {
      const regex = new RegExp(`#\\[runmat_macros::${attribute}\\s*\\(`, "g");
      for (let match; (match = regex.exec(text)); ) {
        const open = text.indexOf("(", match.index);
        const close = findBalanced(text, open);
        const tail = text.slice(close + 2, close + 2200);
        const symbol = tail.match(/(?:pub\s+)?const\s+([A-Za-z0-9_]+)/)?.[1];
        const name = tail.match(/\bname\s*:\s*"([^"]+)"/)?.[1] ?? symbol;
        if (!symbol) throw new Error(`missing spec const in ${source}:${lineAt(text, match.index)}`);
        add(kind, `${name}/${symbol}`, file, text, match.index, { builtin: name });
        if (kind === "gpu_spec") {
          const residency = tail.match(/\bresidency\s*:\s*ResidencyPolicy::([A-Za-z0-9_]+)/)?.[1] ?? "unparsed";
          add("residency_policy", `${name}/${residency}`, file, text, match.index, { builtin: name, policy: residency });
        }
        regex.lastIndex = close + 2;
      }
    }
  }

  if (/crates\/(runmat-runtime|runmat-hir|runmat-lsp|runmat-builtins)\//.test(source)) {
    const fallback = /\bType::Unknown\b/g;
    for (let match; (match = fallback.exec(text)); ) add("semantic_fallback", `Type::Unknown/${enclosingFunction(text, match.index)}`, file, text, match.index);
    const namedFallback = /\b([A-Za-z0-9_]*fallback[A-Za-z0-9_]*)\b/gi;
    for (let match; (match = namedFallback.exec(text)); ) add("semantic_fallback", match[1], file, text, match.index);
  }

  if (source.startsWith("crates/runmat-lsp/") && /shape/i.test(text)) {
    const rules = /\b([A-Za-z0-9_]*shape[A-Za-z0-9_]*)\s*\(/gi;
    for (let match; (match = rules.exec(text)); ) add("lsp_shape_rule", match[1], file, text, match.index);
  }

  if (source.startsWith("crates/runmat-vm/src/")) {
    const special = /\b(?:VmIntrinsicBuiltin|VmDynamicWorkspaceBuiltin|VmIntrinsicExceptionBuiltin)::([A-Za-z0-9_]+)/g;
    for (let match; (match = special.exec(text)); ) add("vm_private_builtin", match[1], file, text, match.index);
  }

  if (source.startsWith("crates/runmat-turbine/src/")) {
    const bridges = /\b(?:pub\s+)?(?:unsafe\s+)?extern\s+"C"\s+fn\s+([A-Za-z0-9_]+)|\bfn\s+(declare_host_[A-Za-z0-9_]+)/g;
    for (let match; (match = bridges.exec(text)); ) add("turbine_host_bridge", match[1] ?? match[2], file, text, match.index);
  }

  if (source === "crates/runmat-accelerate-api/src/lib.rs") {
    const traitStart = text.indexOf("pub trait AccelProvider");
    if (traitStart >= 0) {
      const open = text.indexOf("{", traitStart);
      const close = findBalanced(text, open, "{", "}");
      const body = text.slice(open + 1, close);
      const methods = /\bfn\s+([A-Za-z0-9_]+)\s*\(/g;
      for (let match; (match = methods.exec(body)); ) {
        providerMethods.add(match[1]);
        add("provider_method", match[1], file, text, open + 1 + match.index);
      }
    }
  }

  if (/crates\/(runmat-execution|runmat-execution-artifact|runmat-execution-runner|runmat-test-runner-execution|runmat-node-agent|runmat-runtime)\//.test(source)) {
    const parallel = /\b(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z0-9_]*(?:Task|Future|Pool|Job|Worker|Cluster|Spmd|Distributed|Collective|Parallel)[A-Za-z0-9_]*)/g;
    for (let match; (match = parallel.exec(text)); ) add("parallel_surface", match[1], file, text, match.index);
  }

  if (/crates\/runmat-package(?:-cache|-cache-native)?\//.test(source)) {
    const api = /\bpub\s+(?:(struct|enum|trait)\s+([A-Za-z0-9_]+)|(?:async\s+)?fn\s+([A-Za-z0-9_]+))/g;
    for (let match; (match = api.exec(text)); ) add("package_surface", `${match[1] ?? "fn"}:${match[2] ?? match[3]}`, file, text, match.index);
  }

  if (/extern\s+"C"|\bmex\b|\bffi\b|\bforeign\b|\bjava\b/i.test(text) && !source.includes("/tests/")) {
    const api = /\b(?:pub\s+)?(?:unsafe\s+)?extern\s+"C"\s+fn\s+([A-Za-z0-9_]+)|\bpub\s+(?:struct|enum|trait)\s+([A-Za-z0-9_]*(?:Mex|Ffi|Foreign|Java)[A-Za-z0-9_]*)/g;
    for (let match; (match = api.exec(text)); ) add("interop_surface", match[1] ?? match[2], file, text, match.index);
  }

  if (/\b(?:#\[test\]|#\[tokio::test[^\]]*\]|wasm_bindgen_test)/.test(text)) {
    const tests = /#\[(?:tokio::)?test[^\]]*\][\s\S]{0,240}?\bfn\s+([A-Za-z0-9_]+)/g;
    for (let match; (match = tests.exec(text)); ) add("relevant_test", match[1], file, text, match.index);
  }
}

const providerCallPattern = new RegExp(`\\.(${[...providerMethods].sort((a, b) => b.length - a.length).join("|")})\\s*\\(`, "g");
for (const file of files) {
  const source = relative(file);
  if (!/crates\/(runmat-accelerate|runmat-core|runmat-runtime|runmat-turbine|runmat-vm)\//.test(source)) continue;
  const text = fs.readFileSync(file, "utf8");
  for (let match; (match = providerCallPattern.exec(text)); ) add("provider_direct_call", match[1], file, text, match.index);
}

entries.sort((a, b) => a.id.localeCompare(b.id));
const ids = new Set();
for (const entry of entries) {
  if (ids.has(entry.id)) throw new Error(`duplicate stable inventory id: ${entry.id}`);
  ids.add(entry.id);
  if (!policies[entry.kind]) throw new Error(`unclassified authority kind: ${entry.kind}`);
  if (!/^[A-G]$/.test(entry.cohort)) throw new Error(`unclassified cohort for ${entry.id}`);
  if (!entry.owner || !entry.migration.target_slice || !entry.migration.zero_state) {
    throw new Error(`incomplete ownership/migration classification for ${entry.id}`);
  }
  if (!fs.existsSync(path.join(repo, entry.source.path))) throw new Error(`missing inventory source: ${entry.source.path}`);
}
const catalogNames = catalog.builtins.map((builtin) => builtin.name);
if (new Set(catalogNames).size !== catalogNames.length) throw new Error("duplicate builtin name in docs/builtins/meta.json");
const counts = {};
const cohorts = {};
for (const entry of entries) {
  counts[entry.kind] = (counts[entry.kind] ?? 0) + 1;
  cohorts[entry.cohort] = (cohorts[entry.cohort] ?? 0) + 1;
}
const fingerprint = crypto.createHash("sha256").update(JSON.stringify(entries)).digest("hex");
const report = {
  schema_version: 1,
  purpose: "Deterministic migration inventory for RM-1064; derived evidence, never a semantic runtime authority.",
  fingerprint: `sha256:${fingerprint}`,
  totals: { entries: entries.length, by_kind: Object.fromEntries(Object.entries(counts).sort()), by_cohort: Object.fromEntries(Object.entries(cohorts).sort()) },
  entries,
};
const json = `${JSON.stringify(report, null, 2)}\n`;
const markdown = `# RM-1064 deterministic authority inventory\n\nThis file is generated by \`scripts/development/rm1064-inventory.mjs\`. It is migration evidence, not a semantic authority. The complete normalized JSON report is reproducibly written to \`target/rm1064-inventory/report.json\` by the script's \`--write\` mode rather than checked in as a generated godfile.\n\nFingerprint: \`${report.fingerprint}\`\n\nTotal entries: ${entries.length}\n\n## Counts by authority kind\n\n| Kind | Count |\n| -- | --: |\n${Object.entries(counts).sort().map(([kind, count]) => `| \`${kind}\` | ${count} |`).join("\n")}\n\n## Counts by builtin cohort\n\n| Cohort | Count |\n| -- | --: |\n${Object.entries(cohorts).sort().map(([cohort, count]) => `| ${cohort} | ${count} |`).join("\n")}\n\nEvery JSON entry carries a stable identity, owner, cohort, source location, target slice, and exact zero-state definition. CI's check mode rejects newly added or changed unclassified authority sites.\n`;

if (mode === "--stdout") process.stdout.write(json);
else if (mode === "--write") {
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, json);
  fs.writeFileSync(summaryPath, markdown);
  console.log(`wrote ${path.relative(repo, outputPath)} (${entries.length} entries)`);
} else {
  let failed = false;
  for (const [file, expected] of [[summaryPath, markdown]]) {
    if (!fs.existsSync(file) || fs.readFileSync(file, "utf8") !== expected) {
      console.error(`${path.relative(repo, file)} is stale; run scripts/development/rm1064-inventory.mjs --write`);
      failed = true;
    }
  }
  if (failed) process.exit(1);
  console.log(`RM-1064 inventory is current (${entries.length} entries; ${report.fingerprint})`);
}
