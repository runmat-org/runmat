// @ts-check

import { execFileSync } from "node:child_process";
import { mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { buildInventory } from "./inventory.mjs";
import { buildPlan, validatePlan } from "./plan.mjs";
import { buildExecutionMatrix } from "./execution-matrix.mjs";
import { buildArtifactManifestFromFiles, buildArtifactManifestFromTree, validateArtifactManifest } from "./artifacts.mjs";
import { DESKTOP_HOST_PROTOCOL_PROBE, NATIVE_EMBEDDED_AOT_PROBE, runDesktopHostProtocolProbe, runNativeEmbeddedAotProbe, validateProductProbe } from "./product-probe.mjs";
import { reconcileShardResults } from "./reconcile.mjs";
import { executeShard } from "./runner-process.mjs";
import { computeRunnerDigest, readJson, repositoryRoot, sourceIdentity, writeJson } from "./io.mjs";
import { renderReconciliationHtml, renderReconciliationMarkdown } from "./report-renderer.mjs";

export const PROTOCOL_COMMANDS = Object.freeze(["artifacts", "probe", "inventory", "plan", "matrix", "run", "reconcile"]);

export async function runProtocolCli(argv) {
    const [command, ...rest] = argv;
    if (!PROTOCOL_COMMANDS.includes(command)) throw new Error(`Unknown builtin example verifier command: ${command}`);
    const options = parseOptions(rest);
    if (command === "artifacts") return artifactsCommand(options);
    if (command === "probe") return probeCommand(options);
    if (command === "inventory") return inventoryCommand(options);
    if (command === "plan") return planCommand(options);
    if (command === "matrix") return matrixCommand(options);
    if (command === "run") return runCommand(options);
    return reconcileCommand(options);
}

function artifactsCommand(options) {
    const output = required(options.output, "--output");
    const product = required(options.product, "--product");
    const artifactProfile = required(options.profile, "--profile");
    const sourceRevision = required(options["source-revision"], "--source-revision");
    const producerRevision = options["producer-revision"];
    const files = parseAssignments(arrayOption(options.file), "--file", "artifact role");
    const entrypoints = parseAssignments(arrayOption(options.entrypoint), "--entrypoint", "artifact entrypoint role");
    if (options.root && Object.keys(files).length) throw new Error("artifacts accepts either --root with --entrypoint or --file, not both");
    let manifest;
    if (options.root) {
        manifest = buildArtifactManifestFromTree({ product, artifactProfile, sourceRevision, producerRevision, manifestPath: output, root: options.root, entrypoints });
    } else if (product === "native-cli" && artifactProfile === "embedded-aot") {
        const roles = Object.keys(files);
        if (roles.length !== 1 || roles[0] !== "runmat-binary") throw new Error("Native --file compatibility requires exactly one runmat-binary=PATH");
        const binary = resolve(files["runmat-binary"]);
        manifest = buildArtifactManifestFromTree({
            product,
            artifactProfile,
            sourceRevision,
            producerRevision,
            manifestPath: output,
            root: dirname(binary),
            entrypoints: { "runmat-binary": basename(binary) }
        });
    } else {
        manifest = buildArtifactManifestFromFiles({ product, artifactProfile, sourceRevision, producerRevision, manifestPath: output, files });
    }
    writeJson(output, manifest);
    console.log(`Wrote ${product}/${artifactProfile} artifact manifest to ${resolve(output)}`);
}

async function probeCommand(options) {
    const output = required(options.output, "--output");
    const manifestPaths = arrayOption(options["artifact-manifest"]);
    if (manifestPaths.length !== 1) throw new Error("probe requires exactly one --artifact-manifest");
    const manifestPath = manifestPaths[0];
    const kind = required(options.kind, "--kind");
    if (![NATIVE_EMBEDDED_AOT_PROBE, DESKTOP_HOST_PROTOCOL_PROBE].includes(kind)) throw new Error(`Unsupported product probe kind: ${kind}`);
    const manifest = readJson(manifestPath);
    const probeFields = {
        artifactManifest: manifest,
        artifactManifestPath: manifestPath,
        sourceRevision: manifest.sourceRevision,
        producerRevision: manifest.producerRevision,
        timeoutMs: options["timeout-ms"] === undefined ? undefined : integerOption(options["timeout-ms"], "--timeout-ms", 120_000)
    };
    const probe = kind === NATIVE_EMBEDDED_AOT_PROBE
        ? runNativeEmbeddedAotProbe(probeFields)
        : await runDesktopHostProtocolProbe(probeFields);
    validateProductProbe(probe, { artifactManifest: manifest, artifactManifestPath: manifestPath, verifyFiles: true });
    writeJson(output, probe);
    console.log(`Wrote ${kind} ${probe.result.status} evidence to ${resolve(output)}`);
    if (probe.result.status !== "passed") process.exitCode = 1;
}

function parseAssignments(specifications, flag, label) {
    const values = {};
    for (const specification of specifications) {
        const separator = specification.indexOf("=");
        if (separator <= 0 || separator === specification.length - 1) throw new Error(`${flag} must use ROLE=PATH`);
        const role = specification.slice(0, separator);
        if (Object.hasOwn(values, role)) throw new Error(`Duplicate ${label}: ${role}`);
        values[role] = specification.slice(separator + 1);
    }
    return values;
}

function inventoryCommand(options) {
    const output = required(options.output, "--output");
    const root = repositoryRoot();
    const source = options["source-revision"]
        ? { sourceRevision: options["source-revision"], sourceState: options["source-state"] ?? "clean" }
        : sourceIdentity(root);
    const exported = options.export ? readJson(options.export) : exportDocuments(root);
    const inventory = buildInventory(exported, {
        ...source,
        runnerDigest: options["runner-digest"] ?? computeRunnerDigest(root),
        scope: { builtins: arrayOption(options.builtin), filter: options.filter, limit: options.limit }
    });
    writeJson(output, inventory);
    console.log(`Wrote ${inventory.executionUnits.length} execution units to ${resolve(output)}`);
}

function planCommand(options) {
    const inventoryPath = required(options.inventory, "--inventory");
    const output = required(options.output, "--output");
    const inventory = readJson(inventoryPath);
    const topology = options.topology ? readJson(options.topology) : null;
    const plan = buildPlan(inventory, topology, { product: options.product });
    writeJson(output, plan);
    console.log(`Wrote ${plan.lanes.reduce((sum, lane) => sum + lane.shardCount, 0)} shard assignments to ${resolve(output)}`);
}

function matrixCommand(options) {
    const plan = readJson(required(options.plan, "--plan"));
    const output = required(options.output, "--output");
    const matrix = buildExecutionMatrix(plan);
    writeJson(output, matrix);
    console.log(`Wrote ${matrix.include.length} frozen shard matrix entries to ${resolve(output)}`);
}

function runCommand(options) {
    const inventory = readJson(required(options.inventory, "--inventory"));
    const plan = readJson(required(options.plan, "--plan"));
    validatePlan(plan, inventory);
    const lane = required(options.lane, "--lane");
    const shardIndex = integerOption(options["shard-index"], "--shard-index", 0);
    const resultPath = required(options["result-out"] ?? options.output, "--result-out");
    const lanePlan = plan.lanes.find((candidate) => candidate.lane === lane);
    if (!lanePlan) throw new Error(`Plan contains no lane named ${lane}`);
    const shard = lanePlan.shards[shardIndex];
    if (!shard) throw new Error(`Plan contains no shard ${shardIndex} for ${lane}`);
    if (options["assignment-digest"] && options["assignment-digest"] !== shard.assignmentDigest) throw new Error(`Assignment digest does not match ${lane}/${shardIndex}`);
    let artifactManifest = null;
    let artifactManifestPath = null;
    if (options["artifact-manifest"]) {
        artifactManifestPath = options["artifact-manifest"];
        artifactManifest = readJson(artifactManifestPath);
        validateArtifactManifest(artifactManifest, { sourceRevision: plan.sourceRevision, verifyFiles: true, manifestPath: artifactManifestPath });
        if (artifactManifest.product !== lanePlan.product) throw new Error(`Artifact product does not match lane ${lane}`);
        const productPlan = plan.products.find((candidate) => candidate.kind === lanePlan.product);
        if (!productPlan || artifactManifest.artifactProfile !== productPlan.artifactProfile) throw new Error(`Artifact profile does not match the plan for ${lane}`);
    }
    const result = executeShard({
        inventory,
        plan,
        lanePlan,
        shard,
        artifactManifest,
        artifactManifestPath,
        resultPath,
        workDirectory: options["work-dir"]
    });
    console.log(`Wrote ${lane}/${shardIndex} ${result.status} result to ${resolve(resultPath)}`);
    if (result.status !== "passed") process.exitCode = 1;
}

function reconcileCommand(options) {
    const inventory = readJson(required(options.inventory, "--inventory"));
    const plan = readJson(required(options.plan, "--plan"));
    const paths = [...options.positionals];
    if (options.result) paths.push(...arrayOption(options.result));
    if (options["results-dir"]) {
        paths.push(...findShardResults(resolve(options["results-dir"])));
    }
    if (paths.length === 0) throw new Error("reconcile requires shard result files");
    const artifactPaths = arrayOption(options["artifact-manifest"]);
    const artifactManifests = artifactPaths.map((path) => {
        const manifest = readJson(path);
        validateArtifactManifest(manifest, { sourceRevision: plan.sourceRevision, verifyFiles: true, manifestPath: path });
        return { manifest, manifestPath: path };
    });
    const productProbes = arrayOption(options["product-probe"]).map((path) => {
        const probe = readJson(path);
        validateProductProbe(probe, { sourceRevision: plan.sourceRevision });
        return probe;
    });
    const result = reconcileShardResults(inventory, plan, paths.map(readJson), {
        closure: booleanOption(options.closure),
        artifactManifests,
        productProbes
    });
    const output = required(options.output, "--output");
    writeJson(output, result);
    const reportDirectory = options["report-dir"] ? resolve(options["report-dir"]) : dirname(resolve(output));
    mkdirSync(reportDirectory, { recursive: true });
    writeFileSync(join(reportDirectory, "builtin-example-report.md"), renderReconciliationMarkdown(result), "utf8");
    writeFileSync(join(reportDirectory, "builtin-example-report.html"), renderReconciliationHtml(result), "utf8");
    console.log(`Reconciled ${result.summary.shards} shards: ${result.status}`);
    if (result.status !== "passed") process.exitCode = 1;
}

function findShardResults(root) {
    const results = [];
    const visit = (directory) => {
        for (const entry of readdirSync(directory, { withFileTypes: true })) {
            const path = join(directory, entry.name);
            if (entry.isSymbolicLink()) throw new Error(`Shard result directory contains a symbolic link: ${path}`);
            if (entry.isDirectory()) visit(path);
            else if (entry.isFile() && entry.name.endsWith(".shard-result.json")) results.push(path);
        }
    };
    visit(root);
    return results.sort();
}

function exportDocuments(root) {
    const override = process.env.RUNMAT_EXAMPLE_DOCUMENTATION_BINARY;
    const command = override ? resolve(root, override) : "cargo";
    const args = override
        ? ["--transition"]
        : ["run", "--quiet", "-p", "runmat-builtins", "--bin", "export_builtin_documentation", "--", "--transition"];
    return JSON.parse(execFileSync(command, args, { cwd: root, encoding: "utf8", maxBuffer: 128 * 1024 * 1024 }));
}

function parseOptions(args) {
    const options = { positionals: [] };
    for (let index = 0; index < args.length; index += 1) {
        const arg = args[index];
        if (!arg.startsWith("--")) { options.positionals.push(arg); continue; }
        const name = arg.slice(2);
        if (name === "closure") { options[name] = true; continue; }
        const value = args[++index];
        if (value === undefined || value.startsWith("--")) throw new Error(`${arg} requires a value`);
        if (name === "result" || name === "artifact-manifest" || name === "product-probe" || name === "file" || name === "entrypoint" || name === "builtin") options[name] = [...arrayOption(options[name]), value];
        else options[name] = value;
    }
    return options;
}

function required(value, flag) {
    if (typeof value !== "string" || !value) throw new Error(`${flag} is required`);
    return value;
}

function integerOption(value, flag, fallback) {
    if (value === undefined) return fallback;
    if (!/^(0|[1-9]\d*)$/.test(value)) throw new Error(`${flag} must be a non-negative integer`);
    return Number(value);
}

function booleanOption(value) {
    return value === true || value === "true" || value === "1";
}

function arrayOption(value) {
    if (value === undefined) return [];
    return Array.isArray(value) ? value : [value];
}
