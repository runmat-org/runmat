#!/usr/bin/env node
// @ts-check

// Scoped source conservation, not factory/lease authority or example execution.
// Historical v1 --reviewed reports are not completion evidence.
import { execFileSync, spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { canonicalEvidencePath, publishEvidenceBytes } from "./builtin-migration/atomic-evidence-publication.mjs";
import { reconcileDocumentationSourceFields } from "./builtin-migration/documentation-cutover.mjs";
import { runDocumentationExport } from "./builtin-migration/documentation-export.mjs";
import { compareCodePoint, SHADOW_ROOT, SIDECAR_ROOT } from "./builtin-migration/constants.mjs";
import { contentDigest, evidenceDigest } from "./builtin-migration/evidence.mjs";
import { exact, identity } from "./builtin-migration/schema.mjs";
import { sourceFieldBaselineSource } from "./builtin-migration/source-fields.mjs";
import { sourceSnapshot } from "./builtin-migration/snapshot.mjs";

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const options = argumentsFor(process.argv.slice(2));
const baseline = git(["rev-parse", "--verify", `${options.baseline}^{commit}`]).trim();
if (!/^[a-f0-9]{40}$/.test(baseline)) throw new Error("baseline must resolve to an actual Git commit");
const dispositionBytes = fs.readFileSync(options.dispositions);
const decoded = JSON.parse(dispositionBytes.toString("utf8"));
if (!Array.isArray(decoded)) exact(decoded, ["source_dispositions", "artifact_output"], "reviewed documentation inputs");
const sourceDispositions = Array.isArray(decoded) ? decoded : decoded.source_dispositions;
const expectedSources = Object.fromEntries(options.identities.map(name => [name, []]));
const baselineSources = [];
const paths = git(["ls-tree", "-r", "--name-only", "-z", baseline, "--", SIDECAR_ROOT, SHADOW_ROOT])
    .split("\0").filter(sourcePath => sourcePath.endsWith(".json")).sort(compareCodePoint);
for (const sourcePath of paths) {
    const name = path.posix.basename(sourcePath, ".json").toLowerCase();
    if (!options.identities.includes(name)) continue;
    const bytes = execFileSync("git", ["show", `${baseline}:${sourcePath}`], { cwd: repository, maxBuffer: 16 * 1024 * 1024 });
    const document = JSON.parse(bytes.toString("utf8"));
    if (typeof document.name === "string" && document.name.toLowerCase() !== name) {
        throw new Error(`${sourcePath}: declared baseline identity differs from its filename`);
    }
    const source = sourceFieldBaselineSource(sourcePath, bytes);
    expectedSources[name].push(source);
    baselineSources.push({ identity: name, ...source, bytes: bytes.toString("utf8") });
    if (present(path.join(repository, sourcePath))) throw new Error(`${sourcePath}: old editable source remains in the qualification worktree`);
}
for (const name of options.identities) {
    if (expectedSources[name].length === 0) throw new Error(`${name}: no baseline source exists; this source-conservation audit cannot assert a new identity`);
}
// Capture the actual build inputs, including staged deletions and dirty state.
// No arbitrary prebuilt exporter/file is accepted as evidence for these sources.
const roots = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml", "crates", SIDECAR_ROOT, SHADOW_ROOT,
    "scripts/development/audit-builtin-documentation-cutover.mjs", "scripts/development/builtin-migration"];
const before = sourceSnapshot(repository, roots);
let invocation;
const exported = runDocumentationExport((executable, args, settings) => {
    const actualArgs = [args[0], "--locked", "--offline", ...args.slice(1),
        ...options.identities.flatMap(name => ["--identity", name])];
    invocation = { executable, arguments: actualArgs, cwd: repository };
    return spawnSync(executable, actualArgs, { ...settings, cwd: repository });
}, { repository, transition: false });
const after = sourceSnapshot(repository, roots);
if (evidenceDigest(before) !== evidenceDigest(after)) throw new Error("source changed during the compiled documentation export");
if (exported.status !== 0) throw new Error(`documentation exporter failed (${exported.status}):\n${exported.stderr}`);
const conservation = reconcileDocumentationSourceFields({
    catalog_export: JSON.parse(exported.stdout), catalog_export_bytes: exported.stdout,
    source_dispositions: sourceDispositions, expected_sources: expectedSources,
    identities: options.identities,
});
const payload = {
    schema_version: 1, kind: "runmat-builtin-documentation-local-source-evidence",
    authority: "derived-local-source-evidence-only",
    identities: options.identities,
    baseline: { revision: `git:${baseline}`, sources: baselineSources },
    source: before,
    exporter: { ...invocation, exit_code: exported.status, signal: exported.signal,
        stderr: exported.stderr, stderr_digest: contentDigest(Buffer.from(exported.stderr)) },
    source_dispositions: { path: options.dispositions, content_digest: contentDigest(dispositionBytes),
        bytes: dispositionBytes.toString("utf8"), value: sourceDispositions },
    conservation,
};
const report = { ...payload, digest: evidenceDigest(payload) };
if (options.output) publishEvidenceBytes(canonicalEvidencePath(options.output), `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ scope: options.identities, ...conservation.summary, result: conservation.result,
    evidence: options.output, authority: report.authority }));
if (conservation.result !== "pass") process.exitCode = 1;

function git(args) {
    return execFileSync("git", args, { cwd: repository, encoding: "utf8", maxBuffer: 32 * 1024 * 1024 });
}
function present(target) {
    try { fs.lstatSync(target); return true; }
    catch (error) { if (error.code === "ENOENT") return false; throw error; }
}
function argumentsFor(values) {
    const result = { identities: [], baseline: null, dispositions: null, output: null };
    for (let index = 0; index < values.length; index += 1) {
        const value = values[index];
        if (["--baseline", "--dispositions", "--output"].includes(value)) {
            const argument = values[++index];
            if (!argument || argument.startsWith("--")) throw new Error(`${value} requires a value`);
            const member = value.slice(2);
            if (result[member] !== null) throw new Error(`duplicate ${value}`);
            result[member] = argument;
        } else if (value.startsWith("-")) {
            throw new Error(`unknown option ${value}; v1 --reviewed/--new reports are not source conservation proof`);
        } else {
            identity(value, "documentation identity");
            const name = value.toLowerCase();
            if (result.identities.includes(name)) throw new Error(`duplicate identity ${name}`);
            result.identities.push(name);
        }
    }
    if (!result.identities.length || !result.baseline || !result.dispositions) {
        throw new Error("usage: audit-builtin-documentation-cutover.mjs NAME [NAME ...] --baseline COMMIT --dispositions PATH [--output PATH]");
    }
    result.identities.sort(compareCodePoint);
    result.dispositions = path.resolve(result.dispositions);
    result.output = result.output === null ? null : path.resolve(result.output);
    return result;
}
