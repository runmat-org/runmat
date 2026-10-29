import Ajv from "ajv";
import { mkdirSync, mkdtempSync, renameSync, rmSync, readFileSync, writeFileSync, readdirSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { readJson, copyInput, collectDocuments } from "./documents.mjs";
import { collectExamples } from "./examples.mjs";
import { collectFea } from "./fea.mjs";
import { writeManifest } from "./manifest.mjs";
import { collectGraphics } from "./graphics.mjs";
import { productionDocumentation } from "./production-documentation.mjs";
export function buildSourceDocs({ source, output, requirementsPath, builtinDescriptorsPath, builtinDocumentationPath, verifiedBuiltinDocumentation, evidencePath, snapshotsRoot, graphicsRoot }) {
    source = resolve(source);
    output = resolve(output);
    if (!output.startsWith(source + "/target/"))
        throw new Error("Docs export output must be inside the source target directory");
    const ajv = new Ajv({ allErrors: true });
    const schemas = new Map();
    for (const name of ["document", "fea-manual", "evidence-index", "example-package", "pending-features"])
        schemas.set(name, ajv.compile(readJson(join(source, `docs/schema/${name}.schema.json`))));
    const validate = (name, value) => { const check = schemas.get(name); if (!check(value))
        throw new Error(`${name}: ${ajv.errorsText(check.errors)}`); };
    mkdirSync(dirname(output), { recursive: true });
    const staging = mkdtempSync(join(dirname(output), ".docs-staging-"));
    try {
        const register = readJson(requirementsPath);
        const requirements = new Map(register.requirements.map(requirement => [requirement.id, requirement]));
        const examples = collectExamples(source, staging, validate, requirements, snapshotsRoot);
        if (examples.some(example => example.playback)) collectGraphics(graphicsRoot, staging);
        collectDocuments(source, staging, validate, builtinDocumentationPath, verifiedBuiltinDocumentation);
        collectFea(source, staging, register, examples, evidencePath ? readJson(evidencePath) : { schema_version: 1, records: [] }, validate);
        mkdirSync(join(staging, "builtins"), { recursive: true });
        const descriptors = readJson(builtinDescriptorsPath);
        if (!Array.isArray(descriptors.builtins))
            throw new Error("Invalid built-in descriptors");
        writeFileSync(join(staging, "builtins/descriptors.json"), JSON.stringify(descriptors, null, 2) + "\n");
        for (const name of readdirSync(join(source, "docs/schema")))
            copyInput(source, staging, "docs/schema/" + name, "schema/" + name);
        let revision = "unknown", dirty = true;
        try {
            revision = execFileSync("git", ["rev-parse", "HEAD"], { cwd: source, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
            dirty = !!execFileSync("git", ["status", "--porcelain"], { cwd: source, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] }).trim();
        }
        catch { /* Fixture exports explicitly have unknown identity. */ }
        const manifest = writeManifest(staging, { revision, dirty });
        const backup = output + ".previous";
        rmSync(backup, { recursive: true, force: true });
        try {
            renameSync(output, backup);
        }
        catch (error) {
            if (error.code !== "ENOENT")
                throw error;
        }
        try {
            renameSync(staging, output);
        }
        catch (error) {
            try {
                renameSync(backup, output);
            }
            catch { }
            throw error;
        }
        rmSync(backup, { recursive: true, force: true });
        return manifest;
    }
    catch (error) {
        rmSync(staging, { recursive: true, force: true });
        throw error;
    }
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const args = process.argv.slice(2);
    const options = new Map();
    for (let index = 0; index < args.length; index += 2) {
        if (!args[index + 1] || options.has(args[index]))
            throw new Error("Invalid export arguments");
        options.set(args[index], args[index + 1]);
    }
    for (const required of ["--requirements", "--builtin-descriptors", "--out"])
        if (!options.has(required))
            throw new Error(`Missing ${required}`);
    const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    const verifiedBuiltinDocumentation = productionDocumentation(source, { exportPath: options.get("--builtin-documentation") });
    const manifest = buildSourceDocs({ source, output: options.get("--out"), requirementsPath: options.get("--requirements"), builtinDescriptorsPath: options.get("--builtin-descriptors"), verifiedBuiltinDocumentation, evidencePath: options.get("--evidence-manifest"), snapshotsRoot: options.get("--example-snapshots"), graphicsRoot: options.get("--graphics-runtime") });
    console.log(`Source docs: ${manifest.content_hash} (${manifest.files.length} files)`);
}
