import { mkdirSync, readFileSync, writeFileSync, utimesSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { readJson, copyInput } from "./documents.mjs";
import { hash, safePath, inputFile } from "./manifest.mjs";
import { collectSnapshot, snapshotIdentity } from "./snapshots.mjs";
import { sourceExcerpts } from "./source-excerpts.mjs";
export function collectExamples(source, output, validate, requirements, snapshotsRoot) {
    const catalog = readJson(join(source, "docs/simulation/examples/catalog.json"));
    validate("example-package", catalog);
    const ids = new Set();
    const packages = [];
    for (const example of catalog.examples) {
        safePath(example.id);
        if (ids.has(example.id))
            throw new Error("Duplicate example");
        ids.add(example.id);
        if (new Set(example.files).size !== example.files.length || !example.files.includes(example.entry))
            throw new Error("Incomplete example inventory");
        if (example.model_illustration && !example.files.includes(example.model_illustration.path))
            throw new Error("Example model illustration is missing from its file inventory");
        for (const id of example.requirements)
            if (!requirements.has(id))
                throw new Error(`Unknown example requirement: ${id}`);
        const files = example.files.map(path => {
            const bytes = readFileSync(inputFile(source, "docs/simulation/examples/" + safePath(path)));
            const content = bytes.toString("utf8");
            if (Buffer.from(content).compare(bytes) !== 0 || content.includes("\u0000"))
                throw new Error("This first example exporter supports text assets only");
            return { path, bytes: bytes.length, sha256: hash(bytes), content };
        });
        // Reject undeclared local geometry/import dependencies before offering a complete package.
        for (const file of files.filter(file => file.path.endsWith(".fea"))) {
            const base = file.path.includes("/") ? file.path.slice(0, file.path.lastIndexOf("/") + 1) : "";
            for (const match of file.content.matchAll(/(?:file\s*=\s*|import\s+(?:\w+\s+)*)["']([^"']+)["']/g)) {
                const target = base + match[1].replace(/^\.\//, "");
                if (!files.some(file => file.path === target))
                    throw new Error(`Undeclared example dependency: ${target}`);
            }
        }
        const snapshot_digest = example.snapshot ? snapshotIdentity(snapshotsRoot, example.id) : undefined;
        const source_excerpts = sourceExcerpts(files.find(file => file.path === example.entry).content, example.source_sections);
        const package_identity = { ...example, snapshot_digest, files: files.map(({ content, ...metadata }) => metadata) };
        const digest = hash(JSON.stringify(package_identity));
        const packagePath = `examples/${example.id}/${digest}`;
        for (const file of files)
            copyInput(source, output, "docs/simulation/examples/" + file.path, packagePath + "/files/" + file.path);
        for (const file of files)
            utimesSync(join(output, packagePath, "files", file.path), new Date("1980-01-01T00:00:00Z"), new Date("1980-01-01T00:00:00Z"));
        execFileSync("zip", ["-q", "-X", join(output, packagePath, "example.zip"), ...example.files], { cwd: join(output, packagePath, "files"), env: { ...process.env, TZ: "UTC" } });
        const result = { ...example, files, source_excerpts, package_identity, digest, package_path: packagePath, ...(snapshot_digest ? { snapshot_digest, playback: collectSnapshot(snapshotsRoot, output, { ...example, files }, packagePath) } : {}) };
        mkdirSync(join(output, packagePath), { recursive: true });
        writeFileSync(join(output, packagePath, "package.json"), JSON.stringify(result, null, 2) + "\n");
        packages.push(result);
    }
    mkdirSync(join(output, "examples"), { recursive: true });
    writeFileSync(join(output, "examples/catalog.json"), JSON.stringify({ schema_version: 1, examples: packages }, null, 2) + "\n");
    return packages;
}
