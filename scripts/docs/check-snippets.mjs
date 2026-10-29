// Runs `runmat check` on every ```fea block in docs/simulation (METHOD §2.4, E-DOC-1).
//
//   node scripts/docs/check-snippets.mjs --runmat <path-to-runmat> [--legacy-fea] [--only <doc-prefix>] [--json <report>]
//
// Fence forms:
//   ```fea                    a complete study; it must check clean.
//   ```fea pending=<id>       uses a feature that is not in the current release; it must FAIL `check`.
//                             When it starts passing, this script fails and names the flag to remove.
//   ```fea fragment           a syntax illustration, not a whole file; parsed by the fragment parser
//                             (E-DOC-2) once that entry point exists, and counted here.
//   ```fea source=<file>      an exact excerpt of docs/simulation/examples/<file>; the whole file is checked.
//   ```fea study=<name>       forwards `--study <name>` for files with several study roots.
// Complete snippets resolve relative file paths (CAD files, `.m` generators, imports) against a
// private copy of docs/simulation/examples, the canonical example package.
import { execFileSync } from "node:child_process";
import { cpSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { fencedBlocks, markdownFiles, readText } from "./markdown-scan.mjs";

export function collectSnippets(root, { only } = {}) {
    const snippets = [];
    for (const file of markdownFiles(root, { skip: ["examples", ".artifacts", "qualification"] })) {
        if (only && !file.startsWith(only)) continue;
        for (const block of fencedBlocks(readText(join(root, file)))) {
            if (block.language !== "fea") continue;
            const kind = block.attributes.fragment ? "fragment" : typeof block.attributes.pending === "string" ? "pending" : "complete";
            // An excerpt of a canonical example (docs-lint checks the slice) is checked as the whole file.
            const source = typeof block.attributes.source === "string" ? block.attributes.source : undefined;
            const code = source ? readText(join(root, "examples", source)) : block.code;
            snippets.push({ file, line: block.line, kind, pending: block.attributes.pending, study: block.attributes.study, source, code });
        }
    }
    return snippets;
}

export function checkSnippets({ source, runmat, legacyFea = false, only }) {
    const root = join(source, "docs/simulation");
    const snippets = collectSnippets(root, { only });
    const work = mkdtempSync(join(tmpdir(), "fea-snippets-"));
    const results = [];
    try {
        cpSync(join(root, "examples"), work, { recursive: true });
        for (const [index, snippet] of snippets.entries()) {
            if (snippet.kind === "fragment") { results.push({ ...snippet, verdict: "fragment" }); continue; }
            const path = snippet.source ? join(work, snippet.source) : join(work, `snippet-${index}.fea`);
            if (!snippet.source) writeFileSync(path, snippet.code);
            const args = [...(legacyFea ? ["fea", "check"] : ["check"]), path, ...(snippet.study ? ["--study", snippet.study] : [])];
            let passed = true, output = "";
            try { output = execFileSync(runmat, args, { cwd: work, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 120_000 }); }
            catch (error) { passed = false; output = `${error.stdout ?? ""}${error.stderr ?? ""}`; }
            const verdict = snippet.kind === "pending" ? (passed ? "pending-now-passes" : "pending-fails-as-expected") : (passed ? "clean" : "fails");
            results.push({ ...snippet, verdict, output: output.replaceAll(path, `${snippet.file}:${snippet.line}`).trim() });
        }
    }
    finally { rmSync(work, { recursive: true, force: true }); }
    const failures = results.filter(result => result.verdict === "fails" || result.verdict === "pending-now-passes");
    return { snippets: results.length, failures, results };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const args = process.argv.slice(2);
    const value = flag => { const index = args.indexOf(flag); return index === -1 ? undefined : args[index + 1]; };
    const runmat = value("--runmat");
    if (!runmat) throw new Error("Pass --runmat <path to the runmat CLI>");
    const source = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
    const report = checkSnippets({ source, runmat, legacyFea: args.includes("--legacy-fea"), only: value("--only") });
    if (value("--json")) writeFileSync(value("--json"), JSON.stringify(report, null, 2) + "\n");
    const count = verdict => report.results.filter(result => result.verdict === verdict).length;
    for (const failure of report.failures) {
        console.log(`docs/simulation/${failure.file}:${failure.line}: ${failure.verdict === "fails" ? "does not check clean" : `passes now; remove the pending=${failure.pending} flag`}`);
        if (failure.verdict === "fails") console.log(failure.output.split("\n").slice(0, 12).map(line => `    ${line}`).join("\n"));
    }
    console.log(`check-snippets: ${report.snippets} blocks; ${count("clean")} clean, ${count("pending-fails-as-expected")} pending (fail as expected), ${count("fragment")} fragments, ${report.failures.length} failures`);
    process.exit(report.failures.length ? 1 : 0);
}
