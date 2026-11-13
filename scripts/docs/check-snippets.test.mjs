import test from "node:test";
import assert from "node:assert/strict";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { checkSnippets } from "./check-snippets.mjs";

test("complete blocks must check, pending blocks must fail, fragments are counted", () => {
    const root = mkdtempSync(join(tmpdir(), "snippets-"));
    try {
        const docs = join(root, "docs/simulation");
        mkdirSync(join(docs, "examples"), { recursive: true });
        writeFileSync(join(docs, "examples/geometry.step"), "");
        writeFileSync(join(docs, "page.md"), [
            "# Page", "```fea", "ok", "```", "```fea", "broken", "```",
            "```fea pending=acoustic.static", "broken", "```", "```fea pending=fluid.harmonic", "ok", "```",
            "```fea fragment", "anything", "```", "```text", "ok", "```", "",
        ].join("\n"));
        // A stand-in CLI: `check <file>` passes when the file says "ok" and its CAD neighbor exists.
        const cli = join(root, "runmat");
        writeFileSync(cli, "#!/bin/sh\n[ \"$1\" = check ] && [ -f \"$(dirname \"$2\")/geometry.step\" ] && grep -q '^ok$' \"$2\" || { echo error; exit 1; }\n");
        chmodSync(cli, 0o755);
        const report = checkSnippets({ source: root, runmat: cli });
        assert.deepEqual(report.results.map(result => result.verdict), ["clean", "fails", "pending-fails-as-expected", "pending-now-passes", "fragment"]);
        assert.deepEqual(report.failures.map(failure => failure.line), [5, 11]);
    }
    finally { rmSync(root, { recursive: true, force: true }); }
});
