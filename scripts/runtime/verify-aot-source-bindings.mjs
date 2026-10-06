// Run after scripts/build-runmat-with-aot-runtime.sh has produced an embedded-AOT runmat.
// This is a packaged-product gate: a missing binary or compile capability is a failure.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";

const binary = process.argv[2] && resolve(process.argv[2]);
if (!binary || !existsSync(binary)) {
    throw new Error("pass the existing embedded-AOT runmat binary as the sole argument");
}

const directory = mkdtempSync(join(tmpdir(), "runmat-aot-sources-"));
const timeout = 120_000;
function invoke(program, args = []) {
    const completed = spawnSync(program, args, {
        cwd: directory,
        env: { ...process.env, NO_COLOR: "1" },
        encoding: "utf8",
        maxBuffer: 16 * 1024 * 1024,
        timeout
    });
    if (completed.error) throw completed.error;
    return completed;
}
function compile(name, sourceText) {
    const source = join(directory, `${name}.m`);
    const output = join(directory, process.platform === "win32" ? `${name}.exe` : name);
    writeFileSync(source, sourceText, { encoding: "utf8", flag: "wx" });
    const result = invoke(binary, ["--color=never", "compile", source, "--output", output]);
    assert.equal(result.status, 0, `compiling ${basename(source)} failed:\n${result.stderr}`);
    assert.ok(existsSync(output), `compiling ${basename(source)} did not produce a linked executable`);
    return output;
}

try {
    const direct = compile("aot_source_direct", "function name = aot_source_direct(); name = mfilename(); end\n");
    for (let invocation = 0; invocation < 2; invocation++) {
        const result = invoke(direct);
        assert.equal(result.status, 0, `standalone source invocation ${invocation} failed:\n${result.stderr}`);
        assert.equal(result.stdout.trim(), "'aot_source_direct'", "standalone source was not rebound on each invocation");
    }

    const callback = compile("aot_source_callback", [
        "function name = aot_source_callback()",
        "  f = @local_source;",
        "  name = f();",
        "end",
        "function name = local_source()",
        "  name = mfilename();",
        "end",
        ""
    ].join("\n"));
    const callbackResult = invoke(callback);
    assert.equal(callbackResult.status, 0, `standalone callback failed:\n${callbackResult.stderr}`);
    assert.equal(callbackResult.stdout.trim(), "'aot_source_callback'", "local callback lost its owning file source");

    const failed = compile("aot_source_failure", [
        "function aot_source_failure()",
        "  error('RunMat:test:AotSource', 'aot source failure');",
        "end",
        ""
    ].join("\n"));
    const failure = invoke(failed);
    assert.notEqual(failure.status, 0, "standalone source error must not become success");
    assert.match(failure.stderr, /RunMat:test:AotSource/u);
    assert.match(failure.stderr, /aot source failure/u);
    console.log("linked AOT source, callback, re-invocation, and error witnesses passed");
} finally {
    rmSync(directory, { recursive: true, force: true });
}
