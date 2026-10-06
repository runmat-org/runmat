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

    const frame = compile("aot_native_frame", [
        "function name = aot_native_frame()",
        "  f = @local_frame;",
        "  name = f();",
        "end",
        "function name = local_frame()",
        "  info = getcallinfo();",
        "  name = info.name;",
        "end",
        ""
    ].join("\n"));
    const frameResult = invoke(frame);
    assert.equal(frameResult.status, 0, `standalone native frame failed:\n${frameResult.stderr}`);
    assert.equal(frameResult.stdout.trim(), "'local_frame'", "linked AOT callback inherited the caller frame");

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
    assert.match(
        failure.stderr,
        /aot_source_failure\.m:\d+:\d+/u,
        "standalone error must render the admitted source file and position before its lease retires"
    );

    // These are linked programs, not a VM-only binding check: the two-argument
    // forms must consume no result even inside a one-result function.
    const aritySuccess = compile("aot_arity_zero_result", [
        "function result = aot_arity_zero_result()",
        "  narginchk(0, 0);",
        "  nargoutchk(1, 1);",
        "  result = 7;",
        "end",
        ""
    ].join("\n"));
    const aritySuccessResult = invoke(aritySuccess);
    assert.equal(aritySuccessResult.status, 0, `linked zero-result checks failed:\n${aritySuccessResult.stderr}`);
    assert.equal(aritySuccessResult.stdout.trim(), "7", "zero-result checks changed the function result");

    for (const [name, builtin] of [
        ["aot_arity_nargin_bad_output", "narginchk"],
        ["aot_arity_nargout_bad_output", "nargoutchk"]
    ]) {
        const executable = compile(name, [
            `function result = ${name}()`,
            `  check = @${builtin};`,
            "  result = check(0, 1);",
            "end",
            ""
        ].join("\n"));
        const attempted = invoke(executable);
        assert.notEqual(attempted.status, 0, `${builtin} must reject an assigned result from its two-argument form`);
        assert.match(attempted.stderr, /RunMat:TooManyOutputs/u, `${builtin} lost its output error identifier`);
    }

    const legacyText = compile("aot_arity_legacy_text", [
        "function message = aot_arity_legacy_text()",
        "  message = nargoutchk(2, 2, 1);",
        "end",
        ""
    ].join("\n"));
    const textResult = invoke(legacyText);
    assert.equal(textResult.status, 0, `linked legacy text form failed:\n${textResult.stderr}`);
    assert.match(textResult.stdout, /Not enough output arguments\./u);

    const legacyStruct = compile("aot_arity_legacy_struct", [
        "function identifier = aot_arity_legacy_struct()",
        "  result = nargoutchk(0, 1, 2, 'struct');",
        "  identifier = result.identifier;",
        "end",
        ""
    ].join("\n"));
    const structResult = invoke(legacyStruct);
    assert.equal(structResult.status, 0, `linked legacy struct form failed:\n${structResult.stderr}`);
    assert.match(structResult.stdout, /MATLAB:nargoutchk:tooManyOutputs/u);

    const dealWide = compile("aot_deal_wide", [
        "function result = aot_deal_wide()",
        "  u = uint64(9007199254740992) + uint64(1);",
        "  [a, b, c] = deal(u);",
        "  assert(isa(a, 'uint64') && a == u && b == u && c == u);",
        "  result = 17;",
        "end",
        ""
    ].join("\n"));
    const dealWideResult = invoke(dealWide);
    assert.equal(dealWideResult.status, 0, `linked wide-integer deal failed:\n${dealWideResult.stderr}`);
    assert.equal(dealWideResult.stdout.trim(), "17", "linked deal changed wide-integer class or value");

    const dealZero = compile("aot_deal_zero_outputs", [
        "function result = aot_deal_zero_outputs()",
        "  deal(uint32(7));",
        "  result = 11;",
        "end",
        ""
    ].join("\n"));
    const dealZeroResult = invoke(dealZero);
    assert.equal(dealZeroResult.status, 0, `linked zero-output deal failed:\n${dealZeroResult.stderr}`);
    assert.equal(dealZeroResult.stdout.trim(), "11", "zero-output deal changed the function result");

    const dealBadCount = compile("aot_deal_bad_count", [
        "function result = aot_deal_bad_count()",
        "  f = @deal;",
        "  [a, b, c] = f(uint64(1), uint64(2));",
        "  result = 0;",
        "end",
        ""
    ].join("\n"));
    const dealBadCountResult = invoke(dealBadCount);
    assert.notEqual(dealBadCountResult.status, 0, "linked deal must reject mismatched input/output counts");
    assert.match(dealBadCountResult.stderr, /RunMat:deal:InputOutputCountMismatch/u);

    console.log("linked AOT source, callback, re-invocation, error, arity, and deal witnesses passed");
} finally {
    rmSync(directory, { recursive: true, force: true });
}
