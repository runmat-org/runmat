// Run after scripts/build-runmat-with-aot-runtime.sh has produced an embedded-AOT runmat.
// This is a packaged-product gate: a missing binary or compile capability is a failure.
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { verifyAotSuperConstructors } from "./aot-super-constructors.mjs";

const binary = process.argv[2] && resolve(process.argv[2]);
if (!binary || !existsSync(binary)) {
    throw new Error("pass an existing embedded-AOT runmat binary");
}
const selectedSourceOnly = process.argv[3] === "--selected-source";
if ((selectedSourceOnly && process.argv.length !== 7) ||
    (!selectedSourceOnly && (process.argv.length > 4 || (process.argv[3] && !["--java", "--java-only", "--super-only", "--super-package-only"].includes(process.argv[3]))))) {
    throw new Error("usage: node verify-aot-source-bindings.mjs <runmat> [--java|--java-only|--super-only|--super-package-only|--selected-source <canonical-pack-manifest> <component-root> <independent-host-admission>]");
}
const selectedCompileFlags = selectedSourceOnly ? [
    "--selected-pack-manifest", resolve(process.argv[4]),
    "--selected-components", resolve(process.argv[5]),
    "--selected-host-admission", resolve(process.argv[6])
] : [];
const javaOnly = process.argv[3] === "--java-only";
const superOnly = process.argv[3] === "--super-only";
const superPackageOnly = process.argv[3] === "--super-package-only";
const javaRequested = process.argv[3] === "--java" || javaOnly;
const javaHome = javaRequested && process.env.RUNMAT_TEST_JAVA_HOME;
if (javaRequested && !javaHome) {
    throw new Error("--java requires RUNMAT_TEST_JAVA_HOME to name a JDK with javac, jar, and java");
}

const directory = mkdtempSync(join(tmpdir(), "runmat-aot-sources-"));
const timeout = 120_000;
const traceInvocations = process.env.RUNMAT_AOT_RETAIN_FAILED_ARTIFACTS === "1";
function invoke(program, args = []) {
    if (traceInvocations) console.error(`linked AOT start: ${JSON.stringify({ program, args })}`);
    const started = Date.now();
    const completed = spawnSync(program, args, {
        cwd: directory,
        env: {
            ...process.env,
            RUNMAT_EXECUTION_STATE_DIR: process.env.RUNMAT_EXECUTION_STATE_DIR ?? join(directory, "execution-state"),
            RUNMAT_PACKAGE_CACHE_DIR: process.env.RUNMAT_PACKAGE_CACHE_DIR ?? join(directory, "package-cache"),
            NO_COLOR: "1"
        },
        encoding: "utf8",
        maxBuffer: 16 * 1024 * 1024,
        timeout
    });
    if (traceInvocations) {
        console.error(`linked AOT exit: ${JSON.stringify({ program, elapsed_ms: Date.now() - started, status: completed.status, signal: completed.signal, error: completed.error?.code ?? null })}`);
        if (completed.error || completed.status !== 0 || completed.signal) {
            console.error(`linked AOT output tail: ${JSON.stringify({ stdout: String(completed.stdout ?? "").slice(-2048), stderr: String(completed.stderr ?? "").slice(-2048) })}`);
        }
    }
    if (completed.error) throw completed.error;
    return completed;
}
function compile(name, sourceText, options = []) {
    const source = join(directory, `${name}.m`);
    const output = join(directory, process.platform === "win32" ? `${name}.exe` : name);
    writeFileSync(source, sourceText, { encoding: "utf8", flag: "wx" });
    const result = invoke(binary, ["--color=never", "compile", source, "--output", output, ...options]);
    assert.equal(result.status, 0, `compiling ${basename(source)} failed:\n${result.stderr}`);
    assert.ok(existsSync(output), `compiling ${basename(source)} did not produce a linked executable`);
    return output;
}

function verifyCapturedJavaPolicy() {
    const jdk = resolve(javaHome);
    const executable = name => join(jdk, "bin", process.platform === "win32" ? `${name}.exe` : name);
    for (const name of ["javac", "jar", "java"]) {
        assert.ok(existsSync(executable(name)), `--java requires ${name} in RUNMAT_TEST_JAVA_HOME/bin`);
    }
    const project = join(directory, "aot_java_project");
    const javaSource = join(project, "java", "fixture", "packaged", "Value.java");
    const classes = join(project, "classes");
    const jarPath = join(project, "lib", "fixture.jar");
    const configPath = join(project, "runmat.toml");
    const source = join(project, "src", "aot_java_policy.m");
    const output = join(directory, process.platform === "win32" ? "aot_java_policy.exe" : "aot_java_policy");
    const planPath = join(directory, "aot_java_policy.plan.json");
    mkdirSync(join(project, "java", "fixture", "packaged"), { recursive: true });
    mkdirSync(classes, { recursive: true });
    mkdirSync(join(project, "lib"), { recursive: true });
    mkdirSync(join(project, "src"), { recursive: true });
    writeFileSync(javaSource, [
        "package fixture.packaged;",
        "public final class Value {",
        "  private Value() {}",
        "  public static int read() {",
        "    return Integer.parseInt(System.getProperty(\"runmat.aot.captured.policy\", \"-1\"));",
        "  }",
        "}",
        ""
    ].join("\n"));
    const compiledJava = invoke(executable("javac"), ["-d", classes, javaSource]);
    assert.equal(compiledJava.status, 0, `compiling AOT Java fixture failed:\n${compiledJava.stderr}`);
    const archivedJava = invoke(executable("jar"), ["--create", "--file", jarPath, "-C", classes, "."]);
    assert.equal(archivedJava.status, 0, `archiving AOT Java fixture failed:\n${archivedJava.stderr}`);
    writeFileSync(configPath, [
        "[package]",
        "name = \"aot-java-policy\"",
        "version = \"1.0.0\"",
        "",
        "[sources]",
        "roots = [\"src\"]",
        "",
        "[java-artifacts.fixture]",
        "path = \"lib/fixture.jar\"",
        "",
        "[runtime.foreign.java]",
        `home = ${JSON.stringify(jdk)}`,
        "options = [\"-Drunmat.aot.captured.policy=73\"]",
        ""
    ].join("\n"));
    writeFileSync(source, [
        "function value = aot_java_policy()",
        "  advance = @(x) javaMethod('sum', 'java.lang.Integer', x, int32(2));",
        "  stream = javaMethod('iterate', 'java.util.stream.IntStream', int32(1), advance);",
        "  limited = javaMethod('limit', stream, int64(3));",
        "  values = javaMethod('toArray', limited);",
        "  assert(isequal(values, int32([1, 3, 5])));",
        "  callbackSum = sum(values);",
        "  assert(callbackSum == 9);",
        "  policyValue = javaMethod('read', 'fixture.packaged.Value');",
        "  assert(policyValue == 73);",
        "  value = policyValue;",
        "end",
        ""
    ].join("\n"));
    let compileSource = relative(directory, source);
    if (process.platform !== "win32") {
        const projectAlias = join(directory, "aot_java_project_alias");
        symlinkSync(project, projectAlias, "dir");
        compileSource = join(basename(projectAlias), "src", basename(source));
    }
    const compilation = invoke(binary, [
        "--color=never", "--config", configPath, "compile", compileSource,
        "--output", output, "--link-plan-json", planPath
    ]);
    assert.equal(compilation.status, 0, `compiling linked AOT Java fixture failed:\n${compilation.stderr}`);
    assert.ok(existsSync(output), "linked AOT Java fixture has no executable");
    const plan = JSON.parse(readFileSync(planPath, "utf8"));
    const javaGrant = plan.interop.adapters.filter(item => item.adapter === "java");
    assert.equal(javaGrant.length, 1, "linked AOT Java plan must select exactly one Java adapter grant");
    assert.equal(javaGrant[0].artifact_identities.length, 1, "linked AOT Java plan must grant exactly one JAR identity");
    assert.match(javaGrant[0].artifact_identities[0], /^java:v\d+:sha256:[0-9a-f]{64}$/u);

    // The linked process must use embedded source/JAR bytes and captured
    // policy, not a later project file or surviving local artifact path.
    renameSync(source, `${source}.removed-after-compile`);
    renameSync(jarPath, `${jarPath}.removed-after-compile`);
    writeFileSync(configPath, "[runtime.foreign.java]\noptions = []\n");
    const result = invoke(output);
    assert.equal(result.status, 0, `linked AOT Java execution failed:\n${result.stderr}`);
    assert.equal(result.stdout.trim(), "73", "linked Java lost its archived source/JAR, captured JVM option, or callback reentry");
}

function verifyPureAotSources() {
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

    // The three direct strcmp sites receive actual String scalars from the
    // canonical newline binding. This exercises an emitted and linked native
    // object, not the test-linked NativeEntryPoint used by the S1 feedback test.
    const scopedPlan = join(directory, "aot_s1_closed_world.plan.json");
    const scoped = compile("aot_s1_closed_world", [
        "function result = aot_s1_closed_world()",
        "  left = newline();",
        "  right = newline();",
        "  result = 0;",
        "  if strcmp(left, right)",
        "    result = result + 1;",
        "  end",
        "  if strcmp(left, right)",
        "    result = result + 10;",
        "  end",
        "  if strcmp(left, right)",
        "    result = result + 100;",
        "  end",
        "end",
        ""
    ].join("\n"), ["--policy", "closed-world", "--link-plan-json", scopedPlan]);
    const plan = JSON.parse(readFileSync(scopedPlan, "utf8"));
    assert.equal(plan.policy, "closed-world", "the emitted program must use closed-world bindings");
    for (const name of ["newline", "strcmp"]) {
        assert.ok(
            plan.retained_builtin_bindings.some(binding => binding.name === name && binding.variant === "default"),
            `closed-world link plan omitted the selected ${name} binding`
        );
        assert.ok(
            plan.reachability.edges.some(edge => edge.to === `builtin:${name}` && edge.reason === "direct_call"),
            `compiled source lost its direct ${name} call target`
        );
    }
    // The link plan records reachable targets, not a count of MIR/Native IR
    // call sites. The three source sites and checksum check linked behavior.
    if (process.platform !== "win32") {
        const symbolGate = fileURLToPath(new URL("../check-closed-world-binary.sh", import.meta.url));
        const checked = invoke(symbolGate, [scoped, scopedPlan]);
        assert.equal(checked.status, 0, `closed-world symbol gate failed:\n${checked.stderr}`);
    }
    const scopedResult = invoke(scoped);
    assert.equal(scopedResult.status, 0, `linked scalar strcmp program failed:\n${scopedResult.stderr}`);
    assert.equal(scopedResult.stdout.trim(), "111", "linked scalar strcmp checksum changed");

    const bound = compile("aot_bound_local_capture", [
        "function result = aot_bound_local_capture()",
        "  zeta = 50;",
        "  alpha = 11;",
        "  handle = @captured;",
        "  result = feval(handle, 3);",
        "  function value = captured(delta)",
        "    value = zeta - alpha + delta;",
        "  end",
        "end",
        ""
    ].join("\n"));
    const boundResult = invoke(bound);
    assert.equal(boundResult.status, 0, `standalone bound local capture failed:\n${boundResult.stderr}`);
    assert.equal(boundResult.stdout.trim(), "42", "linked feval lost or reordered its two local captures");

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

    verifyAotSuperConstructors({ directory, compile, invoke });
}

function verifySelectedSourceEvaluation() {
    function requireSelectedPlan(planPath) {
        const plan = JSON.parse(readFileSync(planPath, "utf8"));
        assert.equal(plan.product?.mode, "selected", "source witness must compile against a verified selected pack");
        assert.equal(plan.runtime_capabilities?.dynamic_source_loading, true, "selected runtime lacks actual source service capability");
        assert.ok(
            plan.product.binding.required_services.some(service => service.kind === "source_evaluation") &&
                plan.product.binding.services.some(service => service.kind === "source_evaluation"),
            "selected program did not retain the SourceEvaluation service"
        );
    }
    function compileSelected(name, source) {
        const planPath = join(directory, `${name}.plan.json`);
        const output = compile(name, source, ["--policy", "dynamic-runtime", "--link-plan-json", planPath, ...selectedCompileFlags]);
        requireSelectedPlan(planPath);
        return output;
    }

    const direct = compileSelected("aot_selected_numeric_text", [
        "function result = aot_selected_numeric_text()",
        "  result = str2num('1+2');",
        "end",
        ""
    ].join("\n"));
    const directResult = invoke(direct);
    assert.equal(directResult.status, 0, `selected linked str2num failed:\n${directResult.stderr}`);
    assert.equal(directResult.stdout.trim(), "3", "selected numeric text did not execute its linked evaluator");

    const project = join(directory, "aot_selected_source_project");
    const source = join(project, "src", "aot_selected_wide.m");
    const privateSource = join(project, "src", "private", "private_offset.m");
    const config = join(project, "runmat.toml");
    const output = join(directory, process.platform === "win32" ? "aot_selected_wide.exe" : "aot_selected_wide");
    const planPath = join(directory, "aot_selected_wide.plan.json");
    mkdirSync(join(project, "src", "private"), { recursive: true });
    writeFileSync(config, "[package]\nname = \"aot-selected-source\"\nversion = \"1.0.0\"\n\n[sources]\nroots = [\"src\"]\n");
    writeFileSync(source, [
        "function result = aot_selected_wide()",
        "  result = 7 + str2num('local_double(2)+private_offset(3)');",
        "end",
        "function value = local_double(x)",
        "  value = 2*x;",
        "end",
        ""
    ].join("\n"));
    writeFileSync(privateSource, "function value = private_offset(x)\n  value = x+5;\nend\n");
    const compilation = invoke(binary, ["--color=never", "--config", config, "compile", source,
        "--output", output, "--policy", "dynamic-runtime", "--link-plan-json", planPath, ...selectedCompileFlags]);
    assert.equal(compilation.status, 0, `compiling selected local/private source failed:\n${compilation.stderr}`);
    assert.ok(existsSync(output), "selected local/private source has no linked executable");
    requireSelectedPlan(planPath);
    renameSync(source, `${source}.removed-after-compile`);
    renameSync(privateSource, `${privateSource}.removed-after-compile`);
    const wideResult = invoke(output);
    assert.equal(wideResult.status, 0, `selected local/private source failed:\n${wideResult.stderr}`);
    assert.equal(wideResult.stdout.trim(), "19", "selected source lost exact local/private target ownership");

    const escaped = compileSelected("aot_selected_escaped", [
        "function result = aot_selected_escaped()",
        "  callback = make_callback();",
        "  result = callback(2);",
        "end",
        "function callback = make_callback()",
        "  callback = @late_source;",
        "end",
        "function value = late_source(x)",
        "  value = x + str2num('1+2');",
        "end",
        ""
    ].join("\n"));
    const escapedResult = invoke(escaped);
    assert.equal(escapedResult.status, 0, `escaped selected callback failed:\n${escapedResult.stderr}`);
    assert.equal(escapedResult.stdout.trim(), "5", "retired callback owner lost its linked source/code lease");

    const silent = compileSelected("aot_selected_silent", [
        "function result = aot_selected_silent()",
        "  str2num('1+2');",
        "  result = 9;",
        "end",
        ""
    ].join("\n"));
    const silentResult = invoke(silent);
    assert.equal(silentResult.status, 0, `silent selected source failed:\n${silentResult.stderr}`);
    assert.equal(silentResult.stdout.trim(), "9", "source evaluation leaked a suppressed intermediate result");
}

let failed = false;
try {
    if (selectedSourceOnly) verifySelectedSourceEvaluation();
    else if (superOnly || superPackageOnly) verifyAotSuperConstructors({ directory, compile, invoke, packageOnly: superPackageOnly });
    else if (!javaOnly) verifyPureAotSources();
    if (javaRequested) verifyCapturedJavaPolicy();

    console.log(selectedSourceOnly
        ? "selected linked AOT SourceEvaluation, str2num, local/private, escaped callback, and silent-result witnesses passed"
        : superPackageOnly
        ? "linked AOT packaged superclass witness passed"
        : superOnly
        ? "linked AOT superclass witnesses passed"
        : javaOnly
        ? "linked AOT captured Java policy witness passed"
        : javaRequested
        ? "linked AOT source, callback, closed-world scalar strcmp, re-invocation, error, arity, deal, superclass, and captured Java policy witnesses passed"
        : "linked AOT source, callback, closed-world scalar strcmp, re-invocation, error, arity, deal, and superclass witnesses passed");
} catch (error) {
    failed = true;
    throw error;
} finally {
    if (failed && process.env.RUNMAT_AOT_RETAIN_FAILED_ARTIFACTS === "1") {
        console.error(`retained failed linked AOT fixtures: ${directory}`);
    } else {
        rmSync(directory, { recursive: true, force: true });
    }
}
