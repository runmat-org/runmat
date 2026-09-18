# runmat check

Use `runmat check` to find syntax errors, unresolved function calls, and type or matrix-size problems that RunMat can identify before execution. It analyzes your existing `.m` code without running the script.

## Quick start

[Install the CLI](/docs/runtime/getting-started/install) if `runmat` is not on your path.

To check an existing script, open a terminal in its directory and run:

```bash
runmat check analysis.m
```

Replace `analysis.m` with your script's filename. For a project that calls functions in other directories, see [Project sources](#project-sources).

To try a minimal example first, save the following as `analysis.m` in a new directory:

```matlab
values = [2, 4, 6];
average = mean(values);
disp(average);
```
Run the check from that directory:

```bash
runmat check analysis.m
```
Diagnostic output:

```text
checked analysis.m: 0 error(s), 0 warning(s)
```
RunMat found no errors or warnings during analysis. Next, run the script with representative inputs to check its runtime behavior and results.

Warnings are reported without failing the command by default. Use `-D warnings` when warnings should also make an automated check fail.

The command accepts one file:

```text
runmat check [OPTIONS] <FILE>
```
`<FILE>` is the `.m` script to check. For a multi-file project, check the script you normally run. Configure source roots—the directories RunMat searches for your project's functions—when those functions live in other directories. Do not treat this command as a recursive check of every independent script in a directory.

## What gets checked

`runmat check` analyzes your script and the project functions available through its source configuration. It uses the same analysis as RunMat's editor tooling and can report:

- Syntax and semantic errors.
- Type and matrix-shape incompatibilities that can be established statically.
- Calls that cannot be resolved from built-ins, local functions, imports, or configured project sources.
- Runtime-dependent resolution, such as a call whose lookup depends on `addpath`.

`runmat check` catches problems it can identify from your source code. Values and array sizes that depend on input data may only be known when the script runs. After checking, run your script with representative inputs to validate those cases. Use verbose or JSON output to see analysis completeness by domain.

## Options

| Option | Behavior |
| --- | --- |
| `--path DIRECTORY` | Add a source lookup root for this check. May be repeated. |
| `-D warnings` | Return failure when any diagnostic warning is present. `-D warning` is also accepted. |
| `--json` | Emit structured diagnostics and analysis results as JSON. |
| `-v`, `--verbose` | Include completed analysis domains as well as diagnostics for scripts. |
| `-h`, `--help` | Show command help. |
| `-V`, `--version` | Show the CLI version. |

Global options include `--color auto|always|never` and package-resolution controls `--offline`, `--locked`, and `--frozen`. See [CLI runtime options](/docs/runtime/getting-started/cli#pass-runtime-options), [Configuration](/docs/runtime/getting-started/config), and [Packages](/docs/runtime/packages) for shared configuration and lockfile behavior.

For example:

```bash
runmat check --verbose analysis.m
runmat check --help
```

## Project sources

If `main.m` calls a helper in `toolbox/helper.m`, and `toolbox` is not already configured as a source root:

`main.m`:

```matlab
value = helper(3);
```
`toolbox/helper.m`:

```matlab
function y = helper(x)
y = x * 2;
end
```
From the directory containing `main.m`, run:

```bash
runmat check main.m
runmat check --path toolbox main.m
```
Without the additional source root, RunMat reports that it cannot find `helper`. Adding `--path toolbox` makes the helper available to the check. A helper placed directly beside `main.m` is also discoverable without `--path`.

For multi-file project configuration, define the source roots in a `runmat.toml` beside `main.m`:

```toml
[package]
name = "check-example"

[sources]
roots = ["toolbox"]
```
Then check without an extra path flag:

```bash
runmat check main.m
```
The result is clean. Source roots make cross-file definitions available during analysis; they are not instructions to execute every script under those roots. See [Projects](/docs/runtime/getting-started/projects) for packages, classes, dependencies, and entrypoints.

### Runtime path changes

If your project has no configured source roots and `dynamic.m` adds the helper's directory at runtime:

```matlab
addpath('toolbox');
value = helper(3);
```

```bash
runmat check dynamic.m
```
RunMat reports `RM-RES0002` and points to both the function call and the earlier `addpath`. Because the check does not execute `addpath`, it cannot determine the call's target from that runtime path change alone. The call may still resolve when the script runs. To make the helper available during checking, add its directory with `--path` or configure a source root.

In JSON output, name-resolution coverage is reported as `runtime_dependent`.

## Reading results

Use the diagnostic code, source location, and help text to identify what needs attention. If your check reports one of the problems below, use the corresponding guidance to investigate it.

If you are checking independent files without a project manifest or additional source roots, the commands below show the expected diagnostics. Output blocks show diagnostic stdout; environment-specific startup messages on stderr are omitted.

### An unresolved function

If `missing.m` calls a function that cannot be found:

```matlab
value = definitely_missing(1);
```

```bash
runmat check missing.m
```

```text
warning[RM-RES0001]: cannot find function `definitely_missing`
 --> missing.m:1:9
  |
1 | value = definitely_missing(1);
  |         ^^^^^^^^^^^^^^^^^^^^^ not defined in this file or project
  = note: RunMat checked built-ins, local functions, imports, and every configured source root
  = help: place `definitely_missing.m` beside this source or in a source root configured by `runmat.toml`
checked missing.m: 0 error(s), 1 warning(s)
```
`RM-RES0001` identifies the diagnostic. `missing.m:1:9` points to line 1, column 9. The note explains what was searched; the help suggests how to expose the function. Check the spelling and provide the implementation in a discoverable source location.

This command exits successfully by default. To make the same warning fail an automated check:

```bash
runmat check -D warnings missing.m
```
The diagnostic still says `warning`, but the process exits with status 1.

### A syntax error

If `syntax.m` contains an incomplete assignment:

```matlab
value = ;
```

```bash
runmat check syntax.m
```

```text
error[RunMat:ParseError]: unexpected token: Semicolon; found `;`
 --> syntax.m:1:9
  |
1 | value = ;
  |         ^ syntax error occurs here
checked syntax.m: 1 error(s), 0 warning(s)
```
Supply an expression after `=` and check again. This error exits with status 1.

### A matrix-shape error

If `shape.m` multiplies matrices whose inner dimensions do not match:

```matlab
A = ones(2, 3);
B = ones(4, 2);
C = A * B;
```

```bash
runmat check shape.m
```

```text
error[RM-TYPE-MATMUL]: matrix inner dimensions 3 and 4 do not agree
 --> shape.m:3:1
  |
3 | C = A * B;
  | ^^^^^^^^^ static value contract is not satisfied here
checked shape.m: 1 error(s), 0 warning(s)
```
Matrix multiplication requires matching inner dimensions. If the intended operation uses a 3-by-2 right-hand matrix, change `B` to `ones(3, 2)` and check again. You can use `runmat check` to statically detect linear algebra operation and type rule violations.

## Exit status and automation

| Script check result | Exit status |
| --- | --- |
| No errors or warnings | `0` |
| Warnings, without `-D warnings` | `0` |
| Errors, or warnings with `-D warnings` | `1` |

Invalid command-line usage can have a different nonzero status. In automation, treat any nonzero exit as failure.

### JSON output

Using the independent files above:

```bash
runmat check --json analysis.m
runmat check --json missing.m
runmat check --json syntax.m
```
The outcomes are `clean`, `warnings`, and `failed`, respectively. A script analysis failure still emits its JSON result before exiting nonzero. This does not guarantee a result envelope for every startup, argument, or file-loading failure.

Selected fields from the clean result (other fields omitted):

```json
{
  "schema_version": 1,
  "outcome": "clean",
  "analysis": {
    "async_safety": "complete",
    "definite_assignment": "complete",
    "effects": "complete",
    "name_resolution": "complete",
    "shapes": "partial",
    "syntax": "complete",
    "types": "partial"
  },
  "summary": {
    "errors": 0,
    "warnings": 0,
    "warnings_denied": false
  }
}
```
`outcome` tells you whether the check reported errors or warnings. `analysis` describes the coverage of each kind of analysis.

Here, `types` and `shapes` are `partial`: RunMat checks the type and matrix-size rules it can establish, but coverage is not exhaustive. It can detect the matrix multiplication error shown above without establishing every type or array dimension. A `partial` status does not itself make the check fail.

Each diagnostic includes its severity, code, message, and source location, with additional notes or help when available. Source locations include byte offsets and line/column coordinates. Keep stderr separate when parsing JSON.

### Preserve the exit status in CI

To save a JSON report while preserving the check's exit status in CI, use a shell script such as `check-ci.sh`:

```sh
#!/bin/sh
status=0
runmat check --json -D warnings analysis.m > check.json || status=$?
cat check.json
exit "$status"
```

```bash
sh check-ci.sh
```
This prints the report and exits with the check's status, even though `cat` succeeds. The clean example exits 0. Replacing the contents of `analysis.m` with the unresolved-function or syntax-error example makes it exit 1. Your CI system can retain `check.json` as an artifact.

## Limits and next steps

Use checking to identify source-level problems, then run your script with representative inputs and compare the results you rely on. If a dependency is only loaded at runtime, its availability still needs to be checked during execution.

After a clean check, you can run your script with:

```bash
runmat run analysis.m
```
If you have MATLAB-style tests, see the [CLI test workflow](/docs/runtime/getting-started/cli#test-projects) and [MATLAB compatibility guide](/docs/runtime/matlab-compatibility#check-run-and-test-existing-code).

