# runmat check

Check your source code before running it. `runmat check` reports problems it can find without executing your script.

## Quick start

[Install the CLI](/docs/runtime/getting-started/install) if `runmat` is not on your path.

Save this as `analysis.m` in a new directory:

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
This means no static errors or warnings were found. It does not mean the script has executed or its results are correct.

The command accepts one file:

```text
runmat check [OPTIONS] <FILE>
```
`<FILE>` is the `.m` script to check. For a multi-file project, pass the entry script and configure its source roots as described below. Do not treat this command as a recursive check of every independent script in a directory.

## What gets checked

For `.m` files, checking uses the parser, HIR and MIR lowering, static analysis, source lookup, and compile validation shared with editor tooling. It can report:

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

If your project has an entry script `main.m` that calls a helper:

```matlab
value = helper(3);
```
and the helper is defined in `toolbox/helper.m`:

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
The first command warns that `helper` cannot be found. The second resolves it and reports zero errors and warnings. A helper placed directly beside `main.m` is also discoverable without `--path`.

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
The check reports `RM-RES0002`, with the call site and the earlier `addpath` as a related causal location. Name resolution is `runtime_dependent`: checking does not execute `addpath` to discover the final target. This is a supported dynamic execution boundary, rather than proof that the call will fail. Use source roots or `--path` when the helper should participate in static analysis.

## Reading results

If you are checking independent files in a directory without a project manifest, use the diagnostics below to identify and address common problems. Output blocks show diagnostic stdout; environment-specific startup messages on stderr are omitted.

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
`RM-RES0001` identifies the diagnostic. `missing.m:1:9` points to line 1, column 9. The note explains what was searched; the help suggests how to expose the function. Check the spelling and provide the implementation in a discoverable source location. Do not add an empty placeholder merely to silence the warning.

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
`analysis` describes coverage separately from `outcome`. A `partial` domain means the checker can identify some problems in that domain but does not provide complete coverage of all its rules or cases. Here, type and shape checking can report proven incompatibilities, such as the matrix multiplication error above, without establishing every type or array dimension. `partial` is a coverage indicator, not an additional warning or error; `outcome: clean` means no diagnostics were reported. Diagnostics include severity, code, message, primary source spans, related spans, notes, and help. A span can include byte offsets and line/column coordinates. Keep stderr separate when parsing JSON.

### Preserve the exit status in CI

Save this as `check-ci.sh` beside `analysis.m`:

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

Checking does not run your script, read its runtime input data, exercise every branch, or establish numerical correctness. An unresolved call is not necessarily an unsupported built-in, and a clean result does not guarantee that every runtime dependency is available.

After a clean check, you can run your script with:

```bash
runmat run analysis.m
```
Use representative inputs and compare the results you rely on. If you have MATLAB-style tests, see the [CLI test workflow](/docs/runtime/getting-started/cli#test-projects) and [MATLAB compatibility guide](/docs/runtime/matlab-compatibility#check-run-and-test-existing-code).

