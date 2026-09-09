# runmat check

Check a MATLAB script or FEA document before running it. `runmat check` reports problems it can find without executing your script or running an FEA solver.

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
`<FILE>` is a `.m` script or `.fea` document. For a multi-file project, pass the entry script and configure its source roots as described below. Do not treat this command as a recursive check of every independent script in a directory.

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
| `--path DIRECTORY` | Add a MATLAB lookup root for this check. May be repeated. |
| `-D warnings` | Return failure when any diagnostic warning is present. `-D warning` is also accepted. |
| `--json` | Emit structured output. Script and FEA payloads have different shapes. |
| `-v`, `--verbose` | Include completed analysis domains as well as diagnostics for scripts. |
| `-h`, `--help` | Show command help. |
| `-V`, `--version` | Show the CLI version. |

`--path` and warnings-as-errors apply to the MATLAB script analysis path. FEA uses its own study validation path.

Global options include `--color auto|always|never` and package-resolution controls `--offline`, `--locked`, and `--frozen`. See [CLI runtime options](/docs/runtime/getting-started/cli#pass-runtime-options), [Configuration](/docs/runtime/getting-started/config), and [Packages](/docs/runtime/packages) for shared configuration and lockfile behavior.

For example:

```bash
runmat check --verbose analysis.m
runmat check --help
```

## Project sources

In a separate directory, create `main.m`:

```matlab
value = helper(3);
```
Create a `toolbox` directory beside it, containing `helper.m`:

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

For persistent project configuration, put this `runmat.toml` beside `main.m`:

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

In the same two-file layout, remove `runmat.toml` for this example and save this as `dynamic.m`:

```matlab
addpath('toolbox');
value = helper(3);
```

```bash
runmat check dynamic.m
```
The check reports `RM-RES0002`, with the call site and the earlier `addpath` as a related causal location. Name resolution is `runtime_dependent`: checking does not execute `addpath` to discover the final target. This is a supported dynamic execution boundary, rather than proof that the call will fail. Use source roots or `--path` when the helper should participate in static analysis.

## Reading results

The following examples are independent files in a directory without a project manifest. Output blocks show diagnostic stdout; environment-specific startup messages on stderr are omitted.

### An unresolved function

Save as `missing.m`:

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

Save as `syntax.m`:

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

Save as `shape.m`:

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
Matrix multiplication requires matching inner dimensions. If the intended operation uses a 3-by-2 right-hand matrix, change `B` to `ones(3, 2)` and check again. Whether that change is mathematically appropriate depends on your problem.

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
`analysis` describes completeness separately from `outcome`: here type and shape analysis remain partial. Diagnostics include severity, code, message, primary source spans, related spans, notes, and help. A span can include byte offsets and line/column coordinates. Keep stderr separate when parsing JSON.

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

After a clean check, run the quick-start script:

```bash
runmat run analysis.m
```
It prints `4`. For your own project, use representative inputs and compare the results you rely on. If you have MATLAB-style tests, see the [CLI test workflow](/docs/runtime/getting-started/cli#test-projects) and [MATLAB compatibility guide](/docs/runtime/matlab-compatibility#check-run-and-test-existing-code).

## FEA documents

For `.fea` studies and sweeps, checking loads geometry, resolves selectors, validates the document, and builds a solve plan without running the solver. It can read geometry files and use runtime caches; it is not a promise of zero filesystem activity.

For a small validation example, save this as `triangle.obj`:

```text
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
```
Beside it, save `triangle.fea`:

```yaml
version: 1
kind: study
id: triangle_static
geometry:
  path: triangle.obj
  units: meter
model:
  profile: linear_static_structural
run:
  backend: cpu
```

```bash
runmat check triangle.fea
runmat check --json triangle.fea
```
The human result begins with `OK triangle_static` and reports `validation: passed (0 issues)`. The JSON result contains `validation.valid: true` and a `plan` object. It does not use the script diagnostic envelope shown above.

This triangle uses the default profile scaffold to demonstrate validation plumbing; it is not a physically validated engineering model. Geometry paths are relative to the `.fea` file. See [Using FEA](/docs/fea/using-fea), [Models](/docs/fea/models), and [Solves](/docs/fea/solves) to define and validate a real study.
