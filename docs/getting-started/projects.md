---
title: "Projects"
category: "Getting Started"
section: "1.4"
last_updated: "May 30, 2026"
---

# Projects

A RunMat project is a folder of MATLAB-syntax code anchored by a `runmat.toml` or `runmat.json` manifest. The manifest tells RunMat where the source lives, which other projects it depends on, and which workflows can be run by name. RunMat discovers it by walking up from the source file or working directory, so the same project resolves consistently across the CLI, Desktop, and the LSP.

Typically, you should use a project with a manifest once a single script is no longer enough on its own, for example when it calls helper functions in nearby files, organizes code into packages or classes, hides private functions, pins runtime settings, or defines a named workflow that should run the same way everywhere.

## Direct Files And Named Entrypoints

You can always run a file directly:

```bash
runmat src/analyze_sales.m
runmat run src/analyze_sales
```

The second form can infer `.m` for local paths. Direct file execution is the simplest way to run one script and does not require an entrypoint declaration.

Named entrypoints are useful when a project has one or more canonical entrypoints that you want to run by name:

```bash
runmat run analysis
```

Named entrypoints are declared in the project manifest:

```toml
[package]
name = "sales_report"
version = "0.1.0"

[sources]
roots = ["src"]

[entrypoints.analysis]
path = "src/analyze_sales"

[runtime.language]
compat = "runmat"
```

`path` targets point at a source file. The `.m` extension may be omitted. An entrypoint can also target a discovered module/function pair:

```toml
[entrypoints.summary]
module = "stats"
function = "summarize"
```

Exactly one target form is allowed for each entrypoint: `path`, or `module` plus `function`.

## Source Roots

`[sources].roots` defines where RunMat scans for project source files:

```toml
[sources]
roots = ["src", "tests"]
```

Source roots are relative to the manifest directory. They must exist and cannot use parent-directory traversal. Files outside configured roots can still be run directly by path, but they are not part of the project source index used for module/function entrypoints and cross-file symbol discovery.

## Local Dependencies

Local dependencies make another RunMat project available during composition:

```text
sales-report/
  runmat.toml
  src/analyze_sales.m
  deps/
    shared-tools/
      runmat.toml
      src/+format/titleCase.m
```

Root manifest:

```toml
[package]
name = "sales_report"

[sources]
roots = ["src"]

[dependencies]
tools = { path = "deps/shared-tools", version = "0.1.0" }
```

Dependency manifest:

```toml
[package]
name = "shared_tools"
version = "0.1.0"

[sources]
roots = ["src"]
```

The dependency alias participates in project symbol discovery. A source file from the dependency can be resolved by its own qualified name, by its package-qualified name, or through the root dependency alias when imports or function handles need that form.

## Complete Project Example

This project has one top-level script, a sibling helper function, a private helper, a package function, and a class folder:

```text
sales-report/
  runmat.toml
  src/
    analyze_sales.m
    normalizeRows.m
    private/
      localScale.m
    +stats/
      summarize.m
    @Report/
      Report.m
```

`runmat.toml`:

```toml
[package]
name = "sales_report"
version = "0.1.0"

[sources]
roots = ["src"]

[entrypoints.analysis]
path = "src/analyze_sales"
```

`src/analyze_sales.m`:

```matlab
sales = [100 120 130; 80 95 105];

scaled = localScale(sales);
normalized = normalizeRows(scaled);

[totals, averages] = stats.summarize(normalized);

report = Report("sales", totals);
headline = report.title();

disp(headline);
```

`src/normalizeRows.m`:

```matlab
function out = normalizeRows(x)
    rowTotals = sum(x, 2);
    out = x ./ rowTotals;
end
```

`src/private/localScale.m`:

```matlab
function y = localScale(x)
    y = x * 100;
end
```

`src/+stats/summarize.m`:

```matlab
function [totals, averages] = summarize(x)
    totals = sum(x, 1);
    averages = mean(x, 1);
end
```

`src/@Report/Report.m`:

```matlab
classdef Report
    properties
        Name
        Totals
    end

    methods
        function obj = Report(name, totals)
            obj.Name = name;
            obj.Totals = totals;
        end

        function text = title(obj)
            text = "Report: " + obj.Name;
        end
    end
end
```

Run the project entrypoint:

```bash
cd sales-report
runmat run analysis
```

The top-level script variables become candidates for the session workspace after execution. Locals inside `normalizeRows`, `localScale`, `stats.summarize`, and `Report.title` stay inside their function frames.

## What Projects Do Not Change

Project composition does not replace MATLAB source rules. Packages still use `+pkg` folders, classes still use class files and `@ClassName` folders, and private functions remain private to their source area.

`import` controls name visibility inside source code. Dependencies control which external project symbols are available to the resolver. Keeping those responsibilities separate lets RunMat preserve MATLAB-style code while giving hosts a stable project boundary.

## Related Docs

- [Configuration Reference](/docs/runtime/getting-started/config)
- [Command Line Interface](/docs/runtime/getting-started/cli)
- [Module Composition](/docs/runtime/compiler/modules)
