---
title: "MATLAB Language Extensions"
category: "Getting Started"
section: "1.7"
last_updated: "August 16, 2026"
---

# MATLAB Language Extensions

RunMat runs MATLAB-language source code and also offers optional capabilities that are specific to RunMat. These additions are called language extensions. An extension may be new syntax, a new function, an extra argument form, or support for a data type that is outside the documented MATLAB behavior of an otherwise compatible function.

Extensions let new RunMat programs use runtime features such as asynchronous execution while existing MATLAB programs retain a predictable compatibility boundary. They are enabled by default in `runmat` mode and excluded where required by `matlab` mode.

## What Counts as an Extension

An extension is deliberate user-visible behavior beyond RunMat's MATLAB compatibility target. Common forms include:

- RunMat syntax such as `async function` and `await`.
- RunMat APIs such as `spawn` and the `data.*` persistence namespace.
- RunMat utility functions without a compatibility-target counterpart, such as `urlencode` and `urldecode`.
- Additional builtin signatures, options, aliases, or accepted data classes identified as RunMat-only on a function's reference page.
- Explicit device behavior that is useful in RunMat but is not part of the compatible function surface.

Extensions are part of RunMat's supported interface. They are documented, tested, and assigned stable compatibility metadata when they affect a builtin call.

## What Does Not Count as an Extension

An internal optimization does not change the language surface. JIT compilation, kernel fusion, automatic GPU residency, transparent gathering for host execution, and a different implementation algorithm are not extensions when the program's accepted inputs and observable result remain compatible.

An unimplemented feature is also not an extension. If RunMat does not yet support a documented form, the function reference should state that limitation directly. Enabling extension mode does not imply that every unsupported MATLAB feature becomes available.

## Choosing a Compatibility Mode

Set the mode in `runmat.toml`:

```toml
[runtime.language]
compat = "runmat"
```

| Mode | Extension behavior |
| --- | --- |
| `runmat` | Enables supported RunMat language and builtin extensions. This is the default. |
| `matlab` | Excludes extensions that fall outside the MATLAB compatibility target and uses MATLAB-oriented error identifiers where supported. |
| `strict` | Tightens permissive syntax such as command-style calls. It does not select a historical MATLAB release or a different numeric model. |

Compatibility mode is a policy for the whole execution request. It does not disable transparent runtime optimizations, and it does not change documented numeric results merely because RunMat uses a different execution path.

## Reading Function Documentation

Each builtin reference separates ordinary behavior from RunMat-only forms. When a call is an extension, the page identifies the affected argument or behavior and the mode required to use it. Independent extensions on the same function remain independent: enabling `runmat` mode permits the implemented forms, but one extension does not silently broaden another argument.

When `matlab` mode rejects an extension, RunMat reports a compatibility error before performing provider access, file I/O, graphics mutation, or another avoidable side effect. Stable extension identifiers in builtin metadata allow editors and other tooling to explain the rejected form.

For the broader language and builtin coverage model, see [MATLAB Language Compatibility](/docs/runtime/getting-started/compatability). For configuration details, see the [Configuration Reference](/docs/runtime/getting-started/config). Contributors should use the [Semantic Compatibility Engineering Policy](/docs/runtime/development/backwards-compat) when classifying a new behavior.
