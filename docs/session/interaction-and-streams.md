---
title: "Interaction & Streams"
category: "Session Engine"
section: "7.5"
last_updated: "May 28, 2026"
---

# Interaction & Streams

The session installs per-execution runtime guards so host-visible output is captured as structured data instead of leaking through global process state.

## Captured Channels

| Channel | Source | Outcome field |
| --- | --- | --- |
| stdout/stderr/clear screen | Runtime console buffer | `streams` |
| MATLAB warnings | Runtime warning store | `diagnostics` and `warnings` in host payloads |
| display values | Session display policy | `display_events` and `flow` |
| plots | Plotting hooks | `figures_touched` |
| input/key events | Async interaction handler | `stdin_events` |
| profiling | Runtime/provider telemetry | `profiling` |
| fusion metadata | Optional session setting | `fusion_plan` |

Each execution resets the thread-local console buffer, warning store, recent-figure tracker, provider telemetry, current source context, and interrupt hook before running user bytecode.

## Async Input

Hosts can install an async input handler with `install_async_input_handler`. The runtime calls this handler when `input` or keypress-style interactions need host data. Without a host handler, native execution falls back to default terminal input helpers.

The session also installs an expression-evaluation hook for numeric `input()` parsing. It compiles and runs the typed expression through the same parser, HIR, bytecode, and interpreter path used by normal execution. On native hosts, that nested evaluation runs on a dedicated thread with a larger stack to avoid stack pressure from re-entering the interpreter.

## Cancellation

Cancellation is cooperative. `cancel_execution` flips an `AtomicBool` shared with the runtime interrupt hook. The VM polls that flag between interpreter steps and returns an execution-cancelled runtime error when it observes cancellation.

Long native, JIT, or provider operations can only stop at their polling or return boundaries. Hosts should treat cancellation as a request, not as preemptive thread termination.

## Diagnostics

Runtime errors are normalized into the configured error namespace, then populated with call-stack information from the session `SourcePool` when VM call frames are available. The public `ExecutionOutcome` carries diagnostics as structured records with code, severity, message, and optional span.

Warnings are also surfaced as diagnostics with warning severity. A request can therefore finish successfully while still returning diagnostic entries.

## Reentrancy

`ActiveExecutionGuard` prevents two simultaneous executions on the same `RunMatSession`. This protects shared fields such as `workspace_values`, `variable_array`, input hooks, and source context. Hosts that need concurrency should use multiple sessions.
