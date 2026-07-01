---
title: "Host Integration"
category: "Session Engine"
section: "7.6"
last_updated: "May 28, 2026"
---

# Host Integration

The session API is intentionally host-neutral. Different frontends wrap it differently, but they all submit source through `ExecutionRequest` and consume structured results.

## CLI REPL

The CLI REPL creates one `RunMatSession`, keeps it alive for the process, and submits ordinary source lines as `SourceInput::Text { name: "<repl>", text }`. Special dot commands such as `.stats`, `.gc`, and `.info` are handled by the CLI before code reaches the session.

Lines beginning with `!` are also CLI-only shell escapes. The native CLI runs the rest of the line in the platform shell, prints stdout and stderr, reports non-zero exits, and then returns to the REPL. This is host execution, not a `runmat-core` ABI feature: `.m` scripts, WASM, notebook hosts, and editor hosts do not receive implicit shell execution by submitting source that starts with `!`.

The CLI uses in-memory line-editor history. Persistent notebook or command history is not part of `runmat-core`.

## WASM And TypeScript

`RunMatWasm` wraps `RunMatSession` for browser and TypeScript callers. Its main methods include:

| Method | Role |
| --- | --- |
| `executeRequest` | Submit source and return a serialized execution payload. |
| `resetSession` | Recreate the underlying session while preserving configuration. |
| `cancelExecution` | Request cooperative cancellation. |
| `setInputHandler` | Route runtime input prompts to JavaScript. |
| `workspaceSnapshot` | Return workspace metadata and preview tokens. |
| `materializeVariable` | Materialize a workspace value by name or token. |
| `exportWorkspaceState` / `importWorkspaceState` | Persist or restore workspace replay payloads. |
| `setFusionPlanEnabled` / `fusionPlanForSource` | Emit or inspect fusion plan metadata. |
| `clearWorkspace` / `dispose` | Clear state and release host callbacks. |

The WASM wrapper temporarily moves the Rust session out of its `RefCell` around async operations. This avoids holding a mutable borrow across `await` while preserving a single logical session.

## Notebook Hosts

`runmat-core` does not define a notebook cell graph. Notebook-style hosts should keep their own cell model, source order, dirty state, execution history, and persistence. They should use the same `RunMatSession` when cells share a workspace, and they should provide meaningful source names so diagnostics and source identities remain useful.

Cell re-execution is host policy. The session only sees the source request it is asked to run and the workspace state already present in that session.

## Editor And LSP Hosts

Editor integrations should use session results for runtime state, not for static language facts. Static parse/lower/diagnostic features belong in the compiler and LSP layers. Runtime workspace snapshots are useful for variable panes, hover previews, and live plotting state after execution.

## Lifecycle Responsibilities

Hosts should:

- Serialize requests per session.
- Decide when to create, reset, or dispose sessions.
- Persist notebook cells, command history, and UI layout outside `runmat-core`.
- Use workspace deltas for routine updates and full snapshots when requested.
- Treat preview tokens as short-lived.
- Handle cancellation as cooperative.
- Keep source names stable enough for diagnostics and source-bound workspace keys.
