# Host Integration

The session API is intentionally host-neutral. Different frontends wrap it differently, but they all submit source through `ExecutionRequest` and consume structured results.

## Execution Profiling

Execution results report the command's measured wall duration, including admitted cleanup callbacks, as `profiling.total_ms`. This can exceed `execution_time_ms`, which measures the inner evaluator. The optional `cpu_ms`, `gpu_ms`, and `kernel_count` fields are unset when execution-attributed measurements are unavailable. Hosts should treat an unset field as unknown, not zero. Provider diagnostic counters are shared across commands and are not used to populate these fields; CPU duration cannot be inferred by subtracting GPU duration from elapsed wall time.

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

## Project Snapshots

Hosts can install a validated `FrozenProjectHandoff` so execution, source discovery, and analysis use the same project snapshot. On native targets, await `install_project_handoff(...)` and `clear_project_handoff()` and handle their results. Both retire loaded MEX modules before changing the project revision or artifact registrations. If that retirement fails, the existing project remains installed; the host must handle the failure before attempting another project change.

Browser project installation and clearing remain synchronous because the browser session does not load native MEX modules. The WASM wrapper reports project-operation errors to JavaScript without changing these methods into promises.

Core-created testing sessions use the caller's preference and user-path services, including the same application namespace. If either service is absent, the child leaves it absent rather than discovering another host account or creating an anonymous browser namespace. Testing sessions retain their own workspace, cancellation and search-path state.

## Lifecycle Responsibilities

Native library hosts own a session's local background-task driver. `execute_request` and `invoke_executable` automatically participate in that driver while foreground work is active. Between requests, a host that admits background work must keep `session.local_task_driver().drive_until(next_input_or_event)` polled so callbacks can progress while the session is idle. A completed foreground request does not itself keep polling the driver. Core-created nested testing sessions can share that driver while retaining separate task ownership; unrelated drivers cannot enter it recursively. Hosts must still serialize ordinary requests for one session.

The CLI REPL drives local work while waiting for the next input event. Finite CLI and standalone AOT hosts drive it through foreground execution. A host ending a session normally should await `close_gracefully()` so session-owned workspace values, admitted cleanup callbacks, MEX exit handlers, and foreign adapters retire before the callback driver closes. If closing foreign runtimes separately, await `shutdown_foreign_runtime()` while the driver and originating workspace services are still live; only then call `shutdown_local_tasks()`. `shutdown_local_tasks()`, session `Drop`, and browser `dispose` are forced disposal paths: they cancel remaining work and may abandon queued cleanup callbacks rather than replay them. Forced MEX disposal closes origin callback authority and does not report successful `mexAtExit` completion; arbitrary in-process native code cannot be guaranteed to finish on Drop. Browser sessions use the JavaScript event loop to drive callbacks between requests. These are host-lifetime obligations for the shared task service, not a claim that every timer lifecycle form has already migrated to background execution.

`RuntimeContext` by itself is a cloneable service handle, not a lifecycle owner. Lower-level embedders constructing one directly must install an exact `RuntimeLifecycleHost` with its matching local-task supervisor, drive that supervisor while callbacks can run, and gracefully close or explicitly force-close the host. Native MEX admission additionally requires that exact host-owned task port; the standalone MEX guard binds the same strong supervisor identity and must remain live through typed MEX retirement. Without a cleanup owner, `onCleanup` rejects a valid callback before allocating a cleanup object. Ordinary `RunMatSession`, standalone AOT, browser, and finite worker execution paths install their own host.

Hosts should:

- Serialize requests per session.
- Decide when to create, reset, or dispose sessions.
- Persist notebook cells, command history, and UI layout outside `runmat-core`.
- Use workspace deltas for routine updates and full snapshots when requested.
- Treat preview tokens as short-lived.
- Handle cancellation as cooperative.
- Keep source names stable enough for diagnostics and source-bound workspace keys.
