# RunMat Function Manager (`runmatfunc`)

RunMat Function Manager is a CLI/TUI companion for the RunMat project. It assembles context packs
for builtin authoring, surfaces metadata about existing builtins, emits documentation bundles for
the website, and will orchestrate Codex-based authoring sessions.

## Project Goals

- **Metadata discovery** – enumerate builtin functions, GPU/fusion specs, docs, and tests directly
  from the runtime inventories.
- **Doc emission** – produce a structured JSON + TypeScript schema consumed by the Next.js site.
- **Authoring workflow** – assemble prompts, code snippets, and testing instructions and feed them
  to Codex (interactive or headless).
- **Job automation** – queue batch rewrites or validations (nightly runs).

## Layout Overview

```
src/
  main.rs           # entry point, defers to lib
  lib.rs            # exports run(), wires submodules
  cli/              # clap argument parsing + command dispatch
  app/              # high-level state + actions (context, docs, builtin runs)
  context/          # metadata gathering, prompt rendering, snippet discovery
  codex/            # Codex client integration (placeholder)
  builtin/          # inventory lookup, metadata normalization, templates
  workspace/        # filesystem helpers, diff viewer, test execution (todo)
  jobs/             # queue + scheduler for batch tasks (todo)
  tui/              # terminal UI (list of builtins with keyboard navigation)
  logging.rs        # tracing setup
  errors.rs         # shared error types
```

Implemented pieces:
- `context::manifest/serialize/prompt/gather` – build `AuthoringContext` (prompt + source hints) and emit docs.
- `builtin::inventory` – collect runtime metadata into a normalized manifest.
- CLI commands for manifest/doc emission/builtin summaries and a simple TUI browser.

## Suggested Implementation Order

1. **Logging & CLI** ✅ (`runmatfunc --help`, verbose flag, tracing).
2. **App context/config** ✅ (`AppContext::new`, CLI command wiring).
3. **Builtin inventory module** ✅ (manifest listing + metadata normalization).
4. **Context manifest & doc emission** ✅ (writes `builtins.json` + `builtins.d.ts`).
5. **Workspace + tests** ✅ (read/write helpers, diff preview, targeted `cargo test`).
6. **TUI scaffolding** ✅ (list + detail pane, actions with shortcuts).
7. **Codex integration** ⬜ (stub client in place; enable `embedded-codex` to link codex-rs).
8. **Jobs queue/scheduler** ✅ (batch/headless workflows).

## Run Order Checklist

- [x] Logging module ready (`logging::init`)
- [x] CLI arguments + command dispatch
- [x] AppContext load + placeholder/builtin actions
- [x] Builtin inventory discovery (manifest output)
- [x] Doc emitter writing JSON + d.ts
- [x] Workspace helpers for diffs/tests
- [x] TUI browse view (basic list/detail)
- [ ] Codex client integration
- [x] Job queue + scheduler

## Usage (current)

```
runmatfunc --help                   # view CLI options
runmatfunc manifest                 # print builtin metadata summary
runmatfunc docs --out-dir docs/generated  # emit docs bundle
runmatfunc builtin sin --codex      # assemble authoring context for builtin 'sin' (attempt Codex)
runmatfunc browse                   # interactive TUI (↑/↓ navigate, t run tests, d emit docs, q quit)
runmatfunc builtin sin --diff       # show git diff for builtin-related files alongside context
runmatfunc queue add sin --codex    # enqueue a headless job (stored under artifacts/runmatfunc/queue.json)
runmatfunc queue run                # run queued jobs headlessly (writes transcripts + test logs)
runmatfunc queue list               # inspect queued jobs and their target Codex models
```

> **Note:** Codex execution currently uses a stub client unless the
> `embedded-codex` feature is enabled and the [codex-rs](https://github.com/openai/codex) workspace
> is available. With that feature active (`cargo run -p runmatfunc --features embedded-codex -- builtin sin --codex`),
> the tool will link against `codex-core` directly. The CLI and TUI surface Codex availability so
> you always know whether authoring sessions will call the real client or the stub.

## Configuration

`runmatfunc` loads configuration from `~/.runmatfunc/config.toml` (override with
`RUNMATFUNC_CONFIG`). Environment variables take precedence over file values:

- `RUNMATFUNC_DEFAULT_MODEL` – default Codex model when `--model` is omitted.
- `RUNMATFUNC_ARTIFACTS_DIR` – base directory for logs, transcripts, and queue state.
- `RUNMATFUNC_DOCS_OUTPUT_DIR` – default target for `runmatfunc docs`.
- `RUNMATFUNC_SNIPPET_INCLUDE` / `RUNMATFUNC_SNIPPET_EXCLUDE` – extra glob patterns (comma/semicolon
  separated) merged into snippet discovery.
- `RUNMATFUNC_ENABLE_OPTIONAL_FEATURES=1` – enable optional Cargo features (for example
  `blas-lapack`) required by a builtin’s test plan.

The artifacts directory now contains:

- `tests/<builtin>/` – captured stdout/stderr for targeted cargo test runs.
- `transcripts/` – JSON transcripts emitted by headless queue runs.
- `queue.json` – persisted job queue state.

## Contributing

- Follow Rust edition 2021, keep modules loosely coupled.
- Keep this README updated as modules graduate from stubs to real implementations.
- Prefer async-friendly interfaces in long-running actions (Tokio runtime already available).
- Add tests in `tests/` or module `#[cfg(test)]` blocks; use `insta` snapshots for CLI output when helpful.
- When adding new functionality, update the checklist above so future authors know the remaining work.
