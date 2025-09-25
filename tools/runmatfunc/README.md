# RunMat Function Manager (`runmatfunc`)

RunMat Function Manager is a CLI/TUI companion for the RunMat project. It assembles context packs
for builtin authoring, surfaces metadata about existing builtins, emits documentation bundles for
the website, and eventually orchestrates Codex-based authoring sessions.

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
  context/          # metadata gathering and prompt assembly
  codex/            # Codex client integration (placeholder)
  builtin/          # inventory lookup, metadata normalization, templates
  workspace/        # filesystem helpers, diff viewer, test execution
  jobs/             # queue + scheduler for batch tasks
  tui/              # terminal UI components (layout, widgets, input handling)
  logging.rs        # tracing setup
  errors.rs         # shared error types
```

Each module is currently a stub with TODO markers. Implementation will proceed in stages.

## Suggested Implementation Order

1. **Logging & CLI**: ensure `runmatfunc --help` and verbose logging work end-to-end.
2. **App context/config**: load configuration, wire CLI commands to placeholder actions.
3. **Builtin inventory module**: fetch metadata from `runmat-builtins` and `runmat-accelerate` inventories.
4. **Context manifest & doc emission**: render metadata to JSON + d.ts (for website integration).
5. **Workspace + tests**: scripting for repo interactions (diff, targeted cargo test).
6. **TUI scaffolding**: basic list + detail view of builtins using `ratatui`.
7. **Codex integration**: embed `codex-rs` or call `codex exec` to run authoring sessions.
8. **Jobs queue/scheduler**: support batch processing (headless) for overnight runs.

## Run Order Checklist

- [ ] Logging module ready (`logging::init`)
- [ ] CLI arguments + command dispatch
- [ ] AppContext load + placeholder actions
- [ ] Builtin inventory discovery (manifest output)
- [ ] Doc emitter writing JSON + d.ts
- [ ] Workspace helpers for diffs/tests
- [ ] TUI browse view (basic list/detail)
- [ ] Codex client integration
- [ ] Job queue + scheduler

## Usage (planned)

```
runmatfunc manifest            # print builtin metadata summary
runmatfunc docs emit           # write docs/generated/builtins.json/.d.ts
runmatfunc builtin sin         # run headless authoring job for `sin`
runmatfunc browse              # enter interactive TUI
```

## Contributing

- Follow Rust edition 2021, keep modules loosely coupled.
- Document each new module with a short summary.
- Prefer async-friendly interfaces in long-running actions (Tokio runtime already available).
- Write tests in `tests/` or module-level `#[cfg(test)]` blocks; use `insta` snapshots for CLI output when helpful.

