# Execution Request Source Modes

## Goal

Keep a single execution API surface while making both snippet execution and project/file execution first-class.

## Final request shape

Hosts call one method: `executeRequest(request)`.

`request.source` is a discriminated union:

- text mode (REPL/snippet/cell)
  - `{ kind: "text", name: string, text: string }`
- path mode (filesystem-backed execution)
  - `{ kind: "path", path: string }`

All other request fields (`compatibility`, `hostPolicy`, `requestedOutputs`, workspace handle) apply identically in either mode.

## Semantics

- `kind: "text"`
  - source content comes from request payload directly
  - intended for REPL cells, editor snippets, generated code, and tests
  - `name` is required for diagnostics/source identity
- `kind: "path"`
  - source content is resolved/read through the active filesystem provider
  - intended for project scripts/modules and in-memory filesystem app flows
  - enables wasm hosts to run project files from in-memory/indexeddb/remote providers

## Why this is the resting state

- One API entrypoint, no competing execution method names.
- No ambient hidden source model.
- No loss of REPL ergonomics.
- Correct abstraction boundary: host provides source *kind* explicitly; runtime/compiler own execution semantics.

## Notes for wasm + TS

- wasm must support both source kinds through `executeRequest`.
- TS bindings expose both source variants in the request type.
- Convenience helpers should not add alternate execution methods; they may only build one of the two source variants.
