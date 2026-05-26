## Progress Tracker

| Wave | Builtins | Status | Commit | Validation | Notes |
|---|---|---|---|---|---|
| Wave 0 (Reference) | `zeros`, `linspace` | Done | _fill_ | `runmat-builtins`, `runmat-lsp`, targeted runtime tests | Reference implementation |
| Wave 1 (Array Constructors) | `ones`, `eye` | Done | _fill_ | `cargo fmt`; `cargo test -p runmat-builtins`; `cargo test -p runmat-lsp`; targeted runtime filters `ones_`, `eye_` | Attached descriptors + LSP descriptor tests |
| Wave 2 (Array Constructors + Reduction) | `true`, `false`, `range`, `logspace` | Done | _fill_ | `cargo fmt`; `cargo test -p runmat-builtins`; `cargo test -p runmat-lsp`; targeted runtime filters `true_`, `false_`, `range_`, `logspace_` | Attached descriptors with exhaustive signatures aligned to runtime parser/branch behavior |

## Remaining Work

- Total registered builtins: `568`
- Migrated with attached descriptor: `20`
- Remaining: `548`
