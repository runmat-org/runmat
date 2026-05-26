## Progress Tracker

| Wave | Builtins | Status | Commit | Validation | Notes |
|---|---|---|---|---|---|
| Wave 0 (Reference) | `zeros`, `linspace` | Done | _fill_ | `runmat-builtins`, `runmat-lsp`, targeted runtime tests | Reference implementation |
| Wave 1 (Array Constructors) | `ones`, `eye` | Done | `0dec5d88` | `cargo fmt`; `cargo test -p runmat-builtins`; `cargo test -p runmat-lsp`; targeted runtime filters `ones_`, `eye_` | Attached descriptors + LSP descriptor tests |
| Wave 2 (Array Constructors + Reduction) | `true`, `false`, `range`, `logspace` | Done | `0dec5d88` | `cargo fmt`; `cargo test -p runmat-builtins`; `cargo test -p runmat-lsp`; targeted runtime filters `true_`, `false_`, `range_`, `logspace_` | Attached descriptors with exhaustive signatures aligned to runtime parser/branch behavior |
| Wave 3 (Random Constructors) | `rand`, `randn`, `randi`, `randperm` | Done | `66c6c0c3` | `cargo fmt`; `cargo test -p runmat-builtins`; `cargo test -p runmat-lsp`; targeted runtime filters `rand_`, `randn_`, `randi_`, `randperm_` | Attached descriptors + LSP signature-help coverage for random constructor family |
| Wave 4 (Special Constructors) | `empty`, `magic` | Done | `68d72b80` | `cargo fmt`; `cargo test -p runmat-builtins`; `cargo test -p runmat-lsp`; targeted runtime filters `empty_`, `magic_` | Attached descriptors + LSP signature-help coverage for constructor edge cases |

## Remaining Work

- Total registered builtins: `568`
- Migrated with attached descriptor: `26`
- Remaining: `542`
