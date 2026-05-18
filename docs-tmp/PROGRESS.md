# Progress

## Current Focus

Broad consumer migration and compatibility-surface cleanup, while keeping semantic pipeline validation green.

## Plan Status Snapshot

- Plan 0: semantic HIR type model in place.
- Plan 1: semantic lowering in place.
- Plan 2: MIR + analysis layer in place.
- Plan 3: downstream consumer migration in progress.
- Plan 4: MATLAB core semantic coverage in progress.
- Plan 5: manifest/composition closeout audit still needed.
- Plan 6: nominal class/builtin metadata unification in progress.
- Plan 7: semantic-fact-driven accel/fusion closeout audit still needed.

## Recent Landed Slices

- `a5fac002` RM-378: trim builtin compatibility
  - Removed dead `BuiltinCompatibility::RunMatExtended` variant to keep policy surfaces behavior-backed.
  - Validation: `cargo test -p runmat-builtins`, `cargo fmt --all --check`, `cargo test -p runmat-core --test semicolon_suppression`, `cargo check --workspace`, `git diff --check`.

- (uncommitted in this log entry) semantic test migration in `runmat-gc`
  - Replaced legacy stress-test usage of removed APIs (`HirProgram`, `runmat_vm::execute`) with semantic pipeline (`lowering.assembly -> runmat_mir::lower_assembly -> runmat_vm::compile -> runmat_vm::interpret`).
  - Added `runmat-mir` dev dependency for `runmat-gc`.
  - Validation: `cargo test -p runmat-gc --tests` green.

## Next Resolution Items

- Finish converting remaining legacy test/doc references that imply removed APIs where they block semantic-only confidence.
- Audit Plan 5 deliverables against concrete project/manifest acceptance criteria and capture pass/fail evidence.
- Audit Plan 7 deliverables against semantic-fact-driven fusion criteria and capture pass/fail evidence.
