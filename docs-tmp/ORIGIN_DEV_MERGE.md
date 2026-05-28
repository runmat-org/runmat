# origin/dev Merge Recon (No Merge Performed)

## Snapshot
- Date: 2026-05-22
- Current branch: `v2-compiler-semantics`
- Current HEAD: `8fd2dc6a`
- `origin/dev`: `b7a717c3`
- Merge-base (`HEAD` vs `origin/dev`): `601596dca20111aeaf0e9c1060c2fe84fccf7639`
- Divergence (`HEAD...origin/dev`): `1293` commits on `HEAD` side, `140` commits on `origin/dev` side
- Status: **analysis only** (no merge/cherry-pick/rebase performed)

## High-Level Diff Shape
- Files changed since merge-base on `HEAD`: `382`
- Files changed since merge-base on `origin/dev`: `218`
- File-path intersection: `26`
- `origin/dev` change mix: `129` modified, `89` added
- `HEAD` change mix: `269` modified, `77` added, `36` deleted

## What origin/dev Is Mostly Changing
- Primary concentration is runtime builtin expansion work (`crates/runmat-runtime`, especially new builtin implementations + builtin JSON docs).
- Secondary concentration is website/docs content updates.
- Smaller but relevant changes in VM/parser/plot/accelerate/core glue.

Notable `origin/dev` builtin themes from commit/file scan:
- New or expanded builtins: `fill3`, `db`, `fgetl`, `step`, `impulse`, `tf`, `sinc`, `patch`, `integral`, `fminbnd`, `mode`, `repelem`, `complex`, `deg2rad/rad2deg/sind/cosd/tand`, `sawtooth/square`, `xline/yline`, `suptitle`, multiple image color conversion builtins.

## Intersections (Most Important for Merge Risk)
These are files touched by **both** `HEAD` and `origin/dev` since the common base.

### High-Risk Intersections
- `crates/runmat-core/src/session/run.rs`
  - HEAD delta is very large and includes semantic/runtime plumbing changes.
  - origin/dev also modifies this area.
- `crates/runmat-vm/src/interpreter/dispatch/mod.rs`
  - HEAD has large typed dispatch changes.
  - origin/dev has runtime behavior additions crossing dispatch boundaries.
- `crates/runmat-vm/src/object/resolve.rs`
  - Both sides changed callable/object resolution paths.
- `crates/runmat-turbine/src/compiler.rs`
- `crates/runmat-turbine/src/lib.rs`
- `crates/runmat-turbine/tests/jit.rs`
  - HEAD has major compiler/JIT movement; origin/dev has smaller but non-trivial edits.
- `crates/runmat-parser/src/parser/expr.rs`
  - Both sides changed parsing behavior and tests.
- `crates/runmat-runtime/src/builtins/math/optim/fzero.rs`
  - Both sides modify numeric/optimizer semantics.

### Medium-Risk Intersections
- `crates/runmat-accelerate-api/src/lib.rs`
- `crates/runmat-accelerate/src/backend/wgpu/provider_impl.rs`
- `crates/runmat-accelerate/src/simple_provider.rs`
- `crates/runmat-parser/src/parser/command.rs`
- `crates/runmat-parser/src/parser/stmt.rs`
- `crates/runmat-runtime/src/builtins/io/type_resolvers.rs`
- `crates/runmat-runtime/src/builtins/mod.rs`
- `crates/runmat-vm/src/interpreter/api.rs`
- `crates/runmat-vm/src/lib.rs`
- `crates/runmat-vm/tests/basics.rs`
- `crates/runmat-vm/tests/closures.rs`

### Lower-Risk Intersections
- `crates/runmat-runtime/src/builtins/math/signal/hann.rs`
- `crates/runmat-runtime/src/builtins/math/signal/hamming.rs`
- `crates/runmat-runtime/src/builtins/math/signal/blackman.rs`
- `crates/runmat-core/tests/command_controls.rs`
- `crates/runmat-core/tests/integration.rs`
- `crates/runmat-parser/tests/command_syntax.rs`
- `crates/runmat-parser/tests/parser.rs`

## Important Non-Intersection Observations
- `HEAD` removed `crates/runmat-kernel/*`; `origin/dev` does **not** touch kernel paths directly.
  - This is good: low chance of direct file-level conflicts on kernel removal.
- `origin/dev` does not modify workspace manifests (`Cargo.toml` / `Cargo.lock`) in this range.

## Proposed Merge Plan (No Merge Yet)

### Phase 0: Safety + Prep
1. Create a dedicated integration branch off current `HEAD` for merge rehearsal.
2. Persist this recon note and keep a running conflict ledger (`docs-tmp/ORIGIN_DEV_CONFLICTS.md`) during merge trial.
3. Pin baseline test commands before merge trial:
   - `cargo test -p runmat-runtime --lib`
   - `cargo test -p runmat-vm --test functions`
   - `cargo test -p runmat-vm --test closures`
   - `cargo test -p runmat-core --lib`
   - optionally full elevated workspace sweep later

### Phase 1: Reconcile Core Semantic/Dispatch Intersections First
1. Resolve `session/run`, parser (`expr/command/stmt`), VM dispatch/object resolve, and turbine compiler/lib intersections.
2. Explicitly preserve typed callable identity, expansion semantics, and provider routing invariants from current branch.
3. Only after these compile/behavior invariants hold, proceed to builtin additions.

### Phase 2: Bring origin/dev Builtins in Domain Chunks
1. Integrate control/optim/signal additions (`step`, `impulse`, `tf`, `fminbnd`, `integral`, `mode`, `sinc`, etc.).
2. Integrate plotting/image/shape additions (`fill3`, `patch`, `xline/yline`, color conversions, `repelem`).
3. Reconcile any runtime builtin registry/doc JSON ordering or naming conflicts.

### Phase 3: Test Reconciliation and Identifier Consistency
1. Reconcile any tests expecting legacy identifiers/messages where typed/semantic behavior now differs.
2. Re-run targeted suites after each chunk.
3. Run full elevated workspace once all conflict buckets are resolved.

### Phase 4: Website/Docs Tail (Optional Separate PR)
1. Consider splitting website/docs-only changes from runtime/compiler merge for cleaner review surface.
2. Keep code merge focused on runtime/compiler correctness first.

## Merge Rules to Hold
- No semantic regressions in typed dispatch/callable identity.
- No regressions in expansion behavior (`deal` single-output cell behavior stays intact).
- No regressions in provider routing / gather behavior under parallel tests.
- Preserve current kernel-crate removal unless intentionally revisited.