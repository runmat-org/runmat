# Idioms in the RunMat Compiler

This directory contains the compiler side of VM idioms.

An idiom is a rooted source pattern that the compiler can recognize and lower to a dedicated VM opcode instead of lowering it as a more generic sequence of statements and expressions.

The split is:

- compile-time idiom detection and lowering live under `src/compiler/idioms/`
- runtime idiom execution helpers live under `src/accel/idioms/`

That split is intentional. Detection is a compiler concern. Execution is a VM/runtime concern.

# How it works

The compiler idiom pipeline is:

1. Statement lowering starts from a rooted `HirStmt`.
2. `try_lower_stmt_idiom(...)` asks whether that statement matches any supported idiom.
3. Each idiom detector proves its own rooted pattern.
4. If a match succeeds, the compiler lowers directly to a dedicated VM opcode.
5. Normal statement lowering is skipped for that statement.

The public compiler entrypoint is `try_lower_stmt_idiom(...)` in `src/compiler/idioms/mod.rs`.

This is intentionally generic at the statement boundary. The idiom framework does not encode separate public APIs for loop idioms, assignment idioms, or expression idioms. Each idiom decides internally what kind of rooted statement it can start from.

That keeps the architecture simple:

- the framework is statement-rooted
- the individual idiom owns the pattern details

# Adding a new idiom

For a new idiom, the usual steps are:

1. Add a new detector/lowerer in `src/compiler/idioms/`.
2. Match from `&HirStmt`, not from a special-case public hook for one statement kind.
3. Prove the pattern conservatively.
4. Lower to a dedicated opcode.
5. Add the runtime execution helper under `src/accel/idioms/`.
6. Add tests for:
   - positive detection
   - rejection of near-miss patterns
   - correct lowering shape
   - end-to-end runtime behavior

Good compiler idioms tend to have these properties:

- the rooted source pattern is stable and unambiguous
- lowering to a dedicated opcode preserves existing semantics
- there is a meaningful benefit from specializing execution
- the opcode contract is small and deterministic

If a transformation is hard to prove from HIR, or does not need a dedicated opcode, it probably belongs in a different optimization layer instead of the idiom system.

# Current idioms implemented

## `stochastic_evolution`

Current status:

- compiler detection/lowering lives in `src/compiler/idioms/stochastic_evolution.rs`
- runtime execution lives in `src/accel/idioms/stochastic_evolution.rs`
- lowering targets `Instr::StochasticEvolution`

This idiom is rooted at a `HirStmt::For`, which is the statement boundary for the idiom pattern scan. The detector proves a specific computation graph pattern and then lowers the whole statement directly to one opcode.
