# Idioms in the RunMat VM

This directory contains the runtime side of VM idioms.

An idiom is a rooted source pattern that the compiler can recognize and lower to a dedicated VM opcode instead of lowering it as a more generic sequence of statements and expressions.

The split is:

- compile-time idiom detection and lowering live under `src/compiler/idioms/`
- runtime idiom execution helpers live under `src/accel/idioms/`

That split is intentional. Detection is a compiler concern. Execution is a VM/runtime concern.

The Accelerate API may provide optimized implementations for idioms. In this repo, that usually means:

- a GPU/provider fast path when acceleration is available
- a host fallback when it is not
- one dedicated VM opcode as the contract between compiler and runtime

# How it works

The idiom pipeline is:

1. The compiler sees a rooted statement.
2. `src/compiler/idioms/mod.rs` asks whether that statement matches any supported idiom.
3. If it matches, the compiler lowers it to a dedicated opcode.
4. The interpreter executes that opcode.
5. The runtime helper in `src/accel/idioms/` performs the operation, optionally using an Accelerate provider fast path first and falling back to host execution if needed.

Today the compiler entrypoint is `try_lower_stmt_idiom(...)`, which means idiom matching starts from a statement boundary and each idiom is responsible for proving its own rooted pattern.

This directory does not own pattern detection. It owns runtime execution support for idioms that already survived compiler lowering.

# Adding a new idiom

For a new idiom, the usual steps are:

1. Add compiler-side detection in `src/compiler/idioms/`.
2. Define or reuse a dedicated VM opcode.
3. Lower the matched pattern to that opcode.
4. Add or extend the runtime helper under `src/accel/idioms/`.
5. Prefer a provider fast path plus a host fallback.
6. Add tests for:
   - pattern detection/lowering
   - interpreter/runtime behavior
   - accelerated and fallback behavior where applicable

Good idioms tend to have these properties:

- the source pattern is stable and unambiguous
- lowering to a dedicated opcode preserves existing semantics
- there is a meaningful runtime benefit from specialized execution
- the runtime contract is small and deterministic

If an optimization does not need a dedicated opcode or has unclear source-level semantics, it probably belongs in a different optimization layer instead of the idiom system.

# Current idioms implemented

## `stochastic_evolution`

Current status:

- compiler detection/lowering lives in `src/compiler/idioms/stochastic_evolution.rs`
- runtime execution lives in `src/accel/idioms/stochastic_evolution.rs`
- lowering targets `Instr::StochasticEvolution`

This idiom matches a specific computation graph pattern and replaces it with one opcode. At runtime, the helper tries an accelerated implementation first when available and otherwise executes the host implementation.
