# Multi-output builtin handling in the VM

The current VM has ad-hoc branches inside `Instr::CallBuiltin` for builtins that may return
multiple outputs (e.g. `union`, `histcounts`, `histcounts2`, now `hist`). Each branch:

- directly invokes the runtime helper (`runmat_runtime::builtins::…::evaluate`),
- manually inspects `out_count` in order to push some or all outputs back onto the stack,
- runs any side effects (plot rendering, registry updates) exactly once, and
- fills unused requested outputs with zeros.

## Known limitations / debt

1. **No general multi-output protocol.** Builtins that require special handling must add new
   branches to the VM, which is brittle and easy to forget. There is no shared trait or helper the
   VM can call to “get N outputs”.
2. **Side-effect coupling.** Plotting builtins such as `hist` need to render regardless of
   `nargout`. Today we embed that knowledge in the VM branch itself. A general mechanism should let
   the builtin declare its side effects and outputs without editing the VM.
3. **Error propagation duplication.** Every special-case block repeats the `match { Ok(eval) … }`
   pattern. A shared abstraction would eliminate this repetition and reduce the chance we miss a
   future builtin.

## Current workaround

Until we rework the calling convention, we keep these bespoke branches documented and grouped in
`vm.rs`. When a builtin needs multi-output behaviour, we:

1. Expose an `evaluate` helper in the runtime crate that returns a struct encapsulating all outputs.
2. Add a VM branch that:
   - invokes `evaluate` once,
   - performs any required side effects (e.g. `eval.render_plot()`), and
   - pushes the appropriate number of outputs based on `out_count`.

This keeps behaviour correct but is not the final solution.

## Desired future direction

- Introduce a general `MultiOutputBuiltin` trait or metadata table that advertises how many outputs
  a builtin may produce and provides a uniform “evaluate into Vec<Value>” API.
- Move the “fill extra outputs with zeros” logic into a helper shared by all multi-output calls.
- Let plotting builtins register side-effect callbacks with the plotting subsystem instead of doing
  so from the VM branch.

When we have time to clean this up, migrate the existing branches (`union`, `histcounts`,
`histcounts2`, `hist`, etc.) to the new mechanism and delete this TODO file.
