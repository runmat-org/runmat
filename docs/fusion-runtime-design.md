# Fusion Runtime Execution Plan

This note now records the live execution contract between the Ignition
interpreter, the fusion planner, and the GPU providers. It captures the
original motivation for dynamic WGSL generation and lazy gather, and then
documents the residency hooks and integration tests that were added while
bringing the feature online. The goals remain:

1. Understand the interpreter interactions around a representative trace.
2. Spell out the execution contract across Ignition, the planner, and the
   runtime executor.
3. Capture the residency bookkeeping that keeps GPU tensors live until a
   real sink gathers them.
4. Reference the integration tests that keep the contract honest.

## 1. Representative Trace

We use a trimmed expression (`y = sin(x) .* x + b;`) that the fusion pass
currently recognises as a single elementwise chain. The relevant bytecode
sequence (comments show the stack top on the right) is expected to look like:

```
LoadVar x          ; [x]
CallBuiltin "sin"  ; [sin(x)]
LoadVar x          ; [sin(x), x]
ElemMul            ; [sin(x) .* x]
LoadVar b          ; [..., b]
Add                ; [(sin(x) .* x) + b]
StoreVar y         ; []          # value written back into vars[y]
```

What happens at each step inside `vm.rs`:

- `LoadVar` (`vm.rs:192`): reads `vars[idx]`, pushes a clonable `Value`.
- `CallBuiltin` (`vm.rs:1052`): pops arguments, calls
  `accel_prepare_args` which already triggers promotion (and soon fusions).
- `ElemMul` (`vm.rs:627`): pops the two operands, calls
  `accel_promote_binary(Elementwise, ...)`, then pushes the runtime result.
- `Add` (`vm.rs:437`): same pattern for addition.
- `StoreVar` (`vm.rs:190`): pops the result and writes into `vars[idx]`.

For fused execution we therefore need to capture:

1. The operand `Value`s that would otherwise be popped for each primitive.
2. The `Value` slot that will receive the final result (`StoreVar`, or
   whatever instruction follows the chain).
3. Any temporaries that must stay resident on the GPU after the fusion block.

Because Ignition materialises every intermediate on the stack, the fusion
runner must:

- Pull inputs from a mix of stack values and variables before the first
  primitive executes.
- Replace the whole chain by a single push of the fused result before the
  interpreter reaches the subsequent instruction (`StoreVar` in this case).

## 2. Execution Contract for Fused Kernels

To make the interpreter/accelerator boundary explicit, the proposal is to add
one more layer on top of `FusionPlan`:

### 2.1. Interpreter responsibilities

1. **Activation:** inside `try_execute_fusion_group` the interpreter checks
   whether the current PC matches a planned fusion span. When it does, the
   normal instruction loop is paused and control is handed to the fusion
   executor.

2. **Input capture:** `try_execute_fusion_group` gathers concrete `Value`s
   from the stack, locals, and captured vars according to
   `FusionGroupPlan::inputs`. The stack is left untouched until the executor
   confirms success; the helper tracks the number of slots that will be
   consumed.

3. **Dispatch:** the interpreter calls into `runmat_accelerate::execute_fusion`
   with the plan plus the captured inputs. On success it drops the consumed
   stack entries, pushes the fused result, and advances the PC to the end of
   the span. On failure it falls back to the unfused instruction stream and
   logs the reason under `log::debug`.

4. **Residency clearing on stores:** every instruction that mutates a variable
   or cell/table slot now calls `runmat_accelerate_api::clear_residency` so the
   residency table stays in sync with host data.

### 2.2. Fusion executor responsibilities

1. **GPU intent resolution:** the executor records each `Value` as a
   `PreparedInput`, reusing existing `Value::GpuTensor` handles when possible
   and uploading scalars (constants) as 1-element tensors when needed.

2. **Kernel selection:** build (or reuse from cache) the WGSL module emitted
   by `FusionKernelSpec::new`. The template will be expanded into a full
   compute shader by declaring storage buffers for each distinct tensor, plus
   uniforms describing length/shape.

3. **Launch description:** compute the output shape using the current
   broadcast/unification rules (`ShapeInfo`). Record:
   - Workgroup counts (`dispatch_size`) based on element count.
   - The mapping from `FusionOp` IDs to WGSL temporaries so that the final
     value can be written to a dedicated output buffer.

4. **Execution:** use the WGPU provider to
   - Materialise input buffers (upload if needed).
   - Create an output buffer (reusing buffers when the result overwrites one
     of the inputs).
   - Submit the compute pass and wait for completion.

5. **Result packaging:** on success return a new `Value::GpuTensor`, mark it
   as resident in the global table, and hand it back to the interpreter. On
   failure (shader compilation or dispatch error), bubble up an error so the
   interpreter can fall back to the CPU path.

### 2.3. Residency / Lazy Gather

Residency tracking now lives in `runmat_accelerate::fusion_residency`, a
`Lazy<Mutex<HashSet<u64>>>` keyed by `GpuTensorHandle::buffer_id`.

- **Marking:** `fusion_exec::execute` marks every fused result when the GPU
  provider returns a handle. Providers that upload constants on the fly mark
  those temporary buffers as well.
- **Clearing on gather:** `runmat_runtime::dispatcher::gather_if_needed` and
  the auto-offload gather helpers call
  `runmat_accelerate_api::clear_residency` before copying data back to the
  host. This prevents stale residency bits after a host conversion.
- **Interpreter writes:** all Ignition store instructions (e.g., `StoreVar`,
  `StoreSlice`, `StoreIndex`) also invoke the clear hook so mirrored host data
  cannot remain incorrectly marked resident.
- **Hook registration:** the first fusion execution registers the clear hook
  through `ensure_residency_hooks`, allowing third-party providers to supply
  their own clearing logic if they cache GPU buffers.

Together these rules ensure that tensors stay device-resident while fused
chains remain on the GPU, yet gather operations and host stores promptly free
or recycle the buffers.

## 3. Integration & Test Coverage

- **`crates/runmat-ignition/tests/fusion_gpu.rs`** drives a full interpreter
  run with a deterministic test provider. It verifies that fused chains push
  the correct handle onto the stack, that residency is marked, and that
  `gather_if_needed` clears residency when a sink demands host data.
- **GPU-optional environments:** plotting and kernel tests now skip when the
  sandbox lacks a GPU adapter or blocks socket creation. The fusion test uses
  an in-process provider so it still runs in those environments, ensuring
  residency bookkeeping remains exercised on CI.

Future tests should extend this coverage with duplicate inputs, constant-only
chains, and additional builtin operations to match the expanded fusion
capabilities.

---

With this contract in place, the remaining work is mechanical: thread the
executor into the interpreter loop, add WGSL emission that replaces the
current string template with a real shader, and extend the auto-offload
planner/runtime to consult the residency table before calling `gather_if_needed`.
