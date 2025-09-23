# Fusion Runtime Execution Plan

This note captures the working assumptions we need before wiring dynamic
WGSL generation and lazy gather into the runtime. It contains two things:

1. A representative bytecode trace for a chain that our fusion pass already
   groups, so we can reason about stack/variable interactions.
2. A proposed execution contract between the Ignition interpreter, the
   fusion planner, and the WGPU backend once we begin emitting kernels.

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

1. **Activation** (already partially implemented): before each instruction,
   look up the active fusion group via `set_current_pc`. When the PC matches
   the group's `span.start`, suspend normal instruction execution and hand
   control to the fusion executor.

2. **Input capture:** collect the concrete `Value`s needed by the group.
   - Use the group operations to determine how many stack pops would have
     happened. For the example chain we need the top two stack values and
     `vars[b_idx]`.
   - Do *not* actually mutate the stack yet; we clone the inputs and keep a
     record of how many slots must be consumed on success.

3. **Dispatch:** call a new helper (e.g. `execute_fused_group(plan, ctx)`) and
   pass:
   - The `FusionGroupPlan`.
   - The collected input `Value`s (still in host form).
   - Mutable access to the stack and the `vars` array so the executor can
     install results.

4. **Post-run cleanup:** if the executor succeeds, drop the specified number
   of stack values (since the fused kernel replaces them) and push the fused
   result. If it fails, fall back to the original instruction stream by
   returning `false` and letting the interpreter resume instruction-by-
   instruction execution.

### 2.2. Fusion executor responsibilities

1. **GPU intent resolution:** for each input `Value`, decide whether we can
   reuse an existing `Value::GpuTensor` handle or must promote via
   `NativeAutoOffload::tensor_to_gpu`.

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

5. **Result packaging:** on success return a new `Value::GpuTensor`. The
   interpreter will store it into the target variable or push it onto the
   stack. On failure (shader compilation or dispatch error), bubble up an
   error so the interpreter can fall back to sober execution.

### 2.3. Residency / Lazy Gather

The executor is also the right place to attach residency metadata:

- Each GPU-backed `Value` acquires a `FusionResidency` tag: either owned by
  the current fused group, or borrowed from an upstream kernel. The tag keeps
  a reference count for downstream consumers.
- When `prepare_builtin_args` or other interpreter paths encounter a GPU
  value that still has outstanding fused uses, they skip `gather_if_needed`
  unless the called builtin is marked as a sink.
- Once the reference count reaches zero (e.g., after the fused result is
  finally gathered or stored), the handle returns to normal behaviour.

This metadata can be a lightweight side table keyed by `buffer_id`, updated
when the fusion executor produces new handles.

---

With this contract in place, the remaining work is mechanical: thread the
executor into the interpreter loop, add WGSL emission that replaces the
current string template with a real shader, and extend the auto-offload
planner/runtime to consult the residency table before calling `gather_if_needed`.
