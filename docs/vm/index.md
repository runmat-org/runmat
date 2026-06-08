---
title: "Virtual Machine (VM)"
category: "Virtual Machine (VM)"
section: "3.0"
last_updated: "May 28, 2026"
---

# VM Interpreter & Bytecode

The `runmat-vm` crate provides the core execution engine for RunMat. It defines the custom bytecode format, the compiler that lowers Mid-Level IR (MIR) into that format, and the asynchronous interpreter that executes it. The VM is designed as the middle tier of a tiered execution model, sitting between high-level semantic analysis and low-level JIT or GPU acceleration.

## Bytecode Compilation (MIR → Bytecode)

The `Compiler` struct is responsible for the final lowering of `MirAssembly` into a sequence of `Instr` opcodes This process includes:

- Instruction Lowering: Converting MIR statements (`MirStmt`) and terminators (`MirTerminatorKind`) into stack-based or register-like bytecode instructions
- Fast-Path Detection: The compiler identifies specific patterns, such as "Stochastic Evolution," to enable specialized execution paths
- Fusion Metadata: If the `native-accel` feature is enabled, the compiler generates `FusionMetadata` to assist the GPU fusion engine in identifying groups of instructions that can be offloaded

For details on the lowering logic and jump patching, see [Bytecode Compilation (MIR → Bytecode)](/docs/runtime/vm/bytecode).

---

## Interpreter Dispatch & Execution Loop

The interpreter is an asynchronous loop that processes instructions using a `dispatch_instruction` table. Execution state is maintained within an `ExecutionContext`, which tracks the program counter (`pc`), the value stack, and local variables.

- Modular Handlers: Dispatch is divided into specialized sub-modules for arithmetic, array construction, control flow, and exception handling
- Try/Catch Stack: The VM maintains a `try_stack` to manage MATLAB-compatible `try...catch` blocks and exception redirection
- Runtime Values: The stack and local slots carry `runmat_builtins::Value`; see [Runtime Values & Type Model](/docs/runtime/values) for the value families and host metadata model.

For details on the execution loop and instruction semantics, see [Interpreter Dispatch & Execution Loop](/docs/runtime/vm/interpreter).

---

## Indexing Subsystem

MATLAB's indexing semantics (paren `()`, brace `{}`, and dot `.` indexing) are complex, supporting linear, logical, and multidimensional slice operations. The VM uses an `IndexPlan` to normalize these operations before execution.

- Slice Operations: Handles `A(2:end-1)` using `IndexSliceExpr` and `StoreSliceExpr`
- End Evaluation: Provides specialized logic for the `end` keyword within indexing expressions
- Object Dispatch: If the base value is a class instance, the VM dispatches to `subsref` or `subsasgn` methods

For details on index planning and slice materialization, see [Indexing Subsystem](/docs/runtime/vm/indexing).

---

## Callable Resolution & Function Dispatch

The VM handles multiple types of callables through the `CallableIdentity` enum, including built-ins, anonymous functions, and class methods.

- Resolution Protocol: The VM resolves names to `CallableDescriptor` objects, which determine if a call is a direct host-call, a bytecode jump, or a dynamic `feval` dispatch.
- Closure Capture: Supports lexical closures by mapping `Value::Closure` captures to the function's local variable slots.
- Semantic Hooks: Integrates with the `runmat-runtime` layer for calling built-in functions with variable input/output counts (`nargin`/`nargout`).

For details on the calling convention and method dispatch, see [Callable Resolution & Function Dispatch](/docs/runtime/vm/dispatch).
---

## Data Structures & Layout

The VM relies on several key structures to manage program data and layout:

| Entity | Role | Location |
| --- | --- | --- |
| Instr | The opcode enum representing all VM operations. | crates/runmat-vm/src/bytecode/instr.rs |
| Bytecode | The top-level container for instructions, spans, and metadata. | crates/runmat-vm/src/bytecode/program.rs |
| FunctionRegistry | Maps FunctionId to FunctionBytecode for semantic calls. | crates/runmat-vm/src/bytecode/program.rs |
| VmAssemblyLayout | Maps MIR local IDs to physical VM stack/variable slots. | crates/runmat-vm/src/layout/mod.rs |
