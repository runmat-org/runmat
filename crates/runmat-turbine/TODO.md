### JIT completion plan (parity + performance)

- Workspace builds and all tests pass single-threaded.
- RunMat binary verified via stdin piping and script execution (`--no-jit --plot-headless`).
- Added boolean literal support end-to-end: VM opcode `LoadBool`, compiler emits for `true`/`false`, JIT gracefully falls back; instruction hashing updated.
- Named globals/persistents handled as no-ops in JIT; interpreter parity maintained.
- Instruction set doc updated to include the new opcodes.

Since we want a production-hardened system with (to start) full MATLAB code execution parity and we’ve driven the interpreter to 100% language semantics across grammar, HIR, and VM, the JIT should mirror all behavior, then exceed the VM on performance. Below is a prioritized, detailed plan with clear acceptance gates.

- Baseline goals
  - 100% opcode coverage: compile every `Instr` without interpreter fallback.
  - Identical mex error identifiers/messages on failures.
  - Exact semantics for varargin/varargout, multi-assignment, expansion, indexing/slicing, OOP, imports, globals/persistents, try/catch, and feval.
  - All tests green under JIT with single-threaded execution.

### Phase 0 — Unblock and stabilize the JIT
- Fix build/tests
  - [x] Update `runmat-turbine/tests/jit.rs` `UserFunction` initializers to include `has_varargin` and `has_varargout: false`.
  - [x] Handle `DeclareGlobalNamed`, `DeclarePersistentNamed` (no-op under JIT; VM owns semantics).
  - [ ] Implement `PackToRow`, `PackToCol` in JIT (currently falls back to interpreter).
  - [x] Add boolean literal path: VM `LoadBool`, compiler emission, JIT hashing; JIT falls back for now.
- ABI deltas
  - [ ] Standardize JIT function ABI: `(args: &[Value], out_count: usize, in_count: usize, ctx: &mut ThreadCtx) -> Result<Vec<Value>, String>`.
  - [ ] Add a compiled-function registry keyed by function name for direct calls (avoid string lookups at hot sites).
- Acceptance
  - [x] JIT crate builds and runs unit/integration tests.
  - [x] Unsupported ops fail open to interpreter without correctness loss.
  - [ ] `PackToRow/PackToCol` have native JIT lowering and tests.

### Phase 1 — Full opcode parity (semantics-first baseline)
- Per-instruction lowering (complete coverage)
  - Data/stack/locals: Load/StoreVar, Load/StoreLocal, Enter/ExitScope, Swap, Pop.
  - Arithmetic/relational/logical: all scalar and element-wise ops; delegate to runtime helpers where needed.
  - Construction: CreateMatrix, CreateMatrixDynamic, CreateCell2D, CreateRange.
  - Indexing (gather)
    - Index, IndexSlice, IndexSliceEx, IndexRangeEnd, Index1DRangeEnd, IndexCell, IndexCellExpand (all N-D + end-arithmetic + logical masks).
  - Stores (scatter)
    - StoreIndex, StoreIndexCell, StoreSlice, StoreSliceEx, StoreSlice1DRangeEnd (broadcast + shape laws).
  - Calls and expansion
    - CallBuiltin, CallBuiltinExpand{Last,At,Multi}, CallFunction, CallFunctionMulti, CallFunctionExpand{At,Multi}, feval and feval-expand.
  - OOP and static accesses
    - Load/StoreMember, Load/StoreMemberDynamic, LoadMethod, CallMethod, LoadStaticProperty, CallStaticMethod, RegisterClass.
  - Control flow and exceptions
    - Jump/JumpIfFalse, AndAnd/OrOr, EnterTry/PopTry, Return/ReturnValue with try/catch and last-exception propagation.
  - Imports/globals/persistents
    - RegisterImport, DeclareGlobal/Persistent (+ named variants) with name binding parity.
- Name resolution and dispatch
  - Preserve precedence (locals > user functions > specific imports > wildcard imports > Class.* statics) with the same compile-time and runtime checks as the VM.
  - Builtin vs user-function dispatch must match VM outcomes, including ambiguity errors.
- Acceptance
  - For each opcode, add a JIT unit test invoking just that instruction’s semantics.
  - Run the entire interpreter test suite in “JIT mode”; where tests use feval, varargout, cell/function expansion, indexing N-D gather/scatter, OOP, try/catch, ensure byte-for-byte mex-id/message parity.

### Phase 2 — Exceptions and error model uniformity
- Lower all potentially-failing ops as “checked calls”
  - JIT helpers return status flags; no unwinding needed.
  - Maintain a per-frame try-stack; on failure, branch to catch landing pad, bind catch var, continue.
- Ensure all error paths report mex identifiers/messages exactly like the VM; share error-formatting helpers with the runtime.
- Acceptance
  - Negative test suite (UndefinedFunction/Variable, TooManyInputs/Outputs, VarargoutMismatch, SliceNonTensor, IndexStepZero, MissingSubsref/Subsasgn, CellIndexType) passes under JIT.

### Phase 3 — GC integration (roots, barriers, safepoints)
- Rooting model
  - Add a per-thread shadow-root buffer in Turbine; at safepoints and before any allocation-capable runtime call, spill live `Value` references into a temporary `VariableArrayRoot` and register with the global RootScanner.
  - GC safepoints at: function prologues/epilogues, call sites, back-edges of loops.
- Write barriers
  - Insert `gc_record_write` at all field/element writes:
    - StoreIndexCell, StoreMember, StoreMemberDynamic, Struct field writes, cell element writes.
- Allocation
  - Route all allocations via runtime helpers returning `Value` and ensure no raw unrooted `GcPtr` escapes between safepoints.
- Acceptance
  - GC stress tests (existing + JIT-specific) pass with the JIT: deep recursion, large cells, repeated slice updates, persistent/global churn.
  - No lost/incorrect references under minor/major GC.

### Phase 4 — Performance tiering (baseline → optimized)
- Baseline JIT (Tier 0)
  - Straightforward lowering with minimal speculation. All phases above complete.
- Optimizing JIT (Tier 1)
  - Shape/type specialization at hot sites (e.g., numeric tensor fast paths).
  - Loop optimizations:
    - Strength reductions, LICM, bounds hoisting for N-D indexing loops.
    - Specialized 2-D fast paths: A(:, J) and A(I, :) gather/scatter.
  - Polymorphic inline caches (PICs) for:
    - Method calls (operator overloads), property loads/stores, static resolution under Class.* imports.
    - Guards on class name; megamorphic fallback to generic call.
  - Math and BLAS/LAPACK calls
    - Direct-call veneers to runtime-accelerated kernels with pre-verified shapes.
    - Vectorized element-wise ops on dense tensors (SIMD via target ISA or through a codegen backend).
- OSR and deopt
  - Hot loop counters to trigger OSR.
  - Guard failures or megamorphic explosions trigger deopt to baseline code or slow stub, not the interpreter.
- Acceptance
  - Microbench parity beats VM on indexing/scatter, element-wise ops, small BLAS cases; property tests remain green.
  - No regression under GC stress.

### Phase 5 — Varargs/varargout and multi-output polish
- ABI for varargout
  - JIT functions accept `out_count` and populate named outputs then draw from varargout cells; enforce TooManyOutputs/VarargoutMismatch.
- Expansion composition
  - Optimize “function/cell expansion into slice targets” by emitting PackToRow/Col directly with precomputed lengths when statically known; runtime sizing otherwise.
- Acceptance
  - All expansion and multi-assign tests green under JIT globally; include degenerate shapes/empties.

### Phase 6 — Feval, handles, closures, and nested functions
- Closure representation
  - Generate `CreateClosure` stubs: captures reside in a compact array carried in `Value::Closure`.
  - feval paths:
    - If closure/builtin compiled, call JIT entry directly.
    - If user function not yet compiled, JIT-compile on demand and update handle.
- Acceptance
  - feval + expand multi tests pass; nested functions and captures validated with property and negative tests.

### Phase 7 — OOP completeness and dispatch fast paths
- Instance/static dispatch
  - Inline cache on method lookup; cache miss uses runtime lookup (respecting access/private/static).
  - Dependent properties: synthesize getter/setter calls as in VM.
  - subsref/subsasgn chains: build compact index descriptors and call method with identical semantics.
- Acceptance
  - OOP negative/positive suites pass; operator overloading grid covered; shadowing/import precedence confirmed.

### Phase 8 — Tooling, diagnostics, and test harness
- Engine mode runner
  - Add an engine flag (jit/interpreter) to the test harness; default CI runs both.
- Coverage and counters
  - Per-opcode JIT execution counters and “fallback” counters; assert zero fallback before marking complete.
- Fuzzing and property tests
  - Run existing fuzz/property seeds under JIT; add JIT-specific seeds for indexing/scatter and try/catch.

### Cross-cutting engineering details
- IR and codegen
  - Introduce a small SSA-like IR in Turbine for optimization (CSE, const-fold, LICM, loop opts).
  - Backend options: keep the current custom codegen or adopt Cranelift for portability and SIMD; wrap calls to runtime functions for complex ops.
- Threading model
  - JIT compilation single-threaded for determinism in tests; code execution remains single-threaded per your test heuristic.
- Compatibility
  - No changes to `runmat-runtime` unless required for new helper entry points; prefer using existing builtins like `__make_cell`.

### Definition of done
- 100% of interpreter tests pass in JIT mode with `--test-threads=1`.
- A “no-fallback” assertion demonstrating every executed opcode ran through native JIT paths.
- GC stress + OOP/operator-overload suites green.
- Microbenchmarks show clear wins over VM for: 2‑D gather/scatter, N‑D broadcast scatter, element-wise ops, small BLAS calls.

### Immediate next steps
- Implement `PackToRow`/`PackToCol` lowering in Turbine; add focused tests (packing + slice stores).
- Optional: compile `LoadBool` to numeric 1.0/0.0 in JIT to remove fallback in trivial paths.
- Add a JIT-engine mode toggle in harness to run critical suites (indexing/scatter, try/catch, expansion) under JIT and assert zero fallbacks for covered ops.

- Added complete opcode coverage to `crates/runmat-ignition/INSTR_SET.md`, so the JIT plan can target each instruction precisely.