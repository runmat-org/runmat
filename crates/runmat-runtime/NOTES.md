Interpreter error semantics and MException
----------------------------------------

- Builtins return `Result<Value, String>`. The interpreter (VM) converts error strings into `MException` when inside try/catch.
- Format for identifiers: prefer `Category:Identifier: message`, but VM uses the last `:` to split into `identifier` and `message`.
- Catch binding: `catch e` stores `Value::MException { identifier, message, stack }` in the bound variable.
- `rethrow(e)` accepts either `MException` (preferred) or strings (legacy) and converts back to string error for upstream propagation.
- Next work: change builtins to surface richer identifiers; optionally change builtin signatures to produce `MException` directly when we tackle full runtime overhaul.

Closures and feval
------------------

- HIR lowers `@(x) expr` to `HirExprKind::AnonFunc { params: Vec<VarId>, body: HirExpr }`.
- Compiler synthesizes a `UserFunction` with a unique name and emits `CreateClosure(name, capture_count)`. Capture analysis is pending; currently `capture_count=0`.
- VM implements `CallFeval(argc)` to handle:
  - `Value::Closure` by prepending captures and dispatching to builtin or user function; 
  - `"@name"` strings as function handles to builtins; 
  - `Value::FunctionHandle(name)` for builtin dispatch.
- Method handles via `LoadMethod` produce a `Closure` bound to the receiver in captures.

Constructors and classes
------------------------

- Fallback: calling an unknown builtin whose name matches a registered class creates a default-initialized `Object` (temporary until proper constructors).
- Static and instance member/method access is enforced through class registry from the VM opcodes.

Indexing and tensors
--------------------

- 2D slices supported via `IndexSlice`; N-D generalization planned with per-dimension selectors (colon, scalar, vector/range, logical mask, end arithmetic).
- Assignments: scalar element (`StoreIndex`) and 2D slice assignment via `StoreSlice` for `A(:,j)=` and `A(i,:)=`.

Accelerate (GPU)
----------------

- `gpuArray`/`gather` builtins route through `runmat-accelerate-api` provider interface, defaulting to an in-process test provider.
- Planner to decide CPU/GPU routing exists in `runmat-accelerate`; kernels TBD.

Docs and builtins metadata
--------------------------

- Builtins registry now stores `category`, `doc`, and `examples` fields to support structured docs generation; macro currently populates defaults.


