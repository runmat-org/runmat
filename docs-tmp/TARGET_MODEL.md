# Target Model

## Goal

RunMat should remain MATLAB-native at the source level while using an explicit semantic compiler model internally.

The target architecture should support:

- MATLAB-compatible source ergonomics
- explicit module, function, class, binding, and entrypoint identity
- semantic HIR as the compiler source of truth
- VM slot layout derived from HIR, not encoded in HIR
- first-class analysis facts for types, shapes, effects, captures, workspace visibility, fusibility, and parallel safety
- config-driven project composition in a later stage

## Core Principles

- MATLAB compatibility is the default user experience.
- HIR models semantic ownership and identity, not VM layout.
- Runtime slots are implementation details derived from semantic bindings.
- Functions are semantic items, not statements.
- Classes are semantic items, not statements.
- Bindings are semantic identities, not variable-array indexes.
- Captures are relations to existing bindings, not duplicate bindings.
- Source-level scripts lower to synthetic entry functions.
- Analysis facts are products keyed by stable semantic IDs.
- Workspace state is derived from semantic binding visibility.
- HIR preserves semantic/source truth; MIR normalizes control flow and dataflow for analysis and code generation.
- Async and host interaction are explicit execution semantics, not incidental runtime callbacks.

## Compiler Pipeline

The target compiler pipeline has distinct semantic layers:

1. Parse source into AST.
2. Resolve names, modules, imports, classes, functions, and bindings into semantic HIR.
3. Lower HIR into MIR for normalized control flow, dataflow, captures, and effect analysis.
4. Run analysis over semantic HIR plus MIR.
5. Derive VM/runtime layout and bytecode from MIR plus analysis facts.
6. Execute through the runtime, async host interfaces, and optional acceleration providers.

The primary layers are:

- AST: source syntax.
- HIR: semantic/source model for modules, functions, classes, bindings, entrypoints, and diagnostics.
- MIR: normalized compiler IR for control flow, dataflow, definite assignment, effects, captures, async boundaries, and codegen planning.
- VM/runtime layout: concrete slots, frames, closure environments, workspace export maps, and bytecode.

Rules:

- HIR should stay source-faithful enough for diagnostics, LSP, and workspace semantics.
- MIR should be easier to analyze than source-shaped HIR.
- VM bytecode should not be the first normalized IR.
- Fusion, dataflow, and effect analysis should eventually consume MIR/analysis facts rather than reconstructing semantics from bytecode.

## Compatibility Modes

Compatibility mode is an input to parsing, lowering, resolution, diagnostics, LSP, and runtime entrypoint policy.

Conceptually:

```rust
pub enum CompatibilityMode {
    MatlabStrict,
    RunMatExtended,
    Interactive,
}
```

Rules:

- `MatlabStrict` accepts MATLAB-compatible source and rejects RunMat-only syntax where possible.
- `RunMatExtended` enables RunMat language extensions such as async syntax, `await`, `spawn`, and `isolated` where supported.
- `Interactive` enables REPL/notebook conveniences such as top-level await, persistent session workspace behavior, and host-defined display policy.
- Compatibility mode should affect diagnostics explicitly; unsupported extensions should produce clear mode-aware diagnostics.

## Identity Model

The target model distinguishes local arena IDs from stable semantic identities.

Local IDs are compact IDs allocated inside one `HirAssembly` or MIR body:

- `ModuleId`
- `FunctionId`
- `ClassId`
- `EntrypointId`
- `BindingId`
- `ExprId`
- `StmtId`
- `BasicBlockId`
- `LocalId`

Stable identities are qualified semantic paths suitable for diagnostics, cross-module resolution, LSP, and caching:

- package identity
- module path
- item path
- class path
- function path
- method path
- property path

Conceptually:

```rust
pub struct DefPath {
    pub package: PackageName,
    pub module: QualifiedName,
    pub item: Vec<DefPathSegment>,
}

pub enum DefPathSegment {
    Function(SymbolName),
    Class(SymbolName),
    Method(SymbolName),
    Property(SymbolName),
    Entrypoint(EntrypointName),
    Synthetic(SyntheticName),
}
```

Rules:

- Local numeric IDs are stable only within a compiler product.
- Cross-session, cross-file, and incremental identities should use qualified paths plus source/package/cache keys.
- Diagnostics and LSP should avoid exposing local arena IDs as stable public identity.
- Source spellings are represented by domain-specific name newtypes such as `SymbolName`, `BindingName`, `MemberName`, `EntrypointName`, and `DimSymbol`; raw strings are not primary semantic identity.

## Semantic IDs

The target HIR uses local arena IDs for semantic entities inside one compiler product:

- `ModuleId`
- `FunctionId`
- `ClassId`
- `EntrypointId`
- `BindingId`
- `ExprId`
- `StmtId`
- `SourceId`

`VarId` should not be part of the target semantic HIR. If a VM-local slot ID exists, it belongs to VM layout metadata, not semantic HIR.

These IDs are stable for the lifetime of a `HirAssembly`, but stable cross-session identity comes from qualified symbol paths and source/package/cache keys.

## Top-Level HIR Shape

The primary compiler product is `HirAssembly`.

Conceptually:

```rust
pub struct HirAssembly {
    pub modules: Vec<HirModule>,
    pub functions: Vec<HirFunction>,
    pub classes: Vec<HirClass>,
    pub bindings: Vec<HirBinding>,
    pub entrypoints: Vec<HirEntrypoint>,
}
```

Invariants:

- `HirAssembly` is the canonical owner of functions, classes, bindings, modules, and entrypoints.
- Module inventories contain top-level module items only.
- Nested functions are reached through `HirFunction.parent`, not duplicated in module inventories.
- Class methods are ordinary `HirFunction`s referenced by class method metadata.

## Modules

`HirModule` represents an authored source unit after semantic lowering.

Conceptually:

```rust
pub struct HirModule {
    pub id: ModuleId,
    pub name: QualifiedName,
    pub source_id: SourceId,
    pub source_unit: SourceUnitKind,
    pub imports: Vec<HirImport>,
    pub top_level_functions: Vec<FunctionId>,
    pub classes: Vec<ClassId>,
    pub synthetic_entry_function: Option<FunctionId>,
}

pub enum SourceUnitKind {
    ScriptFile,
    FunctionFile,
    ClassFile,
    PackageFunctionFile,
    ClassFolderMethodFile,
    ReplSubmission,
    NotebookCell,
}
```

Rules:

- A module owns top-level functions and classes by ID.
- A module may have a synthetic entry function if source has script-like top-level executable code.
- Module imports are semantic items used by resolution and diagnostics.
- Module identity should become qualified and stable once project composition lands.

## Source Unit Semantics

Source unit kind affects parsing, declaration visibility, entrypoint creation, and resolution precedence.

Rules:

- Script files lower top-level executable statements into a synthetic entry function and may define local functions.
- Function files define a primary function plus optional local functions.
- Class files define one nominal class plus structured class members.
- Package function files contribute package-qualified function identity.
- Class-folder method files contribute methods to the corresponding nominal class identity.
- REPL submissions and notebook cells lower through synthetic source units with interactive entrypoint policy.
- Local functions are visible across the source unit regardless of declaration order.
- `return` legality and behavior depend on source unit and function/script context.
- `arguments` blocks belong to function/class metadata rather than ordinary executable statements.

## Entrypoints

`HirEntrypoint` is an explicit execution target over a lowered `HirFunction`.

Entrypoints should not create another semantic function taxonomy. Function shape belongs to `HirFunction.kind`; entrypoints only describe how a host/project selects a function for execution and how results are exposed.

Conceptually:

```rust
pub struct HirEntrypoint {
    pub id: EntrypointId,
    pub name: Option<EntrypointName>,
    pub target: FunctionId,
    pub origin: EntrypointOrigin,
    pub policy: EntrypointPolicy,
}

pub enum EntrypointOrigin {
    ProjectDeclared,
    SourcePath,
    ReplSubmission,
    NotebookCell,
    HostSynthetic,
}

pub struct EntrypointPolicy {
    pub workspace_export: WorkspaceExportPolicy,
    pub top_level_await: bool,
}

pub enum WorkspaceExportPolicy {
    ExportTopLevelBindings,
    ReturnFunctionOutputs,
    HostDefined,
}
```

Rules:

- Execution begins from an entrypoint, not from arbitrary raw statement lists.
- Script-like source lowers to a synthetic entry function.
- REPL and notebook submissions use ephemeral synthetic entrypoints.
- Function-oriented entrypoints target ordinary `HirFunction`s.
- Top-level await is entrypoint policy, not a special function kind.
- Workspace export behavior is entrypoint policy, not inferred from VM slots.
- Binding `WorkspaceVisibility` describes which bindings are eligible for export; `WorkspaceExportPolicy` describes what the selected entrypoint actually exposes.

## Functions

`HirFunction` is the uniform executable semantic node.

It covers:

- named functions
- nested functions
- anonymous functions
- synthetic entry functions
- class methods

Conceptually:

```rust
pub struct HirFunction {
    pub id: FunctionId,
    pub module: ModuleId,
    pub parent: Option<FunctionId>,
    pub enclosing_class: Option<ClassId>,
    pub name: FunctionName,
    pub kind: FunctionKind,
    pub params: Vec<BindingId>,
    pub outputs: Vec<BindingId>,
    pub abi: FunctionAbi,
    pub locals: Vec<BindingId>,
    pub captures: Vec<CapturedBinding>,
    pub modifiers: FunctionModifiers,
    pub body: HirBlock,
    pub span: Span,
}
```

Function kinds:

- `Named`
- `Anonymous`
- `SyntheticEntrypoint`
- `ClassMethod { is_static: bool }`

Function modifiers:

- `isolated`
- variadic input/output behavior is represented by `FunctionAbi`

Rules:

- Functions are not statements.
- Every executable body belongs to exactly one function.
- Nested functions use `parent: Some(parent_function)`.
- Anonymous functions are real functions referenced from expressions by `FunctionId`.
- `varargin`, `varargout`, `nargin`, and `nargout` are ordinary or implicit bindings with ABI-backed semantics.

## Bindings

`HirBinding` represents semantic identity for a named or implicit binding.

Conceptually:

```rust
pub struct HirBinding {
    pub id: BindingId,
    pub owner: BindingOwner,
    pub name: BindingName,
    pub role: BindingRole,
    pub storage: BindingStorage,
    pub workspace_visibility: WorkspaceVisibility,
    pub declared_span: Span,
}
```

Binding owners:

- `Module(ModuleId)`
- `Function(FunctionId)`

Binding roles:

- `Parameter`
- `Output`
- `Local`
- `ModuleBinding`
- `ImplicitAns`

Binding storage classes:

- `Lexical`
- `Global`
- `Persistent`

Workspace visibility classes:

- `Hidden`
- `TopLevel`
- `ModuleVisible`
- `ImplicitAns`

Rules:

- Binding identity is semantic and stable within an assembly.
- Binding identity is not VM slot layout.
- Parameters, outputs, ordinary locals, nested locals, and anonymous-function locals are normally `Hidden`.
- Synthetic entry function top-level bindings are normally `TopLevel`.
- Module-owned bindings are `ModuleVisible`.
- `ans` is represented by an `ImplicitAns` binding.

## Captures

Nested MATLAB functions use shared lexical capture by default.

Rules:

- A captured binding remains owned by its declaring function or module.
- Captures are recorded on the capturing function.
- Mutation through a nested function targets the same semantic `BindingId` seen by the parent.
- Captures do not create duplicate bindings.
- Runtime may later lower captured mutable bindings into environment cells, but that is layout, not HIR identity.
- Shared lexical capture is valid for ordinary calls and direct await in the same execution context.
- Captures crossing `spawn` must satisfy spawn-safety rules; spawned tasks must not alias mutable parent frame storage.

Conceptually:

```rust
pub struct CapturedBinding {
    pub binding: BindingId,
    pub from_function: FunctionId,
}
```

## Spawn Safety

`spawn` is the boundary where suspension becomes true concurrent execution. It must not allow ordinary lexical capture to become a data race or a VM memory-safety hazard.

Rules:

- `await(future)` may poll the future in the current execution context and may use ordinary lexical captures.
- `spawn(future)` requires the future and its captured environment to be spawn-safe.
- Spawn-safe captures are immutable snapshots, by-value copies, immutable shared runtime handles, or explicitly synchronized shared handles.
- Mutable lexical captures from the spawning frame are rejected unless they are wrapped in an explicit synchronized/runtime-managed object.
- Spawned futures/tasks must not hold raw references into another task's stack, frame, or unrooted GC storage.
- Discarding an unspawned future means the future never ran; discarding a spawned task follows explicit task lifetime/cancellation policy.
- Provider and GPU handles may be shared across spawned tasks only according to their runtime metadata: immutable sharing, copy-on-write, synchronized mutation, or rejection.

Conceptual analysis fact:

```rust
pub enum SpawnSafetyFact {
    SpawnSafe,
    RequiresIsolation,
    NotSpawnSafe { reason: SpawnSafetyReason },
}

pub enum SpawnSafetyReason {
    MutableLexicalCapture,
    NonSendableRuntimeHandle,
    UnsynchronizedSharedMutation,
    UnknownDynamicCapture,
}
```

## Isolated Functions

`isolated` is a restriction on normal functions, not a separate function model.

Rules:

- An `isolated` function may not lexically capture enclosing local bindings.
- Required outer values must be passed explicitly as parameters.
- `isolated` does not imply purity, determinism, or side-effect freedom.
- Invalid captures by `isolated` functions should be rejected during lowering or early semantic analysis.

## Blocks And Statements

Function bodies are block-structured.

Statements have stable `StmtId`s.

Conceptually:

```rust
pub struct HirBlock {
    pub statements: Vec<HirStmt>,
}

pub struct HirStmt {
    pub id: StmtId,
    pub kind: HirStmtKind,
    pub span: Span,
}
```

Statement kinds include:

- expression statements
- assignments
- multi-assignments
- `if`
- `while`
- `for`
- `switch`
- `try/catch`
- `global` declarations
- `persistent` declarations
- `break`
- `continue`
- `return`
- `import`

Rules:

- `HirStmt::Function` does not exist in the target model.
- `HirStmt::ClassDef` does not exist in the target model.
- `global`, `persistent`, and `import` statements remain for source fidelity and diagnostics, but their semantic meaning is also represented on bindings/modules.

## Places And Assignments

Assignment targets use a unified place model.

Conceptually:

```rust
pub enum HirPlace {
    Binding(BindingId),
    Member(Box<HirExpr>, MemberName),
    MemberDynamic(Box<HirExpr>, Box<HirExpr>),
    Index(Box<HirExpr>, Vec<HirExpr>),
    IndexCell(Box<HirExpr>, Vec<HirExpr>),
}
```

Rules:

- Plain variable assignment is assignment to `HirPlace::Binding`.
- Member, dynamic member, array index, and cell index assignment all share the same target abstraction.
- VM store instructions are derived later from the place and layout metadata.

## Expressions And Calls

Expressions have stable `ExprId`s.

Conceptually:

```rust
pub struct HirExpr {
    pub id: ExprId,
    pub kind: HirExprKind,
    pub span: Span,
}
```

Expression kinds include:

- numeric literals
- string literals
- constants
- binding references
- unary and binary operations
- tensors
- cells
- ranges
- colon and end
- indexing
- cell indexing
- member access
- dynamic member access
- calls
- function handles
- metaclass references
- anonymous function references

Calls use semantic call structure.

Conceptually:

```rust
pub struct HirCall {
    pub callee: HirCallableRef,
    pub args: Vec<HirExpr>,
    pub syntax: CallSyntax,
    pub requested_outputs: RequestedOutputCount,
}

pub enum HirCallableRef {
    Function(FunctionId),
    ClassConstructor(ClassId),
    Builtin(BuiltinId),
    Imported(DefPath),
    DynamicExpr(Box<HirExpr>),
    Unresolved(QualifiedName),
}
```

Call syntax:

- `Plain`
- `Method`
- `DottedInvoke`

Rules:

- Calls should carry semantic identity where possible.
- Calls should carry requested-output context where known.
- Local user function calls should resolve to `FunctionId`.
- Builtins should resolve to stable builtin IDs and metadata where known.
- Imports should preserve canonical identity even when source uses unqualified names.
- Dynamic and unresolved calls degrade conservatively.
- String names remain useful for diagnostics and dynamic lookup, but not as primary semantic identity.

## MATLAB Compatibility Semantics

MATLAB compatibility requires context-sensitive semantics. The compiler target should model these semantics directly instead of treating symptoms as runtime special cases.

### Evaluation Context

Expressions are interpreted relative to their source context.

Conceptually:

```rust
pub enum EvaluationContext {
    Statement,
    Expression,
    AssignmentRhs { targets: OutputTargetList },
    FunctionArgument,
    ConcatElement,
    CellElement,
    IndexOperand,
    CommandArgument,
    NameValueArgument,
    LineSpecArgument,
    DynamicFieldName,
    ForRange,
    Condition,
    SwitchExpression,
    CaseExpression,
    ReturnValue { requested_outputs: RequestedOutputs },
}
```

Rules:

- Bracketed syntax in statement position may be a multi-assignment target list; in expression position it may be concatenation.
- Command syntax is only valid in statement context.
- `ans` assignment policy is determined by statement/expression context and entrypoint policy.
- Resolution of ambiguous forms such as `foo(x)` depends on context and resolver facts.
- Name-value and line-spec parsing are argument parser modes driven by builtin/method metadata.
- Switch/case, condition, and for-range contexts have MATLAB-specific semantics and should not lower as generic Rust-like expression contexts.

### Requested Outputs And Call ABI

MATLAB calls are arity-sensitive. A call receives its arguments plus the number of outputs requested by the call site.

Conceptually:

```rust
pub struct RequestedOutputs {
    pub count: RequestedOutputCount,
    pub targets: OutputTargetList,
}

pub enum RequestedOutputCount {
    Zero,
    One,
    Exactly(usize),
    AtLeast(usize),
    UnknownDynamic,
}

pub struct OutputTargetList {
    pub targets: Vec<OutputTarget>,
}

pub enum OutputTarget {
    Place(HirPlace),
    Discard,
    VarargoutExpansion,
}
```

Rules:

- `f()` in statement context requests zero outputs unless the source context assigns an implicit `ans`.
- `x = f()` requests one output.
- `[a, b] = f()` requests two outputs.
- `[~, idx] = max(x)` requests two outputs and discards the first.
- Builtins and user functions may depend on requested output count.
- `nargout` in a callee observes the requested output count.
- Function summaries may be parameterized by requested output count where behavior differs.

### Multi-Value And Comma-Separated Lists

MATLAB has list-valued flows that are not ordinary arrays and should not be modeled as durable tensor values.

Conceptually:

```rust
pub enum ValueFlowFact {
    NoValue,
    Single(TypeFact),
    CommaList(Vec<TypeFact>),
    UnknownList,
}
```

Rules:

- Multi-output calls produce output lists at list-consuming boundaries.
- `varargin{:}`, `varargout{:}`, and cell expansion can produce comma-separated lists.
- Struct-array field access and overloaded indexing may produce comma-separated lists.
- Function arguments and assignment target lists consume comma-separated lists.
- Ordinary scalar expression contexts reject or collapse list-valued flows according to MATLAB-compatible rules.

### Function ABI

Function semantics should be represented as an ABI, not only as parameter/output vectors.

Conceptually:

```rust
pub struct FunctionAbi {
    pub fixed_inputs: Vec<BindingId>,
    pub varargin: Option<BindingId>,
    pub fixed_outputs: Vec<BindingId>,
    pub varargout: Option<BindingId>,
    pub implicit_nargin: Option<BindingId>,
    pub implicit_nargout: Option<BindingId>,
}
```

Rules:

- Excess inputs are packed into `varargin` as a cell array.
- Requested excess outputs are produced through `varargout`.
- `nargin` and `nargout` are implicit read-only function-local bindings.
- An output that shares a name with an input reuses the same semantic binding.
- The VM frame layout is derived from `FunctionAbi`; the ABI is not inferred from slot order.

### Assignment And Place Mutation

Assignment is semantic mutation, not just storing into a local slot.

Conceptually:

```rust
pub enum PlaceMutationKind {
    BindOrAssign,
    IndexedAssign,
    CellAssign,
    MemberAssign,
    Delete,
}

pub struct PlaceMutation {
    pub place: HirPlace,
    pub kind: PlaceMutationKind,
    pub creation_policy: AssignmentCreationPolicy,
    pub shape_policy: AssignmentShapePolicy,
}

pub enum AssignmentCreationPolicy {
    ExistingOnly,
    CreateBinding,
    CreateArrayByIndex,
    CreateStructFieldPath,
    Overloaded,
}
```

Rules:

- Plain assignment may create a binding in the current workspace.
- Indexed assignment may create and grow an array in MATLAB-compatible contexts.
- Field assignment may create structural fields and intermediate structs where MATLAB-compatible.
- Assignment of `[]` through an indexed place is deletion.
- Slice assignment enforces MATLAB shape compatibility and scalar expansion.
- Object mutation may dispatch through class metadata, including `subsasgn`.
- Handle-object mutation and value-object copy/update behavior are class semantics.

### Indexing Semantics

Indexing is a semantic subsystem with read and write behavior.

Conceptually:

```rust
pub enum IndexKind {
    Paren,
    Brace,
    Dot,
}

pub enum IndexComponent {
    Colon,
    End { dim: Option<usize>, offset: isize },
    Expr(HirExpr),
    Logical(HirExpr),
}

pub struct IndexingSemantics {
    pub kind: IndexKind,
    pub components: Vec<IndexComponent>,
    pub result_context: IndexResultContext,
}

pub enum IndexResultContext {
    ReadSingle,
    ReadCommaList,
    AssignmentTarget,
    DeletionTarget,
    FunctionArgumentExpansion,
}
```

Rules:

- Indexing is 1-based.
- `end` resolves relative to the indexed operand and dimension.
- Linear and multidimensional indexing are distinct semantic modes.
- Paren, brace, and dot indexing have distinct result categories.
- Brace indexing and struct-array field access may produce comma-separated lists.
- Logical indexing follows MATLAB-compatible shape behavior.
- Class indexing can dispatch through `subsref` and `subsasgn` metadata.
- Object indexing may dispatch `end` through class metadata where applicable.
- Deletion assignment and indexed write contexts have different shape rules from indexed reads.
- Table-like and domain-specific indexing should be represented through nominal class metadata, not bespoke parser cases.

### Command Syntax

MATLAB command syntax is source syntax, not a runtime fallback.

Conceptually:

```rust
pub struct HirCommandCall {
    pub command: HirCallableRef,
    pub args: Vec<CommandArgument>,
}

pub enum CommandArgument {
    Word(SymbolName),
    StringLiteral(StringLiteralId),
    OptionToken(CommandOptionName),
}
```

Rules:

- Command syntax is valid only in statement context.
- Newline or semicolon terminates a command statement.
- Bare command arguments lower to char/string values while preserving source spelling for diagnostics.
- Command syntax lowers to ordinary call semantics after parsing/lowering.
- Workspace-affecting commands such as `clear` retain explicit workspace effects.

### Reference And Call Classification

The resolver should classify references and calls into semantic categories.

Conceptually:

```rust
pub enum ReferenceKind {
    Binding(BindingId),
    Function(FunctionId),
    Builtin(BuiltinId),
    Class(ClassId),
    Package(QualifiedName),
    Imported(DefPath),
    RuntimeClass(ClassSymbol),
    Dynamic,
    Unresolved,
}

pub enum CallKind {
    DirectFunction(FunctionId),
    Builtin(BuiltinId),
    Constructor(ClassId),
    StaticMethod { class: ClassId, method: MethodId },
    InstanceMethod { receiver: Box<HirExpr>, method: MethodId },
    PackageFunction(DefPath),
    FunctionHandle,
    Dynamic,
    OverloadedOperator,
    OverloadedIndexing,
}
```

Rules:

- Package-qualified calls are not ordinary member access.
- Builtins use stable builtin IDs and metadata where available.
- Unresolved static references and dynamic runtime lookup are distinct states.
- Operator and indexing overloads resolve through class/runtime metadata when known.

### Workspace Effects

Some MATLAB operations mutate workspace/session state and must be explicit effects.

Conceptually:

```rust
pub enum WorkspaceEffect {
    None,
    ReadsWorkspace,
    CreatesBinding,
    ClearsBinding,
    ClearsFunctionCache,
    MutatesGlobal,
    MutatesPersistent,
    LoadsExternalBindings,
    DynamicEval,
}

pub enum EnvironmentEffect {
    PathMutation,
    WorkingDirectoryMutation,
    FunctionCacheInvalidation,
    DynamicLookupInvalidation,
}
```

Rules:

- `load` can introduce bindings into the active workspace according to entrypoint/function context.
- `clear` can remove bindings, clear function caches, and trigger resource release.
- `global` and `persistent` are storage effects on bindings.
- `eval`, `evalin`, and `assignin` are dynamic workspace effects and analysis barriers.
- Workspace effects are optimization, fusion, and materialization barriers.
- `addpath`, `rmpath`, `path`, `cd`, and `rehash` are environment effects that can invalidate dynamic lookup and cached resolution products according to compatibility/project policy.
- `which` and `exist` query the resolver/runtime environment and should report source/project-aware results.

### Function Handle Semantics

Function handles carry callable identity plus optional closure environment.

Conceptually:

```rust
pub enum FunctionHandleTarget {
    Function(FunctionId),
    Builtin(BuiltinId),
    Method(MethodId),
    Anonymous(FunctionId),
    DefPath(DefPath),
    DynamicName(SymbolName),
}
```

Rules:

- `@name` resolves to function, builtin, method, or dynamic-name handle according to resolution context.
- Anonymous function handles reference a real `HirFunction` and capture lexical bindings through the ordinary capture model.
- `feval`, `arrayfun`, `cellfun`, `str2func`, and `func2str` use function-handle identity and may introduce dynamic call effects.
- Function handles are spawn-safe only when their target and captured environment are spawn-safe.

### Tensor Element Domains

Tensor facts include element domain separately from shape and runtime placement.

Conceptually:

```rust
pub struct TensorTypeFact {
    pub element: ElementTypeFact,
    pub shape: ShapeFact,
    pub sparsity: SparsityFact,
}

pub enum ElementTypeFact {
    Unknown,
    Logical,
    Numeric { class: NumericClass, domain: NumericDomain },
    Char,
    Object(ClassId),
}

pub type TensorElementDomainFact = ElementTypeFact;

pub enum NumericDomain {
    Real,
    Complex,
}
```

Rules:

- Complex storage representation does not change language shape.
- `abs`, `real`, `imag`, and `angle` map complex numeric tensors to real numeric tensors.
- `conj` preserves shape and complex numeric domain.
- FFT-family summaries produce complex numeric tensors where appropriate.
- Runtime/provider storage does not change element domain or shape facts.

### Core Array And Operator Semantics

Array semantics are language semantics, not only builtin implementation details.

Conceptually:

```rust
pub enum EmptyArrayRole {
    EmptyValue,
    ConcatenationIdentity,
    DeletionMarker,
}

pub enum ExpansionSemantics {
    ExactShape,
    ScalarExpansion,
    ImplicitExpansion,
}

pub enum OperatorKind {
    MatrixMultiply,
    ElementwiseMultiply,
    MatrixPower,
    ElementwisePower,
    Mldivide,
    Mrdivide,
    ElementwiseDivide,
    Transpose,
    ConjugateTranspose,
}
```

Rules:

- `[]` is an ordinary empty value in expression contexts, a concatenation identity in MATLAB-compatible concatenation contexts, and a deletion marker in indexed assignment contexts.
- Horizontal, vertical, and N-dimensional concatenation use MATLAB shape rules and class overload hooks.
- Cell, char, string, struct, object, and numeric concatenation have distinct compatibility rules.
- Arithmetic uses scalar expansion and implicit expansion where MATLAB-compatible.
- Matrix operators are distinct from elementwise operators in HIR/MIR and analysis facts.
- Transpose and conjugate transpose are distinct operations and interact with complex element domains.
- Shape/orientation preservation is part of operator and indexing semantics.

### Numeric Classes

Numeric class and numeric domain are separate from shape and provider placement.

Conceptually:

```rust
pub enum NumericClass {
    Double,
    Single,
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Int64,
    UInt64,
}
```

Rules:

- Double is the default numeric class for MATLAB-compatible numeric literals unless context changes it.
- Logical is distinct from numeric class but participates in condition and logical-index contexts.
- Integer promotion, casting, and overflow behavior should be metadata-driven and MATLAB-compatible where supported.
- Sparse arrays are language-visible array representations, not provider placement facts.
- `class`, `isa`, `cast`, and numeric constructors consume type metadata rather than runtime string guesses.

### String And Char Semantics

Character arrays and string arrays are distinct language values.

Rules:

- Single-quoted literals produce char arrays.
- Double-quoted literals produce string values where compatibility mode supports them.
- Command syntax bare words lower to char/string values according to MATLAB compatibility policy; MATLAB-strict behavior should prefer char-vector semantics where appropriate.
- Name-value keys, property names, and many legacy APIs should accept compatible char and string forms through builtin/method metadata.
- Char concatenation, string array construction, and conversion through `char` and `string` are language semantics, not ordinary numeric tensor behavior.

### Structs, Cells, And Object Arrays

Aggregate values have MATLAB-specific indexing, expansion, and assignment behavior.

Conceptually:

```rust
pub enum AggregateKind {
    Struct,
    Cell,
    ObjectArray(ClassId),
}
```

Rules:

- Struct arrays are structural; scalar structs and struct arrays share field metadata but differ in indexing/list behavior.
- Field access over struct arrays may produce comma-separated lists.
- Dynamic field names are semantic member/indexing forms.
- Field assignment may expand struct arrays and create fields where MATLAB-compatible.
- Cell construction, cell indexing `()`, and cell content indexing `{}` are distinct.
- `C{:}` is a comma-list expansion site.
- Object arrays combine nominal dispatch with array indexing and property/method access.
- Value-vs-handle object mutation behavior is class metadata, not local assignment syntax.

### Control Flow Semantics

Control-flow lowering should preserve MATLAB condition and iteration semantics.

Conceptually:

```rust
pub enum LoopIterationSemantics {
    ForColumns,
    WhileCondition,
}
```

Rules:

- `for x = A` iterates over columns of `A` according to MATLAB semantics.
- `if` and `while` conditions require MATLAB-compatible scalar logical/numeric condition behavior.
- `&&` and `||` are scalar short-circuit operators.
- `&` and `|` are elementwise operators, with condition contexts imposing scalar compatibility where applicable.
- `switch` and `case` use MATLAB matching semantics, not Rust-style pattern matching.
- `break`, `continue`, and `return` have source-context-specific legality and diagnostics.

## Resolution Model

Name resolution is an explicit compiler product, not an ad hoc lowering side effect.

The resolver builds a semantic index for:

- lexical bindings
- function declarations
- nested functions
- class declarations
- class methods and properties
- module symbols
- imports
- builtins and runtime metadata symbols
- dynamic and unresolved references

The resolver must classify ambiguous MATLAB-style forms such as `foo(x)` using a principled rule set:

- binding reference plus indexing when `foo` resolves to a value binding
- direct function call when `foo` resolves to a local/user function
- builtin call when `foo` resolves to a builtin
- imported call when `foo` resolves through import visibility
- class constructor when `foo` resolves to a class
- dynamic call or unresolved reference when static resolution cannot prove a target

Resolution should respect MATLAB-compatible precedence. Conceptually, a lookup considers:

- lexical/local bindings
- nested and local functions visible in the source unit
- imported names
- private functions visible from the current source location
- class constructors, static members, and methods where the syntax/context permits
- package-qualified functions and classes
- project/source-path functions according to source index and path precedence
- runtime class metadata and builtins
- dynamic fallback where static resolution cannot prove a target

Resolver products should include:

- canonical symbol identities where known
- import visibility tables
- ambiguity diagnostics
- unresolved reference diagnostics or conservative dynamic fallback classification
- source spelling for diagnostics

Rules:

- Canonical identity is separate from source-local spelling.
- `import` changes visibility, not package availability.
- Dynamic fallback should be explicit; it should not look like a resolved static call.
- Variables generally shadow functions in expression contexts where MATLAB does so.
- Overloaded dispatch may refine a call target using receiver and argument type facts after initial syntactic/name classification.

## MIR

`runmat-mir` is the normalized compiler IR between semantic HIR and VM/runtime lowering.

MIR should represent:

- functions as CFGs
- basic blocks
- terminators
- normalized places
- normalized rvalues
- explicit local temporaries
- capture/environment accesses after semantic capture analysis
- explicit call operands
- requested-output call ABI
- comma-list expansion and consumption points
- explicit place mutation operations
- normalized indexing descriptors
- workspace-effect operations
- future/task creation
- await terminators or await statements with explicit resume points
- async suspension or host-interaction boundaries where needed
- source-span mapping back to HIR

Conceptually:

```rust
pub struct MirAssembly {
    pub bodies: HashMap<FunctionId, MirBody>,
}

pub struct MirBody {
    pub function: FunctionId,
    pub locals: Vec<MirLocal>,
    pub blocks: Vec<BasicBlock>,
    pub source_map: MirSourceMap,
}

pub struct BasicBlock {
    pub id: BasicBlockId,
    pub statements: Vec<MirStmt>,
    pub terminator: MirTerminator,
}
```

MIR is responsible for making these analyses straightforward:

- definite assignment
- control-flow joins
- loop widening
- local dataflow
- capture read/write summaries
- effect propagation
- async boundary tracking
- fusion region identification
- VM slot and frame layout planning

Rules:

- HIR remains the semantic/source truth.
- MIR is derived and may be rebuilt.
- MIR local IDs are not semantic binding IDs.
- MIR locals may map back to `BindingId`s or temporaries.
- VM bytecode is derived from MIR, not directly from source-shaped HIR where avoidable.

## Classes And Objects

Classes are nominal semantic items.

Structs remain structural.

### Structs

Rules:

- Structs are structural.
- Field sets and field-sensitive refinement matter.
- Known fields may be refined through control flow.
- Structural typing remains distinct from nominal class identity.

### Classes

Rules:

- Classes are nominal.
- Identity is based on declared class identity, not field shape.
- Inheritance matters.
- Property lookup and method lookup resolve through class metadata.
- Value-object and handle-object semantics may be represented in type facts.
- Constructors, static methods/properties, operator overloads, and indexing overloads are class dispatch metadata.

Conceptually:

```rust
pub struct HirClass {
    pub id: ClassId,
    pub module: ModuleId,
    pub name: QualifiedName,
    pub super_class: Option<ClassId>,
    pub kind: ClassKind,
    pub properties: Vec<ClassProperty>,
    pub methods: Vec<ClassMethod>,
    pub events: Vec<ClassEvent>,
    pub enumerations: Vec<ClassEnumeration>,
    pub arguments: Vec<ClassArgumentBlock>,
    pub span: Span,
}

pub enum ClassKind {
    Value,
    Handle,
}

pub struct ClassDispatchMetadata {
    pub constructors: Vec<MethodId>,
    pub methods: MethodTable,
    pub static_methods: MethodTable,
    pub properties: PropertyTable,
    pub static_properties: PropertyTable,
    pub operators: OperatorTable,
    pub indexing: IndexingOverloadTable,
    pub value_semantics: ObjectValueSemantics,
}

pub struct PropertyAttributes {
    pub access: AccessLevel,
    pub set_access: AccessLevel,
    pub is_constant: bool,
    pub is_dependent: bool,
    pub is_transient: bool,
    pub is_hidden: bool,
}
```

Rules:

- Modules own class identities.
- Methods are ordinary `HirFunction`s.
- `HirClass.methods` stores method membership and metadata.
- Class properties, methods, events, enumerations, and arguments are structured semantic metadata, not opaque statement payloads.
- Runtime class registry metadata and language-facing class metadata should converge over time.
- Graphics handles, GPU arrays, data objects, and future runtime domains should use the same nominal dispatch surface rather than bespoke compiler special cases.
- Access control and property attributes should be available to diagnostics and LSP even when full enforcement is staged.
- Property defaults, property validation, dependent properties, static properties, constant properties, abstract/sealed/hidden attributes, constructors, events/listeners, enumeration values, display/conversion hooks, and operator/indexing overloads are class metadata.
- `class`, `isa`, `metaclass`, `properties`, and `methods` should query nominal metadata rather than ad hoc runtime strings.
- Tables, timetables, categorical arrays, graphics handles, GPU arrays, and domain objects should be modeled as nominal runtime classes when they need object-specific behavior.

## Type Model Direction

The type model should distinguish structural structs from nominal classes.

Target concepts:

- `Type::Struct { ... }` or equivalent for structural data
- class instance type keyed by `ClassId`
- handle class instance type keyed by `ClassId`, if handle/value distinction remains type-visible
- class reference or metaclass type keyed by `ClassId`

Rules:

- Runtime objects should not permanently degrade to `Type::Unknown`.
- Domain-specific types such as `DataDataset`, `DataArray`, and `DataTransaction` are transitional.
- Long term, runtime domain objects should be represented through nominal class metadata plus analysis hooks.

## Runtime Class Metadata Direction

Rust-defined runtime objects should eventually declare language-facing metadata once and project it into:

- runtime class registration
- nominal type identity
- member and method lookup
- analysis summaries
- LSP/docs/help metadata
- external bindings generation

Guardrail:

- RunMat should not expose arbitrary Rust implementation details as language semantics.
- Rust declarations should generate language-facing metadata consumed by RunMat's class/type system.

## Diagnostics Model

Diagnostics are first-class compiler products.

Target diagnostic structure:

- stable diagnostic code
- severity
- primary span
- secondary spans
- notes
- help text
- optional fix suggestions
- machine-readable category
- source/module/package context

Diagnostics should exist for:

- parser errors
- name resolution errors
- manifest/project errors
- import ambiguity
- type and shape warnings
- definite-assignment issues
- invalid input or output arity
- invalid comma-list use
- invalid assignment growth, deletion, or scalar expansion
- invalid `end` use or unsupported object `end` dispatch
- invalid condition type or non-scalar condition where scalar is required
- command syntax misuse
- resolution precedence ambiguity
- unsupported dynamic workspace/path mutation
- invalid `isolated` captures
- class/member lookup errors
- class access, dispatch, property validation, and overload resolution errors
- async/host-interaction misuse where applicable
- acceleration eligibility and fallback explanation where useful

Rules:

- Diagnostics should talk in source and project terms, not internal loader/layout terms.
- Local compiler IDs should be mapped back to source spans and stable symbols before presentation.
- LSP, CLI, tests, and desktop UI should consume the same diagnostic model.

## Incrementality And Caching

The target architecture should support incremental compilation and analysis.

Compiler products should be cacheable at useful boundaries:

- parsed source
- module source index
- HIR module
- MIR function body
- function summary
- class metadata
- analysis facts
- VM layout and bytecode where safe

Cache keys should be based on stable identities and content/config dependencies:

- package identity
- module path
- source content hash
- manifest hash
- dependency graph hash
- compiler/runtime version
- relevant feature/config flags

Rules:

- Local arena IDs are not cache keys by themselves.
- Function and module summaries should support invalidation by dependency edges.
- LSP should be able to recompute affected modules/functions without rebuilding the world.
- Runtime/provider-specific facts such as concrete device residency are not static analysis cache contents.

## Async And Host Interaction Model

RunMat already relies heavily on async execution. The compiler/runtime model should make async boundaries explicit enough for correctness, diagnostics, and host integration.

RunMat's user-facing async model should be Rust-like:

- ordinary MATLAB-style code remains sequential and runs unchanged
- async is explicit in user source
- async functions or async blocks produce lazy futures
- futures do not execute until awaited or spawned
- `await` polls a future to completion and returns its value or raises its error
- `spawn` schedules a future for concurrent execution and returns a task handle
- cancellation is cooperative and observed at poll/await boundaries

This is intentionally not JavaScript-like promise eager execution. Creating an async value should not by itself begin running user code or trigger side effects.

Async-capable operations include:

- provider-backed acceleration dispatch
- plotting and UI hooks
- user input and host interaction
- filesystem or package loading where asynchronous
- kernel/notebook/wasm host calls
- future async builtins or runtime services

Target model:

- HIR records calls and host-visible operations semantically.
- HIR represents user-facing async constructs explicitly, such as async function/block creation, `await`, and `spawn`.
- MIR marks calls or terminators that may suspend, await host services, or require async runtime scheduling.
- Function summaries include async/suspension behavior.
- VM/runtime lowering preserves async boundaries explicitly.
- Host APIs expose async execution as a first-class mode, not a special-case callback path.

Language-synchronous async remains separate from language-visible futures:

- ordinary builtins may internally suspend while still returning ordinary values to sequential user code
- explicit async functions/blocks return future values
- explicit `spawn` creates task handles and actual concurrency
- explicit `await` is the user-visible suspension point
- direct `await` does not by itself introduce concurrent access to another task
- `spawn` requires spawn-safe captures and values

Top-level execution rules:

- REPL, notebook, and script-like entrypoints normally set `EntrypointPolicy.top_level_await = true`.
- Ordinary function bodies may use `await` only when the function is declared async or is otherwise explicitly async-capable by source syntax.
- Calling an async function produces a future value; it does not run the function body immediately.
- Calling a language-synchronous builtin produces an ordinary value, even if the evaluator internally suspends while computing it.

Conceptual async facts:

```rust
pub enum AsyncBehaviorFact {
    NeverSuspends,
    MaySuspend,
    RequiresAsyncRuntime,
}

pub enum AsyncValueFact {
    NotAwaitable,
    Future { output: Box<TypeFact> },
    Task { output: Box<TypeFact> },
    RuntimeAwaitable { output: Box<TypeFact> },
}
```

Rules:

- Static analysis may classify async behavior conservatively.
- Async behavior is not purity; a pure computation may still be async if provider-backed.
- Synchronous hosts may choose fallback paths or blocking wrappers, but the semantic/runtime model should know where suspension can occur.
- Async boundaries are optimization barriers unless proven safe to reorder.
- No raw GC-managed pointers may live across an await; async frames must root live values explicitly.
- Futures/tasks preserve live locals, stack values, captures, and temporaries across suspension through rootable handles.
- Concurrent user code can only share mutable state through explicit synchronized or runtime-managed handles.
- Racy shared mutation must be rejected or confined to stable documented synchronization semantics; it must never become VM memory corruption.

## Analysis Model

HIR is semantic truth. Analysis stores inferred truth.

Analysis products should be keyed by stable semantic IDs:

- `BindingId`
- `ExprId`
- `FunctionId`
- `ModuleId`
- `ClassId` where relevant
- MIR block/local IDs where facts are purely MIR-local

Conceptual stores:

- `SemanticIndex`
- `BindingFact`
- `ExprFact`
- `FunctionSummary`
- `ModuleSummary`
- `TypeFact`
- `ShapeFact`
- `EffectSummary`
- `ExecutionFact`
- `AsyncBehaviorFact`
- `AsyncValueFact`
- `SpawnSafetyFact`
- `ValueFlowFact`
- `WorkspaceEffect`
- `EnvironmentEffect`
- function-handle facts
- MATLAB semantic facts for empty arrays, expansion, operators, numeric classes, strings/chars, aggregates, and control flow
- `TensorElementDomainFact`

### Analysis Passes

Pass 1: resolution and structural indexing.

- binding ownership tables
- function ownership and nesting
- capture relations
- resolved call identities
- reference and call classification
- function ABI indexing
- module/import resolution state
- block and statement traversal scaffolding

Pass 1.5: HIR-to-MIR lowering.

- CFG construction
- explicit block terminators
- normalized places and rvalues
- temporary local creation
- source mapping from MIR to HIR
- capture access lowering to explicit environment operations where appropriate
- requested-output and comma-list lowering
- explicit place mutation and indexing operation lowering
- async boundary marking
- spawn boundary marking

Pass 2: local flow analysis.

- flow environment keyed by `BindingId` and MIR locals as appropriate
- expression facts keyed by `ExprId`
- MIR-local facts keyed by MIR local/block IDs
- branch joins through lattices
- loop widening
- initialization tracking
- value-flow and comma-list tracking
- tensor element-domain tracking
- workspace effect tracking
- environment effect tracking
- async value and spawn-safety tracking

Pass 3: interprocedural summary propagation.

- function output summaries
- effect summaries
- capture read/write summaries
- resolved call edges
- builtin summary application
- requested-output-sensitive summary application
- workspace effect summary propagation
- environment effect summary propagation
- async/suspension summary propagation
- spawn-safety propagation across future/task boundaries
- recursive and mutually dependent call graph fixpoints where needed

## Type Facts

`TypeFact` is the analysis-facing language type fact. It should describe what kind of value an expression or binding denotes, not where that value happens to reside at runtime.

`runmat_builtins::Type` may remain as a compatibility projection during migration, especially for builtin signatures, but it should not remain the canonical analysis fact model.

Conceptually:

```rust
pub enum TypeFact {
    Never,
    Unknown,
    Scalar(ScalarTypeFact),
    Tensor(TensorTypeFact),
    LogicalArray,
    String,
    CharArray,
    Cell(CellTypeFact),
    Struct(StructTypeFact),
    ClassInstance(ClassInstanceFact),
    ClassRef(ClassId),
    Function(FunctionTypeFact),
    Tuple(Vec<TypeFact>),
    Union(Vec<TypeFact>),
}
```

Rules:

- `Never` means dead or impossible flow.
- `Unknown` means reachable but unknown.
- Joins are conservative and may widen to `Unknown` or `Union`.
- Structs are structural.
- Classes are nominal and reference `ClassId`.
- Tensor facts carry element domain, shape, and sparsity separately.
- Comma-separated lists are value-flow facts, not durable tensor or tuple types.
- Runtime placement does not change language type.
- `runmat_builtins::Type` can be converted into `(TypeFact, ShapeFact)` as a temporary bridge.

The target model intentionally separates language type, shape, execution eligibility, and runtime residency. A double tensor remains the same language type whether it is currently held as a host tensor or a GPU tensor.

## Shape Facts

Shape facts should be distinct from type facts.

Conceptually:

```rust
pub enum ShapeFact {
    Unreachable,
    Unknown,
    Scalar,
    Ranked { rank: usize },
    Shaped { dims: Vec<DimFact> },
}

pub enum DimFact {
    Known(usize),
    Symbolic(DimSymbol),
    Unknown,
}
```

Rules:

- Shape joins must never invent certainty.
- Unknown joined with anything becomes unknown.
- Equal known dimensions remain known.
- Conflicting known dimensions widen to unknown.
- Rank conflicts widen to unknown.

## Initialization Facts

Initialization should be explicit.

Conceptually:

```rust
pub enum InitFact {
    Unassigned,
    MaybeAssigned,
    DefinitelyAssigned,
}
```

Uses:

- definite-assignment diagnostics
- workspace export filtering
- safer branch and loop propagation

## Effect Facts

Effects should make hidden mutation and optimization barriers explicit.

Conceptually:

```rust
pub struct EffectSummary {
    pub reads_captures: BTreeSet<BindingId>,
    pub writes_captures: BTreeSet<BindingId>,
    pub reads_globals: BTreeSet<BindingId>,
    pub writes_globals: BTreeSet<BindingId>,
    pub reads_persistents: BTreeSet<BindingId>,
    pub writes_persistents: BTreeSet<BindingId>,
    pub reads_modules: BTreeSet<BindingId>,
    pub writes_modules: BTreeSet<BindingId>,
    pub may_call_unknown: bool,
    pub may_allocate: bool,
    pub async_behavior: AsyncBehaviorFact,
}
```

Uses:

- capture mutation tracking
- global, persistent, and module state summaries
- fusion barriers
- parallel safety decisions
- async scheduling and reordering barriers

## Fusion And Parallel Safety Facts

Initial classifications can be compact.

Conceptually:

```rust
pub enum FusibilityFact {
    Unknown,
    Fusible,
    NonFusible(FuseBlocker),
}

pub enum ParallelSafetyFact {
    Unknown,
    Safe,
    ReadsSharedState,
    WritesSharedState,
}

pub enum AccelEligibilityFact {
    Unknown,
    Ineligible(FuseBlocker),
    Eligible,
    Preferred,
}

pub enum DataMovementPolicyHint {
    Unknown,
    KeepHost,
    KeepDeviceIfAlreadyThere,
    PreferDeviceForLargeInputs,
}
```

These facts should be derived from semantic HIR and analysis summaries, not from a separate optimizer-only semantic model.

Important boundary:

- semantic analysis may say an expression is fusible, device-eligible, device-beneficial, or safe to execute in parallel
- semantic analysis must not claim that a concrete value is resident on a specific runtime device or backed by a specific provider buffer
- concrete provider choice, promotion, download, materialization, and buffer reuse are runtime/provider decisions

This matters because analysis may run on a different machine from execution. Device availability, memory pressure, precision support, provider choice, and concrete residency are runtime facts.

## Runtime Residency And Provider State

Runtime residency is concrete execution state, not semantic HIR state.

Examples:

- `Value::Tensor` is host-resident runtime data.
- `Value::GpuTensor(GpuTensorHandle)` is device-resident runtime data.
- `GpuTensorHandle.device_id` and `GpuTensorHandle.buffer_id` are runtime/provider identities.
- WGPU buffer pools and reuse classes are provider-internal allocation details.

Rules:

- `TypeFact` does not encode runtime residency.
- `ShapeFact` does not encode runtime residency.
- `AccelEligibilityFact` and `DataMovementPolicyHint` may guide runtime planning but do not guarantee actual placement.
- The runtime planner owns concrete placement and materialization decisions.
- The provider owns buffer allocation, pooling, pipeline caches, and device-specific dispatch details.

## Workspace Model

Workspace state is semantic.

The workspace is a mapping from semantic binding identity to current user-visible value.

Rules:

- Only bindings classified as workspace-visible may enter the workspace.
- Ordinary function locals never appear in the ordinary workspace inspector.
- Nested-function captures do not create duplicate workspace variables.
- Runtime slots do not define workspace visibility.
- `ans` is represented by an implicit binding.
- Globals and persistents are semantically distinct storage classes and may be exposed according to host policy.

### Script-Like Entrypoint Execution

- Assigned `TopLevel` bindings become workspace export candidates.
- Hidden locals remain hidden.
- Helper-function locals remain hidden.
- Captured bindings do not duplicate.
- `ans` appears if implicit result rules produce it.

### Function-Oriented Entrypoint Execution

- Function params, outputs, and locals remain hidden.
- Workspace changes happen only through module-visible bindings, globals, persistents, or host-surfaced return values.

### REPL Or Notebook Execution

- Submitted code lowers to a synthetic entry function.
- Assigned `TopLevel` bindings export to the session workspace.
- Re-running snippets updates semantic workspace bindings.

## VM Relationship

The VM remains slot-based internally for performance.

Rules:

- VM slots are derived from `BindingId` and `FunctionId` layout metadata.
- Bytecode function tables should use semantic function identity internally.
- Function display names remain available for diagnostics and dynamic behavior.
- Runtime workspace export maps from semantic bindings through VM layout to final values.
- VM slot indexes are not semantic IDs.
- VM bytecode and frame layout are derived from MIR plus semantic binding/layout maps.
- Async-capable bytecode paths preserve suspension boundaries discovered in MIR/analysis.
- VM or runtime execution graphs may temporarily continue to build fusion candidates from bytecode and runtime metadata during migration.
- Follow-up work should move fusion candidate construction toward semantic HIR plus `AnalysisStore`, while keeping concrete placement/provider decisions at runtime.

## Imports And Composition Direction

The final project composition model is config-driven.

Source-level `import` controls name visibility and ergonomics. It is not the dependency mechanism.

Target config concepts:

- `[package]`
- `[sources]`
- `[dependencies]`
- `[[entrypoints]]`

Rules:

- Local source roots define local package source discovery.
- Dependencies define external package availability.
- Imports expose names and support unqualified resolution.
- Canonical internal identities remain qualified and stable.
- Scripts disappear internally into entrypoints.

This composition model is follow-on work after the core semantic HIR migration unless explicitly pulled forward.

## Target Examples

### Script-Like Top Level

```matlab
x = rand(10, 10);
y = x * 2;
disp(y);
```

Target lowering:

- one module
- one synthetic entry function
- one entrypoint targeting that function
- `x` and `y` are top-level workspace-visible bindings
- `disp(y)` is a call and does not create workspace state

### Script Plus Local Function

```matlab
x = main(3);

function y = main(n)
    y = n + 1;
end
```

Target lowering:

- top-level executable code becomes synthetic entry function
- `main` is a module-owned `HirFunction`
- call resolves to `FunctionId(main)`
- `x` is workspace-visible
- `n` and `y` inside `main` are hidden

### Nested Shared Capture

```matlab
function y = outer(a)
    acc = 0;

    function bump(x)
        acc = acc + x;
    end

    bump(a);
    y = acc;
end
```

Target lowering:

- `acc` is one binding owned by `outer`
- `bump` captures `acc`
- mutation in `bump` targets the same `BindingId`

### Isolated Nested Function

```matlab
function y = outer(a)
    z = a + 1;

    isolated function w = inner(x, z)
        w = x + z;
    end

    y = inner(a, z);
end
```

Target lowering:

- `inner` is a normal function with `isolated = true`
- `inner` has no captures
- required outer values are passed explicitly

### Anonymous Function With Capture

```matlab
function f = outer(a)
    f = @(x) x + a;
end
```

Target lowering:

- anonymous function becomes a real `HirFunction`
- expression references it by `FunctionId`
- anonymous function captures `a`

### Requested Outputs And Destructuring

```matlab
A = rand(5, 3);
[rows, cols] = size(A);
[~, idx] = max(A(:, 1));
```

Target lowering:

- `[rows, cols] = size(A)` requests exactly two outputs from `size`
- `rows` and `cols` are output targets in an `OutputTargetList`
- `[~, idx] = max(...)` requests exactly two outputs and marks the first output as discarded
- builtin summaries may depend on requested output count
- VM call ABI returns an output list that assignment destructures into targets

### Variadics And Comma Lists

```matlab
function h = myplot(x, y, varargin)
    h = plot(x, y, varargin{:});
end
```

Target lowering:

- `FunctionAbi` records fixed inputs `x` and `y` plus variadic input binding `varargin`
- excess caller arguments are packed into `varargin` as a cell array
- `varargin{:}` is an explicit comma-list expansion site
- the `plot` call consumes the expanded argument list

### Assignment Growth And Indexing

```matlab
x(3) = 10;
A(:, end) = 0;
s.a.b = 1;
```

Target lowering:

- `x(3) = 10` is indexed assignment with array-creation/growth policy
- `A(:, end) = 0` preserves colon and symbolic `end` indexing components until operand/dimension context resolves them
- `s.a.b = 1` is member assignment with struct field-path creation policy when `s` is structural
- class/object assignment may instead dispatch through `subsasgn` metadata

### Sequential Code With Internal Suspension

```matlab
data = webread(url);
result = process(data);
```

Target lowering:

- `webread(url)` is a language-synchronous builtin call
- user code observes ordinary sequential execution
- runtime execution may internally suspend while the host request completes
- no user-visible future or task value is created
- `process(data)` runs after `webread` has produced an ordinary value

### Direct Await Without Spawn

```matlab
async function y = fetch_and_process(url)
    data = webread(url);
    y = process(data);
end

result = await(fetch_and_process(url));
```

Target lowering:

- `fetch_and_process(url)` creates a lazy future value
- the async function body does not run at call creation
- `await` polls that future in the current execution context
- no task handle is created
- no concurrent work is introduced
- ordinary lexical captures remain valid because execution is not concurrent
- execution remains sequential, but may suspend and resume at the await point

### Explicit Async Concurrency

```matlab
async function y = fetch_and_process(url)
    data = webread(url);
    y = process(data);
end

t1 = spawn(fetch_and_process(url1));
t2 = spawn(fetch_and_process(url2));

a = await(t1);
b = await(t2);
```

Target lowering:

- `fetch_and_process(url1)` and `fetch_and_process(url2)` create lazy future values
- calling the async function does not execute its body immediately
- `spawn` schedules each future and returns task handles
- each spawned future must satisfy spawn-safety rules
- captured inputs are copied, immutable, or explicit synchronized/runtime handles
- `await(t1)` and `await(t2)` are explicit suspension points
- MIR records await resume points and values live across each suspension
- task/future frames root live values until completion, cancellation, or drop

### Rejected Spawn With Mutable Capture

```matlab
function y = run_two()
    acc = 0;

    async function bump()
        acc = acc + 1;
    end

    t1 = spawn(bump());
    t2 = spawn(bump());

    await(t1);
    await(t2);
    y = acc;
end
```

Target behavior:

- `bump` captures and mutates `acc`
- direct `await(bump())` would be sequentially valid
- `spawn(bump())` is rejected because it would allow concurrent mutation of a parent lexical binding
- the user should pass values explicitly and combine results, or use an explicit synchronized/runtime-managed shared object

### Class Method

```matlab
classdef MyThing
    methods
        function y = twice(obj, x)
            y = x * 2;
        end
    end
end
```

Target lowering:

- `MyThing` becomes a `HirClass`
- `twice` becomes a `HirFunction`
- `HirClass.methods` references `twice` by `FunctionId`
