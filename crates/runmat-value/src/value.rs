use crate::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Int(IntValue),
    Num(f64),
    /// Complex scalar value represented as (re, im)
    Complex(f64, f64),
    Bool(bool),
    // Logical array (N-D of booleans). Scalars use Bool.
    LogicalArray(LogicalArray),
    String(String),
    // String array (R2016b+): N-D array of string scalars
    StringArray(StringArray),
    // Char array (single-quoted): N-D character array with 2-D row/column caches
    CharArray(CharArray),
    Tensor(Tensor),
    /// Real sparse matrix in compressed sparse column form.
    SparseTensor(SparseTensor),
    /// Complex numeric array; same column-major shape semantics as `Tensor`
    ComplexTensor(ComplexTensor),
    /// Scalar symbolic expression used by `sym`, `syms`, and symbolic math builtins.
    Symbolic(SymbolicExpr),
    /// Dense symbolic array with column-major shape semantics.
    SymbolicArray(SymbolicArray),
    Cell(CellArray),
    /// Scalar structure value. Empty and nonscalar structure arrays use
    /// [`Value::StructArray`].
    Struct(StructValue),
    /// Empty or nonscalar structure array with an explicit homogeneous schema.
    StructArray(StructArray),
    // GPU-resident tensor handle (opaque; buffer managed by backend)
    GpuTensor(runmat_accelerate_api::GpuTensorHandle),
    // Simple object instance until full class system lands
    Object(ObjectInstance),
    /// Homogeneous N-D value/handle object array with column-major storage.
    ObjectArray(ObjectArray),
    /// Handle-object wrapper providing identity semantics and validity tracking
    HandleObject(HandleRef),
    /// Event listener handle for events
    Listener(Listener),
    /// Multiple outputs captured as a list (internal destructuring helper)
    OutputList(Vec<Value>),
    // Function handle pointing to a named function (builtin or user)
    FunctionHandle(String),
    // Function handle whose resolution must stay at the external boundary.
    ExternalFunctionHandle(String),
    // Function handle preserving typed method identity.
    MethodFunctionHandle(String),
    // Function handle with compiler/session semantic identity.
    BoundFunctionHandle {
        name: String,
        function: usize,
    },
    Closure(Closure),
    ClassRef(runmat_types::ClassIdentity),
    MException(MException),
    /// Lazy executable work owned by the active runtime execution service.
    Future(runmat_execution::FutureHandle),
    /// Scheduled work owned by the active runtime execution service.
    Task(runmat_execution::TaskHandle),
    /// Execution pool capability owned by an execution service.
    Pool(runmat_execution::PoolHandle),
    /// Durable batch execution capability.
    Job(runmat_execution::JobHandle),
    /// A distributed array whose partitions remain owned by an execution
    /// service. The handle carries identity and fencing, never array payloads.
    Distributed(Box<runmat_execution::DistributedValueHandle>),
    /// Per-rank values retained by an SPMD gang. Entries are resolved through
    /// the owning execution service rather than embedded in the language value.
    Composite(Box<runmat_execution::CompositeHandle>),
    /// Opaque resource owned by a foreign runtime host and fenced by generation.
    Foreign(ForeignRef),
}
