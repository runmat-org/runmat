use crate::callable::CallableInstanceKind;
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
    /// Scalar string element, with explicit missingness and an immutable text owner.
    String(StringElement),
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
    OutputList(OutputListStorage),
    // Function handle pointing to a named function (builtin or user)
    FunctionHandle(NamedFunctionHandle),
    // Function handle whose resolution must stay at the external boundary.
    ExternalFunctionHandle(NamedFunctionHandle),
    // Function handle preserving typed method identity.
    MethodFunctionHandle(NamedFunctionHandle),
    // Function handle with compiler/session semantic identity.
    BoundFunctionHandle(BoundFunctionHandle),
    Closure(Closure),
    /// Invocation-local callable capability; no source name or executable body.
    ScopedFunctionHandle(ScopedFunctionHandle),
    ClassRef(SharedClassIdentity),
    MException(MException),
    /// Lazy executable work owned by the active runtime execution service.
    Future(SharedFutureHandle),
    /// Scheduled work owned by the active runtime execution service.
    Task(SharedTaskHandle),
    /// Execution pool capability owned by an execution service.
    Pool(runmat_execution::PoolHandle),
    /// Durable batch execution capability.
    Job(SharedJobHandle),
    /// A distributed array whose partitions remain owned by an execution
    /// service. The handle carries identity and fencing, never array payloads.
    Distributed(SharedDistributedHandle),
    /// Per-rank values retained by an SPMD gang. Entries are resolved through
    /// the owning execution service rather than embedded in the language value.
    Composite(SharedCompositeHandle),
    /// Opaque resource owned by a foreign runtime host and fenced by generation.
    Foreign(ForeignRef),
}

impl Value {
    /// Language function-handle class, including opaque invocation capabilities.
    pub fn is_function_handle(&self) -> bool {
        matches!(
            self,
            Self::FunctionHandle(_)
                | Self::ExternalFunctionHandle(_)
                | Self::MethodFunctionHandle(_)
                | Self::BoundFunctionHandle(_)
                | Self::Closure(_)
                | Self::ScopedFunctionHandle(_)
        )
    }

    pub fn string(element: impl Into<StringElement>) -> Self {
        Self::String(element.into())
    }

    pub fn string_text(&self) -> Option<&str> {
        match self {
            Self::String(text) => text.as_str(),
            _ => None,
        }
    }

    pub fn output_list(values: impl Into<OutputListStorage>) -> Self {
        Self::OutputList(values.into())
    }

    pub fn function_handle(name: String) -> Self {
        Self::FunctionHandle(NamedFunctionHandle::new(name))
    }

    pub fn external_function_handle(name: String) -> Self {
        Self::ExternalFunctionHandle(NamedFunctionHandle::new(name))
    }

    pub fn method_function_handle(name: String) -> Self {
        Self::MethodFunctionHandle(NamedFunctionHandle::new(name))
    }

    pub fn bound_function_handle(
        name: String,
        function: usize,
        authority: BoundFunctionAuthority,
        source_role: runmat_types::FunctionSourceRole,
        source_file: Option<String>,
    ) -> Self {
        Self::BoundFunctionHandle(BoundFunctionHandle::new(
            name,
            function,
            authority,
            source_role,
            source_file,
        ))
    }

    pub fn callable_instance_key(&self) -> Option<CallableInstanceKey> {
        match self {
            Self::FunctionHandle(handle) => Some(CallableInstanceKey::new(
                CallableInstanceKind::Workspace,
                handle.instance_id(),
            )),
            Self::ExternalFunctionHandle(handle) => Some(CallableInstanceKey::new(
                CallableInstanceKind::External,
                handle.instance_id(),
            )),
            Self::MethodFunctionHandle(handle) => Some(CallableInstanceKey::new(
                CallableInstanceKind::Method,
                handle.instance_id(),
            )),
            Self::BoundFunctionHandle(handle) => Some(CallableInstanceKey::new(
                CallableInstanceKind::Bound,
                handle.instance_id(),
            )),
            Self::ScopedFunctionHandle(handle) => Some(CallableInstanceKey::new(
                CallableInstanceKind::Scoped,
                handle.instance_id(),
            )),
            Self::Closure(closure) => Some(CallableInstanceKey::new(
                CallableInstanceKind::Closure,
                closure.instance_id(),
            )),
            _ => None,
        }
    }
}
