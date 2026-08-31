use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPortability {
    NativeAndWasm,
    NativeOnly,
    WasmHostBridge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinAcceleratorPolicy {
    Forbidden,
    Optional,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinResidencyPolicy {
    Host,
    PreserveInputs,
    ProduceResident,
    GatherToHost,
    Dynamic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinFusionPolicy {
    Never,
    Candidate,
    Boundary,
}

/// How a builtin may consume execution-owned distributed values.
///
/// This is intentionally narrower than accelerator residency. Each admitted
/// form names an execution strategy that the distributed runtime can validate
/// without inferring semantics from a builtin's spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinDistributedPolicy {
    Unsupported,
    /// Pass execution-owned handles to a builtin whose declared purpose is
    /// metadata or identity inspection. The builtin must not read partitions.
    InspectHandles,
    MaterializeArguments,
    MapUnary,
    /// Construct a distributed scalar whose class, complexity, sparsity, and
    /// distribution owner are selected by a distributed `like` prototype.
    ScalarLikePrototype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinPlacementContract {
    pub portability: BuiltinPortability,
    pub accelerator: BuiltinAcceleratorPolicy,
    pub residency: BuiltinResidencyPolicy,
    pub fusion: BuiltinFusionPolicy,
    pub distributed: BuiltinDistributedPolicy,
}
