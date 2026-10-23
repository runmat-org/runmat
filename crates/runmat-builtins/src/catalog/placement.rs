use runmat_types::NumericClass;
use serde::Serialize;

mod preparation;
pub use preparation::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinPortability {
    NativeAndWasm,
    NativeOnly,
    WasmHostBridge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinAcceleratorPolicy {
    /// No accelerator computation placement. Host inspection, transfer, and
    /// resident alias/container preservation are independent of this policy.
    Forbidden,
    Optional,
    Required,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BuiltinResidencyPolicy {
    Host,
    /// Preserve input residency, including aliases nested in host containers;
    /// this does not imply accelerator computation or a new device allocation.
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
    /// Unary partition mapping for a value input, plus the existing
    /// distributed scalar-like construction for a `like` prototype.
    MapUnaryOrScalarLikePrototype,
    /// Apply an elementwise unary builtin to each partition after validating
    /// placement-specific limits that are narrower than its host contract.
    MapUnaryConstrained(BuiltinDistributedMapContract),
    /// Construct a distributed scalar whose class, complexity, sparsity, and
    /// distribution owner are selected by a distributed `like` prototype.
    ScalarLikePrototype,
}

/// Additional admission rules for partition-local unary execution.
///
/// The builtin's inference rule remains authoritative for its ordinary input
/// contract. This descriptor records only restrictions introduced by the
/// distributed execution surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinDistributedMapContract {
    pub numeric_classes: &'static [NumericClass],
}

impl BuiltinDistributedMapContract {
    pub const fn numeric_classes(numeric_classes: &'static [NumericClass]) -> Self {
        Self { numeric_classes }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BuiltinPlacementContract {
    /// Operand preparation is independent of inference and backend eligibility.
    pub preparation: BuiltinOperandPreparation,
    pub portability: BuiltinPortability,
    pub accelerator: BuiltinAcceleratorPolicy,
    pub residency: BuiltinResidencyPolicy,
    pub fusion: BuiltinFusionPolicy,
    pub distributed: BuiltinDistributedPolicy,
}
