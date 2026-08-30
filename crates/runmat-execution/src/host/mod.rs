//! Typed execution-host compatibility contracts.
//!
//! Program construction records requirements here; schedulers compare those
//! requirements with worker inventories before allocating an attempt.

mod compatibility;
mod target;

pub use compatibility::{
    ExecutionHostInventory, ExecutionHostRequirement, ExecutionHostTarget, ForeignAdapterInventory,
    EXECUTION_HOST_SCHEMA_VERSION,
};
pub use target::{
    NativeAbi, NativeArchitecture, NativeObjectFormat, NativeOperatingSystem, NativeTargetIdentity,
};
