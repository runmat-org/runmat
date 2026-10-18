//! Typed execution-host compatibility contracts.
//!
//! Program construction records requirements here; schedulers compare those
//! requirements with worker inventories before allocating an attempt.

mod compatibility;
mod target;
mod native_abi;
mod native_accelerator_profile;
mod native_deployment;
mod native_probe;
mod native_policy;
mod selected_target;

pub use native_abi::NativeAbiBinding;
pub use native_accelerator_profile::{
    NativeAcceleratorProfile, NATIVE_ACCELERATOR_PROFILE_BYTES, NATIVE_ACCELERATOR_PROFILE_SYMBOL,
};
pub use native_deployment::*;
pub use native_policy::{
    NativeNumericalBehavior, NativeNumericalPolicy, NativeOptimization,
    NATIVE_NUMERICAL_POLICY_SCHEMA_VERSION,
};
pub use native_probe::*;
pub use selected_target::{
    NativeCpuBaseline, NativeEndianness, NativeEnvironment, SelectedNativeTarget,
    SELECTED_NATIVE_TARGET_SCHEMA_VERSION,
};

pub use compatibility::{
    ExecutionHostInventory, ExecutionHostRequirement, ExecutionHostTarget, ForeignAdapterInventory,
    SelectedUnitHostRequirement, EXECUTION_HOST_SCHEMA_VERSION,
};
pub use target::{
    NativeAbi, NativeArchitecture, NativeObjectFormat, NativeOperatingSystem, NativeTargetIdentity,
};
