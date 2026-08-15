//! Exact native-frame materialization and resume policy.

mod materialize;
mod policy;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use materialize::NativeMaterializationContext;
pub use materialize::{MaterializedFrame, MaterializedLocal, ResumeSite};
pub use policy::{DeoptimizationPolicy, FaultInjection, ResumeTarget};
