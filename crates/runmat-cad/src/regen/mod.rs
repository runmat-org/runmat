//! Regeneration planning and execution.
//!
//! - [`deps`]: static dependency analysis (inputs and configurations → features and items).
//! - [`hash`]: chained-hash cache keys.
//! - [`kernel`]: the `CadKernel` seam and canonical operations.
//! - [`engine`]: regeneration with naming, resolution and continue-past-failure.
//! - [`fake`]: a deterministic grid kernel for tests.

pub mod deps;
pub mod engine;
pub mod fake;
pub mod hash;
pub mod kernel;

pub use deps::{Affected, DependencyGraph};
pub use engine::{
    regenerate, FeatureOutcome, FeatureStatus, ItemOutcome, LiveBody, PartState, ReferenceRecord,
    RegenCache, RegenContext, RegenRequest, RegenResult,
};
pub use fake::{FakeImportBox, FakeKernel};
pub use hash::CacheKey;
pub use kernel::{BodyHandle, CadKernel, KernelError, KernelId, KernelOp, KernelOutput};
