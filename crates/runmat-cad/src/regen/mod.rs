//! Regeneration planning and execution.
//!
//! - [`deps`]: static dependency analysis (inputs and configurations → features and items).
//! - [`hash`]: chained-hash cache keys.
//! - [`kernel`]: the `CadKernel` seam and canonical operations.
//! - [`engine`]: regeneration with naming, resolution and continue-past-failure.
//! - [`slots`]: body slots and their lineage signatures across regenerations.
//! - [`fake`]: a deterministic grid kernel for tests.

pub mod deps;
pub mod engine;
pub mod fake;
pub mod hash;
pub mod kernel;
pub mod slots;

pub use deps::{Affected, DependencyGraph};
pub use engine::{
    regenerate, regenerate_metered, FeatureOutcome, FeatureStatus, ItemOutcome, LiveBody,
    PartState, ReferenceRecord, RegenCache, RegenContext, RegenRequest, RegenResult,
};
pub use fake::{FakeImportBox, FakeKernel};
pub use hash::CacheKey;
pub use kernel::{BodyHandle, CadKernel, KernelError, KernelId, KernelOp, KernelOutput};
