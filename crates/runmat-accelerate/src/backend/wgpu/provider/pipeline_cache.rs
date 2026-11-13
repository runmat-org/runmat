//! First-party fused-cache ownership. Driver/compiler-private allocations are
//! deliberately outside these exact known-layout host claims.
#[path = "pipeline_cache/owner.rs"]
mod owner;
#[path = "pipeline_cache/table.rs"]
mod table;
pub(super) use owner::{PipelineAllocation, PipelineOwner};
pub(super) use table::{CacheLookup, PipelineCache};

#[cfg(test)]
#[path = "pipeline_cache/tests.rs"]
mod tests;
