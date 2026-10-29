//! First-party fused-cache ownership. Driver/compiler-private allocations are
//! deliberately outside these exact known-layout host claims.
mod owner;
mod table;
pub(super) use owner::{PipelineAllocation, PipelineOwner};
pub(super) use table::{CacheLookup, PipelineCache};

#[cfg(test)]
mod tests;
