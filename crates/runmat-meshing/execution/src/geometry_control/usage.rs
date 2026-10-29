//! One admitted inventory shared by every view of a stage.
use std::sync::atomic::AtomicU64;

#[derive(Default)]
pub(crate) struct GeometryUsageCounters {
    pub(super) allocation_bytes: AtomicU64,
    pub(super) peak_allocation_bytes: AtomicU64,
    pub(super) search_work: AtomicU64,
    pub(super) iterations: AtomicU64,
    /// Geometry search work observed at the latest exact-geometry probe of this stage.
    pub(super) search_work_at_last_probe: AtomicU64,
}
