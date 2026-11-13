//! Working allocation ownership for one kernel. All views share the same stage
//! ceiling and work counters; only the live returned records cross publication.
use super::GeometryUsageCounters;
use std::sync::{
    atomic::{AtomicU64, Ordering::Relaxed},
    Arc,
};

pub(crate) struct AllocationPhase {
    usage: Arc<GeometryUsageCounters>,
    held: AtomicU64,
}
impl AllocationPhase {
    pub(crate) fn new(usage: Arc<GeometryUsageCounters>) -> Self {
        Self {
            usage,
            held: AtomicU64::new(0),
        }
    }
    pub(super) fn record(&self, bytes: u64) {
        self.held
            .fetch_update(Relaxed, Relaxed, |held| held.checked_add(bytes))
            .expect("phase storage exceeds its admitted stage inventory");
    }
    pub(super) fn release(&self, bytes: u64) {
        self.held
            .fetch_update(Relaxed, Relaxed, |held| held.checked_sub(bytes))
            .expect("released storage not owned by this kernel phase");
    }
    pub(crate) fn transfer_records(&self, bytes: u64) -> Option<RecordStorage> {
        self.held
            .fetch_update(Relaxed, Relaxed, |held| held.checked_sub(bytes))
            .ok()?;
        Some(RecordStorage {
            usage: Arc::clone(&self.usage),
            bytes,
        })
    }
}
impl Drop for AllocationPhase {
    fn drop(&mut self) {
        release(&self.usage, self.held.load(Relaxed));
    }
}
/// Retains already admitted record capacities until their actual last use. It
/// does not include chunk buffers, metadata, or claim measured process residency.
pub(crate) struct RecordStorage {
    usage: Arc<GeometryUsageCounters>,
    bytes: u64,
}
impl Drop for RecordStorage {
    fn drop(&mut self) {
        release(&self.usage, self.bytes);
    }
}
fn release(usage: &GeometryUsageCounters, bytes: u64) {
    usage
        .allocation_bytes
        .fetch_update(Relaxed, Relaxed, |live| live.checked_sub(bytes))
        .expect("released an unowned stage allocation charge");
}
