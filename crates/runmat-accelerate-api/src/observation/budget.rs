//! Provider-owned bound on in-flight instrumentation, not device scheduling.
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

pub struct ProviderObservationBudget {
    limit: NonZeroU32,
    used: AtomicU32,
}
impl ProviderObservationBudget {
    pub fn new(limit: NonZeroU32) -> Self {
        Self {
            limit,
            used: AtomicU32::new(0),
        }
    }
    pub fn limit(&self) -> u32 {
        self.limit.get()
    }
    pub(super) fn acquire(self: &Arc<Self>) -> Option<Permit> {
        self.used
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                (used < self.limit.get()).then(|| used + 1)
            })
            .ok()?;
        Some(Permit(self.clone()))
    }
}
pub(super) struct Permit(Arc<ProviderObservationBudget>);
impl Drop for Permit {
    fn drop(&mut self) {
        self.0.used.fetch_sub(1, Ordering::AcqRel);
    }
}
