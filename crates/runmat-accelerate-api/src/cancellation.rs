use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// A cheap shareable cancellation flag for one provider operation.
///
/// This is data rather than a callback, so numerical providers can poll it at declared
/// cooperative checkpoints without invoking runtime or domain code.
#[derive(Clone, Debug, Default)]
pub struct ProviderCancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl ProviderCancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_shared(cancelled: Arc<AtomicBool>) -> Self {
        Self { cancelled }
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}
