use std::sync::{atomic::AtomicBool, Arc};

/// A cheap shareable cancellation flag for one provider operation.
///
/// This is data rather than a callback, so numerical providers can poll it at declared
/// cooperative checkpoints without invoking runtime or domain code.
#[derive(Clone, Debug, Default)]
pub struct ProviderCancellationToken {
    cancelled: runmat_execution::CancellationToken,
}

impl ProviderCancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_shared(cancelled: Arc<AtomicBool>) -> Self {
        Self {
            cancelled: runmat_execution::CancellationToken::from_shared(cancelled),
        }
    }

    pub fn from_token(cancelled: runmat_execution::CancellationToken) -> Self {
        Self { cancelled }
    }

    pub fn cancel(&self) {
        self.cancelled.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.is_cancelled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn adapter_keeps_live_parent_and_deadline_without_mutating_primary_flag() {
        let owner = Arc::new(AtomicBool::new(false));
        let caller = Arc::new(AtomicBool::new(false));
        let shared = runmat_execution::CancellationToken::from_shared(owner.clone())
            .with_source(caller.clone());
        let adapter = ProviderCancellationToken::from_token(shared.clone());
        assert!(!adapter.is_cancelled());
        caller.store(true, std::sync::atomic::Ordering::Release);
        assert!(adapter.is_cancelled());
        assert!(!owner.load(std::sync::atomic::Ordering::Acquire));
        caller.store(false, std::sync::atomic::Ordering::Release);
        assert!(
            ProviderCancellationToken::from_token(shared.with_deadline(Some(0))).is_cancelled()
        );
        assert!(!owner.load(std::sync::atomic::Ordering::Acquire));
    }
}
