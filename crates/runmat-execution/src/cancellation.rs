//! Live cooperative control shared by hosts, domain adapters and providers.
//! Original flags remain shared; a deadline is evaluated at each checkpoint.
use crate::CancellationReason;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

#[derive(Clone, Debug)]
pub struct CancellationToken {
    state: Arc<CancellationState>,
}

#[derive(Clone, Debug)]
struct CancellationState {
    primary: Arc<AtomicBool>,
    inherited: Vec<Arc<AtomicBool>>,
    deadline_unix_ms: Option<u64>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::from_shared(Arc::new(AtomicBool::new(false)))
    }
}

impl CancellationToken {
    pub fn from_shared(primary: Arc<AtomicBool>) -> Self {
        Self {
            state: Arc::new(CancellationState {
                primary,
                inherited: Vec::new(),
                deadline_unix_ms: None,
            }),
        }
    }

    /// Scoped extension preserves original authorities; adapters clone one Arc.
    pub fn with_source(mut self, source: Arc<AtomicBool>) -> Self {
        if !Arc::ptr_eq(&self.state.primary, &source)
            && !self
                .state
                .inherited
                .iter()
                .any(|flag| Arc::ptr_eq(flag, &source))
        {
            Arc::make_mut(&mut self.state).inherited.push(source);
        }
        self
    }

    /// Enter a child owner without losing the caller's original live controls.
    pub fn with_parent(mut self, parent: &Self) -> Self {
        self = self.with_source(parent.primary_flag());
        for source in &parent.state.inherited {
            self = self.with_source(Arc::clone(source));
        }
        self.with_deadline(parent.state.deadline_unix_ms)
    }

    pub fn with_deadline(mut self, deadline_unix_ms: Option<u64>) -> Self {
        let deadline = match (self.state.deadline_unix_ms, deadline_unix_ms) {
            (Some(parent), Some(child)) => Some(parent.min(child)),
            (parent, child) => parent.or(child),
        };
        if deadline != self.state.deadline_unix_ms {
            Arc::make_mut(&mut self.state).deadline_unix_ms = deadline;
        }
        self
    }

    /// Explicit mutation addresses the original primary cancellation authority.
    pub fn cancel(&self) {
        self.state.primary.store(true, Ordering::Release);
    }

    pub fn primary_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.state.primary)
    }

    fn flag_reason(&self) -> Option<CancellationReason> {
        (self.state.primary.load(Ordering::Acquire)
            || self
                .state
                .inherited
                .iter()
                .any(|flag| flag.load(Ordering::Acquire)))
        .then_some(CancellationReason::User)
    }

    pub fn stop_reason_at(&self, unix_ms: u128) -> Option<CancellationReason> {
        self.flag_reason().or_else(|| {
            self.state
                .deadline_unix_ms
                .filter(|deadline| unix_ms >= u128::from(*deadline))
                .map(|_| CancellationReason::Deadline)
        })
    }

    pub fn stop_reason(&self) -> Option<CancellationReason> {
        self.flag_reason().or_else(|| {
            self.state.deadline_unix_ms.and_then(|deadline| {
                (runmat_time::unix_timestamp_ms() >= u128::from(deadline))
                    .then_some(CancellationReason::Deadline)
            })
        })
    }

    pub fn is_cancelled(&self) -> bool {
        self.stop_reason().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clones_share_inventory_and_child_extension_does_not_change_parent() {
        let parent = CancellationToken::default();
        let clone = parent.clone();
        assert!(Arc::ptr_eq(&parent.state, &clone.state));
        let child = clone.with_deadline(Some(0));
        assert!(!Arc::ptr_eq(&parent.state, &child.state));
        assert_eq!(parent.stop_reason_at(0), None);
        assert_eq!(child.stop_reason_at(0), Some(CancellationReason::Deadline));
    }

    #[test]
    fn original_sources_remain_live_and_deduplicated_after_clone() {
        let session = Arc::new(AtomicBool::new(false));
        let caller = Arc::new(AtomicBool::new(false));
        let token = CancellationToken::from_shared(session.clone())
            .with_source(caller.clone())
            .with_source(caller.clone())
            .with_source(session.clone());
        assert_eq!(token.state.inherited.len(), 1);
        let child = token.clone();
        assert_eq!(child.stop_reason_at(0), None);
        caller.store(true, Ordering::Release);
        assert_eq!(child.stop_reason_at(0), Some(CancellationReason::User));
        assert!(!session.load(Ordering::Acquire));
        caller.store(false, Ordering::Release);
        token.cancel();
        assert_eq!(child.stop_reason_at(0), Some(CancellationReason::User));
    }
    #[test]
    fn child_cannot_extend_or_clear_parent_deadline() {
        let parent = CancellationToken::default().with_deadline(Some(100));
        for child in [
            parent.clone().with_deadline(Some(200)),
            parent.clone().with_deadline(None),
        ] {
            assert_eq!(child.stop_reason_at(99), None);
            assert_eq!(
                child.stop_reason_at(100),
                Some(CancellationReason::Deadline)
            );
        }
        let child = parent.with_deadline(Some(50));
        assert_eq!(child.stop_reason_at(50), Some(CancellationReason::Deadline));
        assert!(!child.primary_flag().load(Ordering::Acquire));
    }
}
