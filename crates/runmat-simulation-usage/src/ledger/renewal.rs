//! Host grant acquisition at batch boundaries, outside the numerical-state lock.
use super::{LedgerError, UsageLedger, Weak};
use crate::OperationCount;

/// The admitted host obtains cumulative capacity and updates the same ledger.
/// Authentication, transport and any original denial cause stay with that host.
/// Numerical kernels neither select a provider nor hold a ledger lock across I/O.
pub trait GrantRenewal: Send + Sync {
    fn request_grant(&self, required: OperationCount) -> Result<(), LedgerError>;
}

impl UsageLedger {
    /// Bind the original host before sharing this ledger with numerical workers.
    /// Weak ownership avoids a cycle between a root and its numerical ledger.
    /// Fixed-grant and asynchronous hosts can leave this port unbound and handle
    /// `MoreGrantRequired` at their own asynchronous execution boundary.
    pub fn with_grant_renewal(mut self, renewal: Weak<dyn GrantRenewal>) -> Self {
        self.renewal = Some(renewal);
        self
    }

    pub(super) fn with_renewal<T>(
        &self,
        mut admit: impl FnMut() -> Result<T, LedgerError>,
    ) -> Result<T, LedgerError> {
        loop {
            let required = match admit() {
                Err(error @ LedgerError::MoreGrantRequired { required, .. }) => {
                    let Some(renewal) = &self.renewal else {
                        return Err(error);
                    };
                    renewal
                        .upgrade()
                        .ok_or(LedgerError::Closed)?
                        .request_grant(required)?;
                    required
                }
                result => return result,
            };
            let granted = self.snapshot()?.granted;
            if granted < required {
                // A partial grant is a refusal for this batch, not permission
                // to execute it or spin on an unchanged upstream response.
                return Err(LedgerError::MoreGrantRequired { required, granted });
            }
            // Concurrent batches may have reserved more capacity while the host
            // was awaited. Revalidate the complete ledger before any effect.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CompletedUsage, UsageCategory};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, OnceLock,
    };

    struct Host {
        ledger: OnceLock<Weak<UsageLedger>>,
        calls: AtomicUsize,
        partial: bool,
    }
    impl GrantRenewal for Host {
        fn request_grant(&self, required: OperationCount) -> Result<(), LedgerError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            if !self.partial {
                self.ledger
                    .get()
                    .unwrap()
                    .upgrade()
                    .unwrap()
                    .update_grant(required)?;
            }
            Ok(())
        }
    }
    fn count(value: u64) -> OperationCount {
        OperationCount::from_operations(value)
    }
    fn fixture(partial: bool) -> (Arc<UsageLedger>, Arc<Host>) {
        let host = Arc::new(Host {
            ledger: OnceLock::new(),
            calls: AtomicUsize::new(0),
            partial,
        });
        let port: Arc<dyn GrantRenewal> = host.clone();
        let ledger = Arc::new(
            UsageLedger::new(count(50), count(2))
                .unwrap()
                .with_grant_renewal(Arc::downgrade(&port)),
        );
        host.ledger.set(Arc::downgrade(&ledger)).ok().unwrap();
        (ledger, host)
    }
    #[test]
    fn local_and_child_batches_acquire_capacity_before_reserving_effects() {
        let (ledger, host) = fixture(false);
        let mut local = ledger.reserve(UsageCategory::Assembly, count(4)).unwrap();
        let mut child = ledger.delegate(count(10)).unwrap();
        child.increase_capacity(count(20)).unwrap();
        assert_eq!(host.calls.load(Ordering::SeqCst), 3);
        assert_eq!(ledger.snapshot().unwrap().reserved, count(24));
        assert!(matches!(
            ledger.reserve(UsageCategory::Solve, count(27)),
            Err(LedgerError::BudgetExceeded { .. })
        ));
        assert_eq!(host.calls.load(Ordering::SeqCst), 3);
        let completed = CompletedUsage {
            solve: count(5),
            ..Default::default()
        };
        child.finish(completed).unwrap();
        local.complete(count(4)).unwrap();
        drop(local);
        assert_eq!(
            ledger.finish().unwrap(),
            CompletedUsage {
                assembly: count(4),
                ..completed
            }
        );
        assert!(matches!(
            ledger.reserve(UsageCategory::Solve, count(1)),
            Err(LedgerError::Closed)
        ));
    }
    #[test]
    fn phase_grant_coalesces_renewal_without_reserving_or_charging_primitive_work() {
        for partial in [false, true] {
            let (ledger, host) = fixture(partial);
            let before = ledger.snapshot().unwrap();
            let admission = ledger.ensure_capacity(count(30));
            assert_eq!(host.calls.load(Ordering::SeqCst), 1);
            if partial {
                assert_eq!(
                    admission,
                    Err(LedgerError::MoreGrantRequired {
                        required: count(30),
                        granted: count(2),
                    })
                );
                assert_eq!(ledger.snapshot().unwrap(), before);
            } else {
                admission.unwrap();
                let admitted = ledger.snapshot().unwrap();
                assert_eq!(admitted.budget, before.budget);
                assert_eq!(admitted.granted, count(30));
                assert_eq!(admitted.completed, before.completed);
                assert_eq!(admitted.reserved, before.reserved);
                for amount in [4, 6, 8] {
                    let mut batch = ledger
                        .reserve(UsageCategory::Assembly, count(amount))
                        .unwrap();
                    batch.complete(count(amount)).unwrap();
                }
                assert_eq!(host.calls.load(Ordering::SeqCst), 1);
                assert_eq!(ledger.snapshot().unwrap().completed.assembly, count(18));
                assert_eq!(ledger.snapshot().unwrap().reserved, count(0));
            }
            let stable = ledger.snapshot().unwrap();
            let required = stable
                .completed
                .total()
                .unwrap()
                .checked_add(count(51))
                .unwrap();
            assert_eq!(
                ledger.ensure_capacity(count(51)),
                Err(LedgerError::BudgetExceeded {
                    required,
                    budget: count(50),
                })
            );
            assert_eq!(host.calls.load(Ordering::SeqCst), 1);
            assert_eq!(ledger.snapshot().unwrap(), stable);
            let completed = ledger.finish().unwrap();
            assert_eq!(completed.assembly, count(if partial { 0 } else { 18 }));
            assert_eq!(ledger.ensure_capacity(count(0)), Err(LedgerError::Closed));
        }
    }

    #[test]
    fn partial_or_lost_host_does_not_admit_work_or_retry_unchanged_grants() {
        let (ledger, host) = fixture(true);
        let before = ledger.snapshot().unwrap();
        assert!(matches!(
            ledger.reserve(UsageCategory::Solve, count(3)),
            Err(LedgerError::MoreGrantRequired { .. })
        ));
        assert_eq!(host.calls.load(Ordering::SeqCst), 1);
        assert_eq!(ledger.snapshot().unwrap(), before);
        drop(host);
        assert!(matches!(
            ledger.reserve(UsageCategory::Solve, count(3)),
            Err(LedgerError::Closed)
        ));
        assert_eq!(ledger.snapshot().unwrap(), before);
    }
}
