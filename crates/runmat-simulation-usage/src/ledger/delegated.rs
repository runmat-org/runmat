//! Exclusive parent-side ownership of a separately executing child's grant.
use super::{LedgerError, UsageLedger};
use crate::{CompletedUsage, OperationCount, UsageCategory};
use std::sync::Arc;

impl UsageLedger {
    /// Set aside disjoint capacity before dispatching a child. The trusted host
    /// binds this reservation to the authenticated child assignment; a count
    /// sent over a transport is not, by itself, execution authority.
    ///
    /// The parent retains this owner until physical retirement is established.
    /// Dropping it cannot establish that proof and does not release capacity.
    pub fn delegate(
        self: &Arc<Self>,
        maximum: OperationCount,
    ) -> Result<DelegatedReservation, LedgerError> {
        self.reserve_capacity(maximum)?;
        Ok(DelegatedReservation {
            ledger: Arc::clone(self),
            maximum,
            completed: CompletedUsage::default(),
        })
    }
}

/// One child owns this fixed portion of the parent's operation grant. This
/// owner cannot be cloned or serialized. Reports are cumulative and preserve
/// each category; replaying a report adds no work or grant.
///
/// This type owns no network, child process or recovery policy. The host must
/// authenticate reports and establish physical retirement before `finish`.
/// It must retain this owner across cancellation and communication failure.
#[must_use = "retain delegated capacity until the child is physically retired"]
pub struct DelegatedReservation {
    ledger: Arc<UsageLedger>,
    maximum: OperationCount,
    completed: CompletedUsage,
}

impl DelegatedReservation {
    pub fn capacity(&self) -> OperationCount {
        self.maximum
    }

    pub fn completed(&self) -> CompletedUsage {
        self.completed
    }

    /// Merge authenticated cumulative completed work into the parent. Validate
    /// every category and all arithmetic before changing either owner's state.
    pub fn report(&mut self, completed: CompletedUsage) -> Result<(), LedgerError> {
        let total = completed.total()?;
        if total > self.maximum {
            return Err(LedgerError::InvalidReservation);
        }
        let mut state = self.ledger.lock()?;
        if state.closed {
            return Err(LedgerError::Closed);
        }
        let mut merged = state.completed;
        let mut delta = OperationCount::ZERO;
        for (category, previous, current) in [
            (
                UsageCategory::Meshing,
                self.completed.meshing,
                completed.meshing,
            ),
            (
                UsageCategory::Assembly,
                self.completed.assembly,
                completed.assembly,
            ),
            (
                UsageCategory::Factorization,
                self.completed.factorization,
                completed.factorization,
            ),
            (UsageCategory::Solve, self.completed.solve, completed.solve),
            (UsageCategory::Other, self.completed.other, completed.other),
        ] {
            let additional = current
                .operations()
                .checked_sub(previous.operations())
                .map(OperationCount::from_operations)
                .ok_or(LedgerError::NonMonotonicUsage)?;
            merged = merged.with_completed(category, additional)?;
            delta = delta.checked_add(additional)?;
        }
        let reserved = state
            .reserved
            .operations()
            .checked_sub(delta.operations())
            .map(OperationCount::from_operations)
            .ok_or(LedgerError::InvalidState)?;
        state.completed = merged;
        state.reserved = reserved;
        self.completed = completed;
        Ok(())
    }

    /// Close after the host has verified the child's physical retirement and
    /// final cumulative report. Release only unused capacity; completed work
    /// remains in the parent's ledger. Rejected final reports retain the hold.
    pub fn finish(mut self, completed: CompletedUsage) -> Result<(), LedgerError> {
        self.report(completed)?;
        let unused = self
            .maximum
            .operations()
            .checked_sub(self.completed.total()?.operations())
            .ok_or(LedgerError::InvalidState)?;
        let mut state = self.ledger.lock()?;
        let reserved = state
            .reserved
            .operations()
            .checked_sub(unused)
            .map(OperationCount::from_operations)
            .ok_or(LedgerError::InvalidState)?;
        let active = state
            .active
            .checked_sub(1)
            .ok_or(LedgerError::InvalidState)?;
        state.reserved = reserved;
        state.active = active;
        Ok(())
    }
}

// Deliberately no Drop release: loss of this host-side owner says nothing about
// whether its child can still execute. The outstanding hold prevents a new
// owner from spending that capacity or the root from certifying final usage.

#[cfg(test)]
mod tests {
    use super::*;

    fn count(value: u64) -> OperationCount {
        OperationCount::from_operations(value)
    }

    #[test]
    fn child_and_local_work_conserve_one_grant_with_cumulative_reports() {
        let ledger = Arc::new(UsageLedger::new(count(100), count(90)).unwrap());
        let mut child = ledger.delegate(count(60)).unwrap();
        assert_eq!(child.capacity(), count(60));
        assert!(matches!(
            ledger.delegate(count(40)),
            Err(LedgerError::MoreGrantRequired { .. })
        ));
        let usage = CompletedUsage {
            meshing: count(10),
            solve: count(20),
            ..Default::default()
        };
        child.report(usage).unwrap();
        child.report(usage).unwrap();
        assert_eq!(child.completed(), usage);
        assert_eq!(ledger.snapshot().unwrap().reserved, count(30));
        assert!(matches!(
            ledger.reserve(UsageCategory::Assembly, count(40)),
            Err(LedgerError::MoreGrantRequired { .. })
        ));
        let mut local = ledger.reserve(UsageCategory::Assembly, count(30)).unwrap();
        assert_eq!(ledger.finish(), Err(LedgerError::OutstandingWork));
        child.finish(usage).unwrap();
        local.complete(count(20)).unwrap();
        drop(local);
        assert_eq!(
            ledger.finish().unwrap(),
            CompletedUsage {
                assembly: count(20),
                ..usage
            }
        );
    }

    #[test]
    fn invalid_reports_are_atomic_and_each_category_is_monotonic() {
        let ledger = Arc::new(UsageLedger::new(count(u64::MAX), count(u64::MAX)).unwrap());
        let mut child = ledger.delegate(count(u64::MAX)).unwrap();
        let initial = CompletedUsage {
            solve: count(5),
            ..Default::default()
        };
        child.report(initial).unwrap();
        let before = ledger.snapshot().unwrap();
        let swapped = CompletedUsage {
            meshing: count(8),
            solve: count(4),
            ..Default::default()
        };
        assert_eq!(child.report(swapped), Err(LedgerError::NonMonotonicUsage));
        let overflow = CompletedUsage {
            meshing: count(u64::MAX),
            solve: count(1),
            ..Default::default()
        };
        assert_eq!(child.report(overflow), Err(LedgerError::CountOverflow));
        assert_eq!(ledger.snapshot().unwrap(), before);
        assert_eq!(child.completed(), initial);
        child.finish(initial).unwrap();
        assert_eq!(ledger.finish().unwrap(), initial);
    }

    #[test]
    fn lost_or_invalidly_closed_child_never_releases_live_spending_authority() {
        for invalid_close in [false, true] {
            let ledger = Arc::new(UsageLedger::new(count(10), count(10)).unwrap());
            let child = ledger.delegate(count(10)).unwrap();
            if invalid_close {
                assert_eq!(
                    child.finish(CompletedUsage {
                        solve: count(11),
                        ..Default::default()
                    }),
                    Err(LedgerError::InvalidReservation)
                );
            } else {
                drop(child);
            }
            assert_eq!(ledger.snapshot().unwrap().reserved, count(10));
            assert!(matches!(
                ledger.delegate(count(1)),
                Err(LedgerError::BudgetExceeded { .. })
            ));
            assert_eq!(ledger.finish(), Err(LedgerError::OutstandingWork));
        }
    }
}
