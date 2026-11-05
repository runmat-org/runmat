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
            closed: false,
        })
    }
}

/// One child owns a disjoint portion of the parent's operation grant. This
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
    closed: bool,
}

impl DelegatedReservation {
    pub fn capacity(&self) -> OperationCount {
        self.maximum
    }

    pub fn completed(&self) -> CompletedUsage {
        self.completed
    }

    /// Increase the child's absolute capacity before publishing a renewal to
    /// it. Only the increment is reserved; replaying the same renewal is inert.
    /// Completed child work already belongs to the parent and remains charged.
    /// The host serializes this owner with authenticated reports and renewals.
    pub fn increase_capacity(&mut self, maximum: OperationCount) -> Result<(), LedgerError> {
        if self.closed {
            return Err(LedgerError::Closed);
        }
        let additional = maximum
            .operations()
            .checked_sub(self.maximum.operations())
            .map(OperationCount::from_operations)
            .ok_or(LedgerError::InvalidGrant)?;
        self.ledger.reserve_additional_capacity(additional)?;
        self.maximum = maximum;
        Ok(())
    }

    /// Raise the child's capacity to at least `required`, plus headroom, so a
    /// child that reserves small counts per element does not wait on its parent
    /// for every element.
    ///
    /// Headroom comes only from operations the parent already holds as grant
    /// and has not reserved, and it never asks the host for more grant. It is
    /// at most the larger of the child's current capacity and the exact
    /// increment, so incremental requests can double capacity. It is also at
    /// most half of the remaining free grant, so sibling children keep the
    /// rest. Headroom is not charged: `finish` returns unused capacity to the
    /// parent.
    pub fn grow_capacity(&mut self, required: OperationCount) -> Result<(), LedgerError> {
        if self.closed {
            return Err(LedgerError::Closed);
        }
        if required <= self.maximum {
            return Ok(());
        }
        let exact = required.operations() - self.maximum.operations();
        let headroom = {
            let state = self.ledger.lock()?;
            let committed = state.completed.total()?.checked_add(state.reserved)?;
            let free = state
                .grant
                .operations()
                .saturating_sub(committed.operations());
            let spare = free.saturating_sub(exact) / 2;
            self.maximum.operations().max(exact).min(spare)
        };
        if headroom > 0 {
            let additional = exact
                .checked_add(headroom)
                .map(OperationCount::from_operations)
                .ok_or(LedgerError::CountOverflow)?;
            // Reserve against the grant already held, without renewal. If a
            // sibling took the free grant since it was read, fall back to the
            // exact increase, which may renew upstream as before.
            let grown = self.maximum.checked_add(additional)?;
            if self.ledger.try_reserve_capacity(additional, false).is_ok() {
                self.maximum = grown;
                return Ok(());
            }
        }
        self.increase_capacity(required)
    }

    /// Merge authenticated cumulative completed work into the parent. Validate
    /// every category and all arithmetic before changing either owner's state.
    pub fn report(&mut self, completed: CompletedUsage) -> Result<(), LedgerError> {
        if self.closed {
            return Err(LedgerError::Closed);
        }
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
    /// remains in the parent's ledger. Rejected final reports retain both the
    /// hold and this owner so the authenticated host can retry reconciliation.
    pub fn finish(&mut self, completed: CompletedUsage) -> Result<(), LedgerError> {
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
        self.closed = true;
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
        assert_eq!(child.finish(swapped), Err(LedgerError::NonMonotonicUsage));
        assert_eq!(ledger.snapshot().unwrap(), before);
        child.finish(initial).unwrap();
        let closed = ledger.snapshot().unwrap();
        assert_eq!(child.finish(initial), Err(LedgerError::Closed));
        assert_eq!(child.report(initial), Err(LedgerError::Closed));
        assert_eq!(
            child.increase_capacity(count(u64::MAX)),
            Err(LedgerError::Closed)
        );
        assert_eq!(ledger.snapshot().unwrap(), closed);
        assert_eq!(ledger.finish().unwrap(), initial);
    }

    #[test]
    fn child_renewals_reserve_only_the_increment_against_shared_capacity() {
        let ledger = Arc::new(UsageLedger::new(count(100), count(50)).unwrap());
        let mut child = ledger.delegate(count(20)).unwrap();
        let usage = CompletedUsage {
            solve: count(10),
            ..Default::default()
        };
        child.report(usage).unwrap();
        let local = ledger.reserve(UsageCategory::Assembly, count(30)).unwrap();
        let before = ledger.snapshot().unwrap();
        assert_eq!(
            child.increase_capacity(count(30)),
            Err(LedgerError::MoreGrantRequired {
                required: count(60),
                granted: count(50),
            })
        );
        assert_eq!(ledger.snapshot().unwrap(), before);
        assert_eq!(child.capacity(), count(20));
        ledger.update_grant(count(100)).unwrap();
        child.increase_capacity(count(70)).unwrap();
        let renewed = ledger.snapshot().unwrap();
        child.increase_capacity(count(70)).unwrap();
        assert_eq!(ledger.snapshot().unwrap(), renewed);
        assert_eq!(
            child.increase_capacity(count(69)),
            Err(LedgerError::InvalidGrant)
        );
        assert!(matches!(
            child.increase_capacity(count(71)),
            Err(LedgerError::BudgetExceeded { .. })
        ));
        assert_eq!(ledger.snapshot().unwrap(), renewed);
        child.finish(usage).unwrap();
        drop(local);
        assert_eq!(ledger.finish().unwrap(), usage);
    }

    #[test]
    fn growing_child_capacity_doubles_from_free_grant_and_returns_headroom_at_finish() {
        let ledger = Arc::new(UsageLedger::new(count(10_000), count(1_000)).unwrap());
        let mut child = ledger.delegate(count(0)).unwrap();
        // Per-element reserves of 10 ops: one renewal per element without headroom.
        let mut renewals = 0;
        for element in 1..=60u64 {
            let required = count(element * 10);
            if required > child.capacity() {
                child.grow_capacity(required).unwrap();
                renewals += 1;
            }
            assert!(child.capacity() >= required);
            // Never more than half of the free grant beyond the exact need.
            assert!(ledger.snapshot().unwrap().reserved <= count(1_000));
        }
        assert!(renewals <= 8, "{renewals} renewals for 60 elements");
        let before = ledger.snapshot().unwrap();
        // Requests at or below the current capacity change nothing.
        child.grow_capacity(count(1)).unwrap();
        assert_eq!(ledger.snapshot().unwrap(), before);
        let usage = CompletedUsage {
            assembly: count(600),
            ..Default::default()
        };
        child.finish(usage).unwrap();
        let after = ledger.snapshot().unwrap();
        assert_eq!(after.reserved, count(0));
        assert_eq!(after.completed, usage);
        assert_eq!(ledger.finish().unwrap(), usage);
    }

    #[test]
    fn growing_child_capacity_leaves_siblings_half_and_never_renews_for_headroom() {
        let ledger = Arc::new(UsageLedger::new(count(1_000), count(100)).unwrap());
        let mut first = ledger.delegate(count(0)).unwrap();
        let mut second = ledger.delegate(count(0)).unwrap();
        first.grow_capacity(count(10)).unwrap();
        // 90 free after the exact 10; headroom is min(max(0, 10), 45) = 10.
        assert_eq!(first.capacity(), count(20));
        first.grow_capacity(count(60)).unwrap();
        // 80 free, exact 40, spare 20; headroom min(max(20, 40), 20) = 20.
        assert_eq!(first.capacity(), count(80));
        // The sibling still gets its exact need from the remaining 20.
        second.grow_capacity(count(20)).unwrap();
        assert_eq!(second.capacity(), count(20));
        // With no free grant, growth is exact and asks upstream as before.
        assert_eq!(
            second.grow_capacity(count(21)),
            Err(LedgerError::MoreGrantRequired {
                required: count(101),
                granted: count(100),
            })
        );
        assert_eq!(second.capacity(), count(20));
        first.finish(CompletedUsage::default()).unwrap();
        second.finish(CompletedUsage::default()).unwrap();
        assert_eq!(ledger.snapshot().unwrap().reserved, count(0));
    }

    #[test]
    fn lost_or_invalidly_closed_child_never_releases_live_spending_authority() {
        for invalid_close in [false, true] {
            let ledger = Arc::new(UsageLedger::new(count(10), count(10)).unwrap());
            let mut child = ledger.delegate(count(10)).unwrap();
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
