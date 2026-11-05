use crate::{CompletedUsage, CountOverflow, OperationCount, UsageCategory};
use std::fmt;
use std::sync::{Mutex, MutexGuard, Weak};

mod delegated;
mod renewal;
pub use delegated::DelegatedReservation;
pub use renewal::GrantRenewal;

/// One run's admitted operation budget. Storage is fixed-size; reserving a
/// numerical batch allocates no heap memory and does not lock per operation.
///
/// Callers reserve before numerical execution, report completed portions at the
/// kernel's defined counting boundaries, and drop unused capacity on cancellation.
/// Network grant renewal and account policy belong to the host admission owner.
pub struct UsageLedger {
    state: Mutex<State>,
    renewal: Option<Weak<dyn GrantRenewal>>,
}

struct State {
    budget: OperationCount,
    grant: OperationCount,
    completed: CompletedUsage,
    reserved: OperationCount,
    active: usize,
    closed: bool,
    invalid: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LedgerSnapshot {
    pub budget: OperationCount,
    pub granted: OperationCount,
    pub completed: CompletedUsage,
    pub reserved: OperationCount,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LedgerError {
    CountOverflow,
    BudgetExceeded {
        required: OperationCount,
        budget: OperationCount,
    },
    MoreGrantRequired {
        required: OperationCount,
        granted: OperationCount,
    },
    InvalidGrant,
    InvalidReservation,
    NonMonotonicUsage,
    OutstandingWork,
    Closed,
    InvalidState,
    GrantUnavailable,
}

impl From<CountOverflow> for LedgerError {
    fn from(_: CountOverflow) -> Self {
        Self::CountOverflow
    }
}

impl fmt::Display for LedgerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CountOverflow => f.write_str("simulation operation count overflow"),
            Self::BudgetExceeded { required, budget } => write!(f, "simulation work needs {required} operations, exceeding spending cap {budget}"),
            Self::MoreGrantRequired { required, granted } => write!(f, "simulation work needs a total grant of {required} operations; {granted} are granted"),
            Self::InvalidGrant => f.write_str("simulation grant must be monotonic and within the spending cap"),
            Self::InvalidReservation => f.write_str("completed simulation work exceeds its reservation"),
            Self::NonMonotonicUsage => f.write_str("completed simulation usage cannot decrease in any category"),
            Self::OutstandingWork => f.write_str("simulation work reservations are still active"),
            Self::Closed => f.write_str("simulation usage ledger is closed"),
            Self::InvalidState => f.write_str("simulation usage ledger state is invalid"),
            Self::GrantUnavailable => f.write_str("the simulation host could not renew the operation grant"),
        }
    }
}
impl std::error::Error for LedgerError {}

impl UsageLedger {
    pub fn new(budget: OperationCount, grant: OperationCount) -> Result<Self, LedgerError> {
        if grant > budget {
            return Err(LedgerError::InvalidGrant);
        }
        Ok(Self {
            renewal: None,
            state: Mutex::new(State {
                budget,
                grant,
                completed: CompletedUsage::default(),
                reserved: OperationCount::ZERO,
                active: 0,
                closed: false,
                invalid: false,
            }),
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>, LedgerError> {
        let state = self.state.lock().map_err(|_| LedgerError::InvalidState)?;
        if state.invalid {
            return Err(LedgerError::InvalidState);
        }
        Ok(state)
    }

    /// A server response supplies an absolute cumulative grant, not an increment.
    /// Replaying the same grant does not reserve or authorize additional work.
    pub fn update_grant(&self, grant: OperationCount) -> Result<(), LedgerError> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(LedgerError::Closed);
        }
        if grant < state.grant || grant > state.budget {
            return Err(LedgerError::InvalidGrant);
        }
        state.grant = grant;
        Ok(())
    }

    pub fn reserve(
        &self,
        category: UsageCategory,
        maximum: OperationCount,
    ) -> Result<WorkReservation<'_>, LedgerError> {
        self.reserve_capacity(maximum)?;
        Ok(WorkReservation {
            ledger: self,
            category,
            remaining: maximum,
        })
    }

    fn reserve_capacity(&self, maximum: OperationCount) -> Result<(), LedgerError> {
        self.with_renewal(|| self.try_reserve_capacity(maximum, true))
    }

    fn reserve_additional_capacity(&self, additional: OperationCount) -> Result<(), LedgerError> {
        self.with_renewal(|| self.try_reserve_capacity(additional, false))
    }

    fn try_reserve_capacity(
        &self,
        maximum: OperationCount,
        new_owner: bool,
    ) -> Result<(), LedgerError> {
        let mut state = self.lock()?;
        if state.closed {
            return Err(LedgerError::Closed);
        }
        let reserved = state.reserved.checked_add(maximum)?;
        let required = state.completed.total()?.checked_add(reserved)?;
        if required > state.budget {
            return Err(LedgerError::BudgetExceeded {
                required,
                budget: state.budget,
            });
        }
        if required > state.grant {
            return Err(LedgerError::MoreGrantRequired {
                required,
                granted: state.grant,
            });
        }
        let active = state
            .active
            .checked_add(usize::from(new_owner))
            .ok_or(LedgerError::CountOverflow)?;
        state.reserved = reserved;
        state.active = active;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<LedgerSnapshot, LedgerError> {
        let state = self.lock()?;
        Ok(LedgerSnapshot {
            budget: state.budget,
            granted: state.grant,
            completed: state.completed,
            reserved: state.reserved,
        })
    }

    /// Capacity for a synchronous phase whose caller owns the ledger
    /// exclusively. Quoting neither reserves nor completes numerical work.
    pub fn exclusive_work_requirement(
        &self,
        additional: OperationCount,
    ) -> Result<OperationCount, LedgerError> {
        let state = self.lock()?;
        if state.closed {
            return Err(LedgerError::Closed);
        }
        if state.active != 0 {
            return Err(LedgerError::OutstandingWork);
        }
        let required = state.completed.total()?.checked_add(additional)?;
        if required > state.budget {
            return Err(LedgerError::BudgetExceeded {
                required,
                budget: state.budget,
            });
        }
        Ok(required)
    }

    /// Final usage is available only after every numerical owner relinquishes its
    /// reservation. Closing is idempotent and prevents later numerical admission.
    pub fn finish(&self) -> Result<CompletedUsage, LedgerError> {
        let mut state = self.lock()?;
        if state.active != 0 {
            return Err(LedgerError::OutstandingWork);
        }
        state.closed = true;
        Ok(state.completed)
    }
}

/// Exclusive ownership of one admitted numerical batch. It cannot be cloned.
/// Dropping releases unused grant; it does not label reserved work as completed.
#[must_use = "numerical work must retain its reservation until completion"]
pub struct WorkReservation<'a> {
    ledger: &'a UsageLedger,
    category: UsageCategory,
    remaining: OperationCount,
}

impl WorkReservation<'_> {
    /// Record an additional completed portion within this batch. Kernel callers
    /// must report any completed prefix before returning a cancellation/error.
    pub fn complete(&mut self, completed: OperationCount) -> Result<(), LedgerError> {
        let remaining = self
            .remaining
            .operations()
            .checked_sub(completed.operations())
            .ok_or(LedgerError::InvalidReservation)?;
        let mut state = self.ledger.lock()?;
        if state.closed {
            return Err(LedgerError::Closed);
        }
        let usage = state.completed.with_completed(self.category, completed)?;
        let reserved = state
            .reserved
            .operations()
            .checked_sub(completed.operations())
            .ok_or(LedgerError::InvalidState)?;
        state.completed = usage;
        state.reserved = OperationCount::from_operations(reserved);
        self.remaining = OperationCount::from_operations(remaining);
        Ok(())
    }
}

impl Drop for WorkReservation<'_> {
    fn drop(&mut self) {
        let mut state = match self.ledger.state.lock() {
            Ok(state) => state,
            Err(poison) => {
                let mut state = poison.into_inner();
                state.invalid = true;
                state
            }
        };
        let reserved = state
            .reserved
            .operations()
            .checked_sub(self.remaining.operations());
        let active = state.active.checked_sub(1);
        match (reserved, active) {
            (Some(reserved), Some(active)) => {
                state.reserved = OperationCount::from_operations(reserved);
                state.active = active;
            }
            _ => state.invalid = true,
        }
    }
}

#[cfg(test)]
mod tests;
