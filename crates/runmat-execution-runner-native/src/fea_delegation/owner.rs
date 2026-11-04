//! Parent-side ownership of one authenticated child's operation capacity.
use super::{Source, WorkerFeaGrant};
use runmat_execution::{resource::ResourceRequest, ProgramExecutionAssignment};
use runmat_runtime::analysis::fea_run_admission::{FeaRunBudget, FeaRunDenied, FeaRunUsage};
use runmat_simulation_usage::{CompletedUsage, DelegatedReservation, LedgerError, OperationCount};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    Prepared,
    MayExecute,
    Reported,
    Retired,
}
struct State {
    reservation: Option<DelegatedReservation>,
    phase: Phase,
}
pub(crate) struct ChildGrant {
    source: Source,
    assignment: ProgramExecutionAssignment,
    initial: FeaRunBudget,
    state: Mutex<State>,
    capacity: AtomicU64,
    retired: AtomicBool,
}
impl ChildGrant {
    pub(crate) fn new(
        source: &Source,
        assignment: &ProgramExecutionAssignment,
        resources: &ResourceRequest,
    ) -> Result<(WorkerFeaGrant, Arc<Self>), String> {
        assignment.validate().map_err(|error| error.to_string())?;
        resources.validate().map_err(|error| error.to_string())?;
        assignment
            .resources
            .validate_for_request(resources)
            .map_err(|error| error.to_string())?;
        // Start with no spending authority. The child's first numerical batch
        // requests its actual required capacity through the private channel.
        let mut reserved = source
            .delegate(OperationCount::ZERO)
            .map_err(|error| error.to_string())?;
        let grant = WorkerFeaGrant {
            assignment: assignment.clone(),
            resources: resources.clone(),
            parent: reserved.delegation,
        };
        if let Err(error) = grant.validate(assignment, resources) {
            // No launch descriptor has left this constructor.
            let _ = reserved.reservation.finish(CompletedUsage::default());
            return Err(error);
        }
        let owner = Arc::new(Self {
            source: Arc::clone(source),
            assignment: assignment.clone(),
            initial: grant.parent.budget,
            state: Mutex::new(State {
                reservation: Some(reserved.reservation),
                phase: Phase::Prepared,
            }),
            capacity: AtomicU64::new(0),
            retired: AtomicBool::new(false),
        });
        Ok((grant, owner))
    }

    pub(crate) fn validate_assignment(
        &self,
        assignment: &ProgramExecutionAssignment,
    ) -> Result<(), FeaRunDenied> {
        if assignment != &self.assignment {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        Ok(())
    }

    pub(crate) fn budget(&self) -> Result<FeaRunBudget, FeaRunDenied> {
        let root = self.source.current_budget()?;
        if root.budget_ops != self.initial.budget_ops
            || root.heartbeat_millis != self.initial.heartbeat_millis
            || root.expires_at_unix_millis < self.initial.expires_at_unix_millis
        {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        if self.retired.load(Ordering::Acquire) {
            return Err(LedgerError::Closed.into());
        }
        // Cancellation/deadline checks cannot wait behind a renewal transport.
        Ok(FeaRunBudget {
            granted_ops: OperationCount::from_operations(self.capacity.load(Ordering::Acquire)),
            ..root
        })
    }

    /// Set before attempting spawn: a spawn error can still leave a live tree.
    pub(crate) fn may_execute(&self) -> Result<(), FeaRunDenied> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
        if state.phase != Phase::Prepared {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state.phase = Phase::MayExecute;
        Ok(())
    }

    pub(crate) fn report(
        &self,
        usage: &FeaRunUsage,
        required: Option<OperationCount>,
    ) -> Result<FeaRunBudget, FeaRunDenied> {
        self.source.current_budget()?;
        {
            let mut state = self
                .state
                .lock()
                .map_err(|_| FeaRunDenied::LicenseInvalid)?;
            if state.phase != Phase::MayExecute {
                return Err(FeaRunDenied::LicenseInvalid);
            }
            let reservation = state.reservation.as_mut().ok_or(LedgerError::Closed)?;
            reservation.report(usage.completed_ops)?;
            if let Some(required) = required {
                if required > reservation.capacity() {
                    reservation.increase_capacity(required)?;
                }
            }
            self.capacity
                .store(reservation.capacity().operations(), Ordering::Release);
        }
        self.budget()
    }

    pub(crate) fn settled(&self, usage: &FeaRunUsage) -> Result<(), FeaRunDenied> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
        if state.phase != Phase::MayExecute {
            return Err(FeaRunDenied::LicenseInvalid);
        }
        state
            .reservation
            .as_mut()
            .ok_or(LedgerError::Closed)?
            .report(usage.completed_ops)?;
        state.phase = Phase::Reported;
        Ok(())
    }

    /// Called only after the process owner establishes full-tree retirement.
    /// A successful result additionally requires the final cumulative report.
    pub(crate) fn retire(&self, successful: bool) -> Result<(), FeaRunDenied> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| FeaRunDenied::LicenseInvalid)?;
        if state.phase == Phase::Retired {
            return Ok(());
        }
        if successful && state.phase != Phase::Reported {
            return Err(LedgerError::OutstandingWork.into());
        }
        let reservation = state.reservation.as_mut().ok_or(LedgerError::Closed)?;
        let completed = reservation.completed();
        reservation.finish(completed)?;
        state.reservation.take();
        state.phase = Phase::Retired;
        self.retired.store(true, Ordering::Release);
        Ok(())
    }
}
impl Drop for ChildGrant {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(|error| error.into_inner());
        if state.phase == Phase::Prepared {
            if let Some(mut reservation) = state.reservation.take() {
                // A never-attempted launch cannot have executed numerical work.
                let completed = reservation.completed();
                let _ = reservation.finish(completed);
            }
        }
        // Other drops do not prove retirement. DelegatedReservation preserves
        // the outstanding hold and prevents root usage certification.
    }
}
